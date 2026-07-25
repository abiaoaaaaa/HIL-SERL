import glob
import os
import pickle as pkl
import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.training import checkpoints
import numpy as np
import optax
from tqdm import tqdm
from absl import app, flags

from serl_launcher.data.data_store import ReplayBuffer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.networks.reward_classifier import create_classifier

from experiments.mappings import CONFIG_MAPPING


FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("num_epochs", 150, "Number of training epochs.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_float("train_ratio", 0.67, "Ratio of training data (default 0.67 for 2:1 train/test split).")


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    # 使用collect_classifier_data=True，匹配采集数据时的图像键
    env = config.get_environment(fake_env=True, save_video=False, classifier=False, collect_classifier_data=True)

    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)

    # ==================== 数据加载 ====================
    print("=" * 80)
    print("加载数据集...")
    print("=" * 80)

    # 加载所有成功样本
    success_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data", "*success*.pkl"))
    all_pos_transitions = []
    for path in success_paths:
        success_data = pkl.load(open(path, "rb"))
        for trans in success_data:
            if "images" in trans['observations'].keys():
                continue
            trans["labels"] = 1
            trans['actions'] = env.action_space.sample()
            all_pos_transitions.append(trans)

    # 加载所有失败样本
    failure_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data", "*failure*.pkl"))
    all_neg_transitions = []
    for path in failure_paths:
        failure_data = pkl.load(open(path, "rb"))
        for trans in failure_data:
            if "images" in trans['observations'].keys():
                continue
            trans["labels"] = 0
            trans['actions'] = env.action_space.sample()
            all_neg_transitions.append(trans)

    print(f"总成功样本: {len(all_pos_transitions)}")
    print(f"总失败样本: {len(all_neg_transitions)}")

    # ==================== 平衡数据集 ====================
    # 将正负样本数量平衡到相同
    rng = np.random.default_rng(42)
    min_count = min(len(all_pos_transitions), len(all_neg_transitions))

    sampled_pos = rng.choice(all_pos_transitions, size=min_count, replace=False)
    sampled_neg = rng.choice(all_neg_transitions, size=min_count, replace=False)

    print(f"平衡后: 成功={len(sampled_pos)}, 失败={len(sampled_neg)}")

    # ==================== 划分训练集和测试集 (2:1) ====================
    print(f"\n按 {FLAGS.train_ratio:.0%}:{1-FLAGS.train_ratio:.0%} 划分训练集和测试集...")

    # 随机打乱并划分成功样本
    rng.shuffle(sampled_pos)
    n_train_pos = int(len(sampled_pos) * FLAGS.train_ratio)
    train_pos = sampled_pos[:n_train_pos]
    test_pos = sampled_pos[n_train_pos:]

    # 随机打乱并划分失败样本
    rng.shuffle(sampled_neg)
    n_train_neg = int(len(sampled_neg) * FLAGS.train_ratio)
    train_neg = sampled_neg[:n_train_neg]
    test_neg = sampled_neg[n_train_neg:]

    print(f"训练集: 成功={len(train_pos)}, 失败={len(train_neg)}, 总计={len(train_pos)+len(train_neg)}")
    print(f"测试集: 成功={len(test_pos)}, 失败={len(test_neg)}, 总计={len(test_pos)+len(test_neg)}")

    # ==================== 创建训练集 Buffer ====================
    train_pos_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=20000,
        include_label=True,
    )
    for trans in train_pos:
        train_pos_buffer.insert(trans)

    train_neg_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=50000,
        include_label=True,
    )
    for trans in train_neg:
        train_neg_buffer.insert(trans)

    pos_iterator = train_pos_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
        },
        device=sharding.replicate(),
    )

    neg_iterator = train_neg_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
        },
        device=sharding.replicate(),
    )

    # ==================== 创建测试集 Buffer ====================
    test_pos_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=20000,
        include_label=True,
    )
    for trans in test_pos:
        test_pos_buffer.insert(trans)

    test_neg_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=50000,
        include_label=True,
    )
    for trans in test_neg:
        test_neg_buffer.insert(trans)

    # ==================== 初始化分类器 ====================
    print("\n" + "=" * 80)
    print("初始化分类器...")
    print("=" * 80)

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    pos_sample = next(pos_iterator)
    neg_sample = next(neg_iterator)
    sample = concat_batches(pos_sample, neg_sample, axis=0)

    rng, key = jax.random.split(rng)
    classifier = create_classifier(key,
                                   sample["observations"],
                                   config.classifier_keys,
                                   )

    def data_augmentation_fn(rng, observations):
        for pixel_key in config.classifier_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    @jax.jit
    def train_step(state, batch, key):
        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params}, batch["observations"], rngs={"dropout": key}, train=True
            )
            return optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        logits = state.apply_fn(
            {"params": state.params}, batch["observations"], train=False, rngs={"dropout": key}
        )
        train_accuracy = jnp.mean((nn.sigmoid(logits) >= 0.5) == batch["labels"])

        return state.apply_gradients(grads=grads), loss, train_accuracy

    @jax.jit
    def eval_step(state, batch, key):
        """测试集评估"""
        logits = state.apply_fn(
            {"params": state.params}, batch["observations"], train=False, rngs={"dropout": key}
        )
        loss = optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()
        accuracy = jnp.mean((nn.sigmoid(logits) >= 0.5) == batch["labels"])
        return loss, accuracy, logits

    # ==================== 训练循环 ====================
    print("\n" + "=" * 80)
    print(f"开始训练 ({FLAGS.num_epochs} epochs)...")
    print("=" * 80)

    # 用于记录训练历史
    train_history = {
        'loss': [],
        'accuracy': [],
        'epoch': []
    }

    for epoch in tqdm(range(FLAGS.num_epochs)):
        # Sample equal number of positive and negative examples
        pos_sample = next(pos_iterator)
        neg_sample = next(neg_iterator)
        # Merge and create labels
        batch = concat_batches(
            pos_sample, neg_sample, axis=0
        )
        rng, key = jax.random.split(rng)
        obs = data_augmentation_fn(key, batch["observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "labels": batch["labels"][..., None],
            }
        )

        rng, key = jax.random.split(rng)
        classifier, train_loss, train_accuracy = train_step(classifier, batch, key)

        # 记录训练历史
        train_history['loss'].append(float(train_loss))
        train_history['accuracy'].append(float(train_accuracy))
        train_history['epoch'].append(epoch + 1)

        # 每10个epoch打印一次详细信息
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch: {epoch+1}/{FLAGS.num_epochs}, "
                f"Train Loss: {train_loss:.4f}, "
                f"Train Accuracy: {train_accuracy:.4f}"
            )

        # 第一个epoch打印数据分布信息
        if epoch == 0:
            print("\n📊 数据分布信息 (Epoch 1):")
            print(f"  - Batch size: {batch['observations']['state'].shape[0]}")
            print(f"  - State shape: {batch['observations']['state'].shape}")

            # 打印状态统计
            state_data = np.array(batch['observations']['state'])
            print(f"  - State 范围: [{state_data.min():.3f}, {state_data.max():.3f}]")
            print(f"  - State 均值: {state_data.mean():.3f} ± {state_data.std():.3f}")

            # 打印图像键
            image_keys = [k for k in batch['observations'].keys() if k != 'state']
            print(f"  - 图像键: {image_keys}")
            for img_key in image_keys:
                img_shape = batch['observations'][img_key].shape
                print(f"    - {img_key}: {img_shape}")

            # 打印标签分布
            labels = np.array(batch['labels'])
            pos_count = np.sum(labels == 1)
            neg_count = np.sum(labels == 0)
            print(f"  - 标签分布: 成功={pos_count}, 失败={neg_count}")
            print("")

    # ==================== 保存模型 ====================
    print("\n" + "=" * 80)
    print("保存模型...")
    print("=" * 80)

    checkpoints.save_checkpoint(
        os.path.join(os.getcwd(), "classifier_ckpt/"),
        classifier,
        step=FLAGS.num_epochs,
        overwrite=True,
    )
    print(f"✅ 模型已保存到: {os.path.join(os.getcwd(), 'classifier_ckpt/')}")

    # 打印训练历史总结
    print("\n📈 训练历史总结:")
    train_loss_arr = np.array(train_history['loss'])
    train_acc_arr = np.array(train_history['accuracy'])
    print(f"  - 初始 Loss: {train_loss_arr[0]:.4f}")
    print(f"  - 最终 Loss: {train_loss_arr[-1]:.4f}")
    print(f"  - Loss 下降: {train_loss_arr[0] - train_loss_arr[-1]:.4f}")
    print(f"  - 初始 Accuracy: {train_acc_arr[0]:.4f}")
    print(f"  - 最终 Accuracy: {train_acc_arr[-1]:.4f}")
    print(f"  - 最高 Accuracy: {train_acc_arr.max():.4f} (Epoch {train_history['epoch'][train_acc_arr.argmax()]})")
    print(f"  - 平均 Accuracy: {train_acc_arr.mean():.4f} ± {train_acc_arr.std():.4f}")

    # ==================== 测试集评估 ====================
    print("\n" + "=" * 80)
    print("在测试集上评估...")
    print("=" * 80)

    def sigmoid_np(x):
        return 1.0 / (1.0 + np.exp(-x))

    # 评估测试集
    test_results = {
        'pos': {'correct': 0, 'total': len(test_pos), 'logits': [], 'sigmoids': []},
        'neg': {'correct': 0, 'total': len(test_neg), 'logits': [], 'sigmoids': []}
    }

    # 测试成功样本
    print("\n评估测试集成功样本...")
    for trans in tqdm(test_pos):
        obs = {key: trans['observations'][key] for key in trans['observations'].keys()}
        # 添加batch维度
        obs_batch = {key: val[None, ...] if isinstance(val, np.ndarray) else np.array([val])
                     for key, val in obs.items()}

        rng, key = jax.random.split(rng)
        logits = classifier.apply_fn(
            {"params": classifier.params}, obs_batch, train=False, rngs={"dropout": key}
        )
        logit_val = float(logits.item())
        sigmoid_val = sigmoid_np(logit_val)
        pred = sigmoid_val > 0.5

        test_results['pos']['logits'].append(logit_val)
        test_results['pos']['sigmoids'].append(sigmoid_val)
        if pred:  # 真实标签是成功，预测也应该是成功
            test_results['pos']['correct'] += 1

    # 测试失败样本
    print("评估测试集失败样本...")
    for trans in tqdm(test_neg):
        obs = {key: trans['observations'][key] for key in trans['observations'].keys()}
        # 添加batch维度
        obs_batch = {key: val[None, ...] if isinstance(val, np.ndarray) else np.array([val])
                     for key, val in obs.items()}

        rng, key = jax.random.split(rng)
        logits = classifier.apply_fn(
            {"params": classifier.params}, obs_batch, train=False, rngs={"dropout": key}
        )
        logit_val = float(logits.item())
        sigmoid_val = sigmoid_np(logit_val)
        pred = sigmoid_val > 0.5

        test_results['neg']['logits'].append(logit_val)
        test_results['neg']['sigmoids'].append(sigmoid_val)
        if not pred:  # 真实标签是失败，预测也应该是失败
            test_results['neg']['correct'] += 1

    # ==================== 打印测试结果 ====================
    print("\n" + "=" * 80)
    print("📊 测试集详细结果")
    print("=" * 80)

    # 成功样本统计
    pos_logits = np.array(test_results['pos']['logits'])
    pos_sigmoids = np.array(test_results['pos']['sigmoids'])
    pos_acc = test_results['pos']['correct'] / test_results['pos']['total'] * 100

    print(f"\n✅ 成功样本统计:")
    print(f"  - 总数: {test_results['pos']['total']}")
    print(f"  - 正确: {test_results['pos']['correct']}")
    print(f"  - 准确率: {pos_acc:.2f}%")
    print(f"  - Logit 范围: [{pos_logits.min():.3f}, {pos_logits.max():.3f}]")
    print(f"  - Logit 均值: {pos_logits.mean():.3f} ± {pos_logits.std():.3f}")
    print(f"  - Sigmoid 范围: [{pos_sigmoids.min():.3f}, {pos_sigmoids.max():.3f}]")
    print(f"  - Sigmoid 均值: {pos_sigmoids.mean():.3f} ± {pos_sigmoids.std():.3f}")

    # 失败样本统计
    neg_logits = np.array(test_results['neg']['logits'])
    neg_sigmoids = np.array(test_results['neg']['sigmoids'])
    neg_acc = test_results['neg']['correct'] / test_results['neg']['total'] * 100

    print(f"\n❌ 失败样本统计:")
    print(f"  - 总数: {test_results['neg']['total']}")
    print(f"  - 正确: {test_results['neg']['correct']}")
    print(f"  - 准确率: {neg_acc:.2f}%")
    print(f"  - Logit 范围: [{neg_logits.min():.3f}, {neg_logits.max():.3f}]")
    print(f"  - Logit 均值: {neg_logits.mean():.3f} ± {neg_logits.std():.3f}")
    print(f"  - Sigmoid 范围: [{neg_sigmoids.min():.3f}, {neg_sigmoids.max():.3f}]")
    print(f"  - Sigmoid 均值: {neg_sigmoids.mean():.3f} ± {neg_sigmoids.std():.3f}")

    # 总体统计
    total_correct = test_results['pos']['correct'] + test_results['neg']['correct']
    total_samples = test_results['pos']['total'] + test_results['neg']['total']
    overall_acc = total_correct / total_samples * 100

    print("\n" + "=" * 80)
    print("📈 总体统计")
    print("=" * 80)
    print(f"总样本数: {total_samples}")
    print(f"总正确数: {total_correct}")
    print(f"总体准确率: {overall_acc:.2f}%")
    print(f"\n成功样本准确率: {test_results['pos']['correct']}/{test_results['pos']['total']} = {pos_acc:.2f}%")
    print(f"失败样本准确率: {test_results['neg']['correct']}/{test_results['neg']['total']} = {neg_acc:.2f}%")

    # 混淆矩阵
    TP = test_results['pos']['correct']
    FN = test_results['pos']['total'] - test_results['pos']['correct']
    TN = test_results['neg']['correct']
    FP = test_results['neg']['total'] - test_results['neg']['correct']

    print("\n🔢 混淆矩阵:")
    print(f"  真实成功 -> 预测成功: {TP} (真阳性 TP)")
    print(f"  真实成功 -> 预测失败: {FN} (假阴性 FN)")
    print(f"  真实失败 -> 预测失败: {TN} (真阴性 TN)")
    print(f"  真实失败 -> 预测成功: {FP} (假阳性 FP)")

    # 精确率、召回率、F1
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n📊 评估指标:")
    print(f"  - 精确率 (Precision): {precision:.2%}")
    print(f"  - 召回率 (Recall): {recall:.2%}")
    print(f"  - F1分数: {f1:.2%}")

    # 分离度分析
    sigmoid_separation = abs(pos_sigmoids.mean() - neg_sigmoids.mean())
    print(f"\n🎯 分类器性能分析:")
    print(f"  - Sigmoid 分离度: {sigmoid_separation:.3f}")
    if sigmoid_separation < 0.2:
        print(f"    ⚠️  分离度较低，分类器可能不够自信")
    elif sigmoid_separation < 0.5:
        print(f"    ✓  分离度中等，分类器表现良好")
    else:
        print(f"    ✓✓ 分离度较高，分类器非常自信")

    # 阈值建议
    print("\n🎚️  阈值分析:")
    print(f"  - 当前阈值: 0.50")
    print(f"  - 成功样本 sigmoid 均值: {pos_sigmoids.mean():.3f}")
    print(f"  - 失败样本 sigmoid 均值: {neg_sigmoids.mean():.3f}")
    suggested_threshold = (pos_sigmoids.mean() + neg_sigmoids.mean()) / 2
    print(f"  - 建议阈值 (两者中点): {suggested_threshold:.3f}")

    # 在不同阈值下的性能
    print(f"\n📉 不同阈值下的准确率:")
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        tp = np.sum(pos_sigmoids >= threshold)
        tn = np.sum(neg_sigmoids < threshold)
        acc = (tp + tn) / total_samples * 100
        print(f"  - 阈值 {threshold:.1f}: {acc:.2f}% (TP={tp}, TN={tn})")

    print("=" * 80)



if __name__ == "__main__":
    app.run(main)