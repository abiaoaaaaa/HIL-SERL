#!/usr/bin/env python3
"""
分类器准确率测试工具 - Test Classifier Accuracy

功能：
    从标注好的成功/失败数据集加载图像，使用训练好的分类器进行
    预测，计算准确率、召回率、精确率等评估指标。

主要功能：
    1. 加载训练好的分类器模型
    2. 从数据集加载成功和失败的图像样本
    3. 批量预测并计算评估指标
    4. 输出混淆矩阵和详细统计信息

工作流程：
    1. 初始化：
       - 配置 JAX 环境（CPU模式，避免GPU内存问题）
       - 加载分类器模型（从 checkpoint）
       - 设置成功阈值（sigmoid > threshold）

    2. 加载数据集：
       a) 成功样本：
          - 从 DATA_DIR/success/*.pkl 加载
          - 每个 pkl 文件包含一个 episode 的数据
          - 提取 side_image（或其他相机的图像）
       b) 失败样本：
          - 从 DATA_DIR/failure/*.pkl 加载
          - 同样提取相机图像

    3. 批量预测：
       - 遍历所有样本
       - 调用分类器获取 logit 和 sigmoid 值
       - 根据阈值判断预测类别（成功/失败）
       - 显示进度条（tqdm）

    4. 计算评估指标：
       - 混淆矩阵：
         * True Positive (TP): 真成功预测为成功
         * True Negative (TN): 真失败预测为失败
         * False Positive (FP): 真失败预测为成功
         * False Negative (FN): 真成功预测为失败
       - 准确率 (Accuracy): (TP + TN) / Total
       - 精确率 (Precision): TP / (TP + FP)
       - 召回率 (Recall): TP / (TP + FN)
       - F1 分数: 2 * (Precision * Recall) / (Precision + Recall)

    5. 输出结果：
       - 打印混淆矩阵
       - 打印各项评估指标
       - 显示分类阈值信息

数据格式：
    pkl 文件结构：
    {
        "observations": [
            {
                "side_image": np.array([H, W, C]),  # 相机图像
                ...
            },
            ...
        ],
        ...
    }

配置说明：
    - CLASSIFIER_KEY: 分类器使用的键名（如 "side_classifier"）
    - CHECKPOINT_PATH: 分类器模型检查点路径
    - DATA_DIR: 数据集根目录（包含 success/ 和 failure/ 子目录）
    - SUCCESS_THRESHOLD: sigmoid 阈值（> threshold 为成功）

输出：
    控制台显示：
    ============================================================
    混淆矩阵：
                    预测成功    预测失败
    真实成功          TP          FN
    真实失败          FP          TN

    评估指标：
    - 总样本数: N
    - 成功样本: N_success
    - 失败样本: N_failure
    - 准确率 (Accuracy): XX.X%
    - 精确率 (Precision): XX.X%
    - 召回率 (Recall): XX.X%
    - F1 分数: X.XXX
    ============================================================

使用方法：
    cd /home/xlb/code_marvin/hil-serl

    # 编辑配置（如果需要）
    # - CLASSIFIER_KEY: 分类器键名
    # - CHECKPOINT_PATH: 模型路径
    # - DATA_DIR: 数据集路径
    # - SUCCESS_THRESHOLD: 阈值

    # 运行测试
    python utils/test_tools/test_classifier_accuracy.py

应用场景：
    1. 评估新训练的分类器性能
    2. 调整分类阈值以平衡精确率和召回率
    3. 验证分类器在测试集上的泛化能力
    4. 对比不同训练轮次的模型性能

注意事项：
    1. 确保数据集目录结构正确（success/ 和 failure/ 子目录）
    2. CLASSIFIER_KEY 必须与训练时使用的键名一致
    3. 数据集图像预处理必须与训练时一致
    4. JAX 配置为 CPU 模式，避免 GPU 内存问题
    5. 阈值调整：
       - 降低阈值 → 更多预测为成功 → 召回率↑ 精确率↓
       - 提高阈值 → 更少预测为成功 → 召回率↓ 精确率↑
"""
import os
import sys

# 设置JAX环境 - 必须在import jax之前
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # 强制使用CPU避免CUDA问题

import glob
import pickle as pkl
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from pathlib import Path

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
serl_infra_path = os.path.join(project_root, 'serl_robot_infra')
examples_path = os.path.join(project_root, 'examples')
sys.path.insert(0, project_root)
sys.path.insert(0, serl_infra_path)
sys.path.insert(0, examples_path)

from serl_launcher.networks.reward_classifier import load_classifier_func

# ==================== 配置 ====================
# 注意：训练时用的是 classifier_keys，可能是 side_policy 或其他键
CLASSIFIER_KEY = "side_classifier"  # 改为实际存在的键
CHECKPOINT_PATH = "/home/xlb/code_marvin/hil-serl/examples/classifier_ckpt/checkpoint_150"
DATA_DIR = Path("/home/xlb/code_marvin/hil-serl/examples/classifier_data")
SUCCESS_THRESHOLD = 0.5  # 降低到0.5试试

print("=" * 80)
print("分类器准确率测试")
print("=" * 80)
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"数据目录: {DATA_DIR}")
print(f"分类器键: {CLASSIFIER_KEY}")
print(f"成功阈值: sigmoid > {SUCCESS_THRESHOLD}")
print()

# ==================== 加载数据 ====================
print("[1/3] 加载数据集...")

success_paths = glob.glob(str(DATA_DIR / "*success*.pkl"))
failure_paths = glob.glob(str(DATA_DIR / "*failure*.pkl"))

print(f"  - 成功数据文件: {len(success_paths)}")
print(f"  - 失败数据文件: {len(failure_paths)}")

# 加载成功样本
success_samples = []
for path in tqdm(success_paths, desc="加载成功样本"):
    data = pkl.load(open(path, "rb"))
    for trans in data:
        # 提取图像 - shape可能是 (1, H, W, 3) 需要squeeze
        if CLASSIFIER_KEY in trans['observations']:
            img = trans['observations'][CLASSIFIER_KEY]
            # 如果是 (1, H, W, 3) 则squeeze成 (H, W, 3)
            if img.ndim == 4 and img.shape[0] == 1:
                img = img[0]
            success_samples.append(img)

# 加载失败样本
failure_samples = []
for path in tqdm(failure_paths, desc="加载失败样本"):
    data = pkl.load(open(path, "rb"))
    for trans in data:
        # 提取图像 - shape可能是 (1, H, W, 3) 需要squeeze
        if CLASSIFIER_KEY in trans['observations']:
            img = trans['observations'][CLASSIFIER_KEY]
            # 如果是 (1, H, W, 3) 则squeeze成 (H, W, 3)
            if img.ndim == 4 and img.shape[0] == 1:
                img = img[0]
            failure_samples.append(img)

print(f"\n成功样本数: {len(success_samples)}")
print(f"失败样本数: {len(failure_samples)}")

if len(success_samples) == 0 or len(failure_samples) == 0:
    print("[ERROR] 没有足够的数据样本！")
    exit(1)

# ==================== 加载分类器 ====================
# ==================== 加载分类器 ====================
print("\n[2/3] 加载分类器...")
try:
    # 准备一个样本用于初始化
    sample_obs = {CLASSIFIER_KEY: success_samples[0][None, ...]}  # [1, H, W, 3]

    # 创建JAX随机key
    rng = jax.random.PRNGKey(0)

    classifier_func = load_classifier_func(
        key=rng,
        sample=sample_obs,
        image_keys=[CLASSIFIER_KEY],
        checkpoint_path=CHECKPOINT_PATH,
    )
    print("  ✅ 分类器加载成功")
except Exception as e:
    print(f"[ERROR] 分类器加载失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ==================== 测试分类器 ====================
print("\n[3/3] 测试分类器...")

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def test_samples(samples, true_label_name):
    """测试一组样本"""
    correct = 0
    total = len(samples)

    logits = []
    sigmoids = []
    predictions = []

    for img in tqdm(samples, desc=f"测试{true_label_name}样本"):
        # 准备观测
        obs = {CLASSIFIER_KEY: img[None, ...]}  # [1, H, W, 3]

        # 推理
        classifier_logit = classifier_func(obs)
        logit_val = float(classifier_logit.item())
        sigmoid_val = sigmoid(logit_val)
        pred = sigmoid_val > SUCCESS_THRESHOLD

        logits.append(logit_val)
        sigmoids.append(sigmoid_val)
        predictions.append(pred)

        # 判断正确性
        if true_label_name == "成功":
            # 真实标签是成功，预测也应该是成功(True)
            if pred:
                correct += 1
        else:
            # 真实标签是失败，预测也应该是失败(False)
            if not pred:
                correct += 1

    accuracy = correct / total * 100

    # 统计
    logits = np.array(logits)
    sigmoids = np.array(sigmoids)

    print(f"\n{true_label_name}样本统计:")
    print(f"  - 总数: {total}")
    print(f"  - 正确: {correct}")
    print(f"  - 准确率: {accuracy:.2f}%")
    print(f"  - Logit 范围: [{logits.min():.3f}, {logits.max():.3f}]")
    print(f"  - Logit 均值: {logits.mean():.3f} ± {logits.std():.3f}")
    print(f"  - Sigmoid 范围: [{sigmoids.min():.3f}, {sigmoids.max():.3f}]")
    print(f"  - Sigmoid 均值: {sigmoids.mean():.3f} ± {sigmoids.std():.3f}")

    return correct, total, logits, sigmoids

# 测试成功样本
success_correct, success_total, success_logits, success_sigmoids = test_samples(
    success_samples, "成功"
)

# 测试失败样本
failure_correct, failure_total, failure_logits, failure_sigmoids = test_samples(
    failure_samples, "失败"
)

# ==================== 总结 ====================
total_correct = success_correct + failure_correct
total_samples = success_total + failure_total
overall_accuracy = total_correct / total_samples * 100

print("\n" + "=" * 80)
print("总体统计")
print("=" * 80)
print(f"总样本数: {total_samples}")
print(f"总正确数: {total_correct}")
print(f"总体准确率: {overall_accuracy:.2f}%")
print()
print(f"成功样本准确率: {success_correct}/{success_total} = {success_correct/success_total*100:.2f}%")
print(f"失败样本准确率: {failure_correct}/{failure_total} = {failure_correct/failure_total*100:.2f}%")
print()

# 混淆矩阵
print("混淆矩阵:")
print(f"  真实成功 -> 预测成功: {success_correct} (真阳性 TP)")
print(f"  真实成功 -> 预测失败: {success_total - success_correct} (假阴性 FN)")
print(f"  真实失败 -> 预测失败: {failure_correct} (真阴性 TN)")
print(f"  真实失败 -> 预测成功: {failure_total - failure_correct} (假阳性 FP)")
print()

# 精确率和召回率
precision = success_correct / (success_correct + (failure_total - failure_correct)) if (success_correct + (failure_total - failure_correct)) > 0 else 0
recall = success_correct / success_total if success_total > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"精确率 (Precision): {precision:.2%}")
print(f"召回率 (Recall): {recall:.2%}")
print(f"F1分数: {f1:.2%}")
print()

# 阈值建议
print("阈值分析:")
print(f"当前阈值: {SUCCESS_THRESHOLD}")
print(f"成功样本 sigmoid 均值: {success_sigmoids.mean():.3f}")
print(f"失败样本 sigmoid 均值: {failure_sigmoids.mean():.3f}")
print(f"建议阈值 (两者中点): {(success_sigmoids.mean() + failure_sigmoids.mean()) / 2:.3f}")
print("=" * 80)
