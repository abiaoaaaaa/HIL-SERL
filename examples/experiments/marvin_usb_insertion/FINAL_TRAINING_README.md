# Marvin USB insertion 最终训练版本

更新时间：2026-08-03。

## 固定条件

最终运行条件为 scale `15 mm`、yaw scale `0.03 rad`、10 Hz 策略、200 Hz 流式控制、平移速度/加速度限制 `60 mm/s` 与 `200 mm/s²`。Actor 连续动作使用 EMA `alpha=0.65`；夹爪不经过 EMA。训练 batch 为 256，replay/demo buffer 容量均为 20000，critic warmup 为 2000 个更新步。

注意：15 mm 是单步目标增量上限。60 mm/s 在 0.1 秒内的稳态最大位移约为 6 mm，从静止起步还会受 200 mm/s² 加速度限制，因此大动作会由状态化流式参考跨控制周期平滑追踪，这是预期行为。

## 新训练的数据边界

默认 warm start：

```text
checkpoints_scale10_rot0025_from124k/checkpoint_270000
```

该 checkpoint 学到的是 scale10/rot0.025 下的归一化策略；最终环境将同一归一化输出解释为 scale15/rot0.03，即平移目标增量放大 1.5 倍、yaw 放大 1.2 倍。该组合已经过真机 scale/速度测试，但开始新训练时仍应保持 SpaceMouse 干预可用，并重点观察最初若干 episode。

新输出目录：

```text
checkpoints_final_scale15_rot003_stream60_fresh
```

通过 `--restore_weights_only` 只从旧 checkpoint 复制网络参数；优化器重新初始化。不传 `--legacy_replay_path`，也不从旧目录加载 `buffer/` 或 `demo_buffer/`。新录 demo 不做 legacy MDP migration。原有目录不会删除，借助新的输出目录实现逻辑上的空 buffer。

全局训练步号从 270001 接续，避免 warm-start Actor 再执行 `random_steps=500`，但新目录中的优化器、在线 replay 和 intervention demo buffer 都从空状态开始。开始后的前 2000 个 learner 更新只训练 critic，随后恢复 actor 更新。

## 启动顺序

1. 录制新 demo：

```bash
cd examples/experiments/marvin_usb_insertion
bash run_record_demos.sh 20
```

2. 启动 learner，把实际生成的 demo 绝对路径作为第一个参数。当前默认使用
   Actor/Critic 两层 `(256, 256)` 隐藏层、medium 图像增强。默认从 `checkpoints_final_scale15_rot003_stream60_fresh` 的最新 checkpoint 只加载模型权重；优化器、在线 replay 和 intervention demo buffer 均重新初始化，demo 数据由下面的路径单独加载。medium 模式保留 DrQ 的四周填充 `4 px` 后随机裁剪，并使用亮度、对比度和饱和度系数 `0.85～1.15`、色相 `±0.02`；四项颜色增强分别以 `0.3` 概率独立触发：

```bash
bash run_learner.sh \
  /home/xlb/code_marvin/hil-serl/examples/demo_data/marvin_usb_insertion_20_demos_2026-08-04_10-28-33.pkl
```

3. 另一个终端启动真机 Actor：

```bash
bash run_actor.sh
```

Actor 和 learner 必须使用相同的 `RUN_NAME`。默认值已在两个脚本中保持一致。
若要为另一组全新实验指定目录：

```bash
RUN_NAME=checkpoints_256_medium_run2 \
bash run_learner.sh /absolute/path/to/new_demo.pkl

RUN_NAME=checkpoints_256_medium_run2 \
bash run_actor.sh
```

learner 默认拒绝在含有 checkpoint、`buffer/` 或 `demo_buffer/` 的目录中启动“全新训练”。只有明确续训时才使用 `ALLOW_RESUME=1`。
默认 `PRETRAIN_STEP=0`，脚本会自动选择源目录中编号最大的 checkpoint；当前为 `checkpoint_392000`。只有网络参数 shape 一致的 `(256, 256)` checkpoint 才能通过 `PRETRAIN_DIR` 加载。
