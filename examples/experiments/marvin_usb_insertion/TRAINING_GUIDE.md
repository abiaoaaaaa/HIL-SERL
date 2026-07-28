# 训练启动指南 (内存优化版)

## 🚀 快速启动

### 1. 训练前准备
```bash
cd /home/xlb/code_marvin/hil-serl/examples/experiments/marvin_usb_insertion

# 清理内存
sudo ./free_memory.sh

# 关闭 Chrome
pkill -9 chrome

# 确认可用内存 > 40GB
free -h
```

### 2. 启动训练 (3个终端)

**终端1: Learner (训练器)**
```bash
cd /home/xlb/code_marvin/hil-serl/examples/experiments/marvin_usb_insertion
bash run_learner.sh
```

**终端2: Actor (采集器)**
```bash
cd /home/xlb/code_marvin/hil-serl/examples/experiments/marvin_usb_insertion
bash run_actor.sh
```

**终端3: 内存监控**
```bash
cd /home/xlb/code_marvin/hil-serl/examples/experiments/marvin_usb_insertion
./monitor_memory.sh
```

---

## 📊 内存预期

| 阶段 | 内存占用 | 状态 |
|------|----------|------|
| 启动 | ~10 GB | ✅ 正常 |
| 填充buffer | ~20 GB | ✅ 正常 |
| 开始训练 | ~28 GB | ✅ 正常 |
| 稳定运行 | ~30-35 GB | ✅ 安全 |
| **峰值** | **<40 GB** | ✅ **目标** |

**警告阈值**: >40GB → 可能 OOM

---

## ⚙️ 已应用的优化

```python
# config.py
replay_buffer_capacity = 5000    # 200k → 5k (节省 ~35GB)
batch_size = 128                  # 256 → 128 (节省 ~5GB)
cta_ratio = 2                     # 保持低值

# run_learner.sh
XLA_PYTHON_CLIENT_MEM_FRACTION=.4  # JAX 限制 40% 内存
XLA_PYTHON_CLIENT_ALLOCATOR=platform # 更好的内存释放
```

**总节省**: ~57GB (90GB → 33GB)

---

## 🔧 故障排查

### 问题1: 还是 OOM (>50GB)

**解决方案A**: 进一步降低 buffer
```python
# config.py:228
replay_buffer_capacity = 3000  # 5000 → 3000
```

**解决方案B**: 减少相机
```python
# config.py:210
image_keys = ["side_policy", "wrist_1"]  # 去掉 wrist_2
```

### 问题2: 训练速度变慢

**原因**: batch_size 降低导致
**影响**: 每步更新效率降低，但总体收敛不受影响
**可接受**: 内存安全 > 速度

### 问题3: JAX 报错 "Out of memory"

**解决方案**: 提高内存限制
```bash
# run_learner.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5  # .4 → .5
```

---

## 📈 训练监控

### 实时监控命令
```bash
# 简化版
watch -n 3 'free -h && ps aux | grep train_rlpd | head -1'

# 详细版
./monitor_memory.sh
```

### 检查日志
```bash
# 查看最新的内存监控日志
tail -f memory_monitor_*.log
```

### 可视化 (可选)
```bash
# 绘制内存曲线
python3 << EOF
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('memory_monitor_*.log')
df.plot(x='时间', y='训练进程RSS(GB)')
plt.axhline(y=40, color='r', linestyle='--', label='危险线')
plt.legend()
plt.savefig('memory_usage.png')
print("已保存: memory_usage.png")
EOF
```

---

## ✅ 成功标志

训练成功启动的标志：
1. Learner 显示 "Filling up replay buffer" 进度条
2. Actor 开始采集数据 (看到 step 数字增长)
3. 内存监控显示 30-35GB 稳定运行
4. 无 "Out of memory" 或 "Killed" 错误

---

## 📞 需要帮助

如果还是遇到 OOM:
1. 检查 `monitor_memory.sh` 日志，找到内存峰值时间点
2. 查看 syslog: `grep -i 'killed process' /var/log/syslog | tail -5`
3. 进一步降低 buffer 或 batch_size
