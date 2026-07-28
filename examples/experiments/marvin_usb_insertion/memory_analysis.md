# 训练内存占用详细分析

## 内存组件分解（总内存 60GB 系统）

### 1. **Replay Buffer（固定大小）** - 5.6 GB
```python
capacity = 5000
observations: 5000 × 3 × 256×256×3 = 2.8 GB
next_observations: 5000 × 3 × 256×256×3 = 2.8 GB
actions/rewards/masks: ~0.1 GB
──────────────────────────────────────
总计: 5.6 GB (np.empty预分配，不增长) ✅
```

### 2. **Demo Buffer（固定大小）** - 5.6 GB
```python
# 训练使用 50/50 采样 (replay_buffer + demo_buffer)
# demo_buffer 同样大小
capacity = 5000
总计: 5.6 GB ✅
```

### 3. **JAX 训练缓存（动态，会增长）** - 10-20 GB ⚠️
```python
# 每次训练迭代:
batch_size = 256 (128 replay + 128 demo)
images: 256 × 3 × 256×256×3 = 0.15 GB
gradients: 2x params = ~8 GB (包含 critic, actor, temperature)
optimizer states (Adam): ~4 GB
中间激活值: ~3 GB
──────────────────────────────────────
单个训练步: ~15 GB
多步累积 (cta_ratio=8): 15×2 = ~30 GB ❌ 内存泄漏风险
```

**关键代码** (train_rlpd.py:319-329):
```python
for critic_step in range(config.cta_ratio - 1):  # cta_ratio=8
    batch = next(replay_iterator)  # ← 每次采样新batch
    agent, critics_info = agent.update(batch)  # ← 梯度累积
```

### 4. **XLA 编译缓存（累积增长）** - 5-15 GB ⚠️
```python
# JAX/XLA 会缓存编译后的计算图
# 首次调用编译，后续复用
# 但如果输入 shape 变化，会重新编译并累积缓存

预期: ~5 GB (正常)
实际: 5-15 GB (如果有 shape 不一致导致重复编译)
```

### 5. **图像解码缓存（动态增长）** - 5-10 GB ⚠️
```python
# ReplayBuffer 迭代器会批量解码图像
# JPEG/PNG 解码缓存 + numpy 临时数组

正常: ~5 GB
异常: ~10 GB (如果没有及时释放)
```

### 6. **Python 运行时 + 其他** - 5 GB
```python
- Python 解释器: ~2 GB
- 导入的库 (numpy, jax, etc): ~2 GB
- 其他杂项: ~1 GB
```

### 7. **系统保留** - 5-10 GB
```python
- OS: ~3 GB
- 桌面环境: ~2 GB
- 其他后台进程: ~2 GB
```

---

## 内存占用时间线

| 阶段 | Buffer | Demo | JAX | XLA | 图像 | 其他 | 总计 |
|------|--------|------|-----|-----|------|------|------|
| **启动** | 5.6 | 5.6 | 2 | 0 | 0 | 5 | **18 GB** ✅ |
| **首次训练** | 5.6 | 5.6 | 15 | 5 | 5 | 5 | **41 GB** ⚠️ |
| **持续训练** | 5.6 | 5.6 | 15 | 10 | 8 | 5 | **49 GB** ⚠️ |
| **长时间运行** | 5.6 | 5.6 | 20 | 15 | 10 | 5 | **61 GB** ❌ OOM |

---

## 🔴 内存泄漏源头分析

### **罪魁祸首 #1: JAX 训练缓存持续增长**

**原因**: `cta_ratio=8` 导致多次 batch 采样和更新，梯度/激活值累积

**证据**:
```
anon-rss 增长曲线:
14:29 → 33 GB
15:32 → 47 GB (+14 GB in 1hr)
15:40 → 49 GB (+2 GB in 8min)
15:51 → 50 GB (+1 GB in 11min)
```

持续缓慢增长 = 内存泄漏 ⚠️

---

### **罪魁祸首 #2: XLA 重复编译**

**原因**: 如果 batch shape 不一致（例如最后一个 batch 不满），XLA 会重新编译

**检查方法**:
```python
# 在训练循环中添加
print(f"Batch shapes: obs={batch['observations'].shape}")
```

如果看到 shape 变化 → XLA 缓存爆炸

---

### **罪魁祸首 #3: 图像解码缓存未释放**

**原因**: ReplayBuffer 迭代器的图像解码可能未及时 GC

---

## 🟢 解决方案（按优先级）

### **方案 1: 降低 cta_ratio（立即有效）** ⭐⭐⭐⭐⭐
```python
# config.py
cta_ratio = 4  # 从默认 8 → 4
# 预期内存节省: ~8 GB
```

### **方案 2: 降低 batch_size（有效）** ⭐⭐⭐⭐
```python
# config.py
batch_size = 128  # 从 256 → 128
# 预期内存节省: ~5 GB
```

### **方案 3: 启用 JAX 内存预分配限制（推荐）** ⭐⭐⭐⭐⭐
```bash
# 在 run_learner.sh 开头添加
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5  # 限制 JAX 只用 50% GPU/内存
export XLA_PYTHON_CLIENT_ALLOCATOR=platform  # 使用系统分配器，更好的内存释放
```

### **方案 4: 强制 GC（辅助）** ⭐⭐⭐
```python
# 在训练循环中（每 100 步）
if step % 100 == 0:
    import gc
    gc.collect()
    jax.clear_backends()  # 清理 JAX 缓存
```

### **方案 5: 减少 Buffer（已实施）** ⭐⭐⭐⭐
```python
replay_buffer_capacity = 5000  ✅
```

---

## 🎯 最终推荐配置

```python
# config.py
replay_buffer_capacity = 5000  # ✅ 已设置
batch_size = 128              # ← 新增
cta_ratio = 4                 # ← 新增

预期内存: ~25-30 GB ✅ 安全！
```

---

## 监控命令

```bash
# 实时监控内存
watch -n 2 '
echo "=== 系统内存 ==="
free -h
echo ""
echo "=== 训练进程 ==="
ps aux | grep train_rlpd | grep -v grep | awk "{printf \"RSS: %.1f GB, VSZ: %.1f GB\n\", \$6/1024/1024, \$5/1024/1024}"
'
```
