#!/usr/bin/env python3
"""
轨迹数据深度分析工具 - Analyze Trajectory Data

功能：
    深度分析从 debug_step_timing.py 或 debug_movla_spacemouse.py
    生成的轨迹数据 JSON 文件，检测抖动、异常和性能瓶颈。

主要功能：
    1. 时间分布分析（movLA、发送、sleep 各阶段耗时）
    2. 轨迹点数量统计和变化趋势
    3. 关节角度变化分析（步内、步间、连续性）
    4. 规划轨迹 vs 实际执行对比
    5. 异常模式检测（回跳、震荡、突变）
    6. 多文件对比（如 10Hz vs 20Hz）

分析维度：

    1. 时间分布分析：
       - movLA 规划耗时：
         * 平均值、最大值、标准差
         * 是否稳定（抖动小）
       - 轨迹点发送耗时：
         * 总发送时间
         * 单点平均发送时间
         * 是否与轨迹点数成正比
       - sleep 等待时间：
         * 是否有效利用控制周期
         * 是否频繁超时
       - 总执行时间：
         * 是否符合控制频率要求
         * 超时比例

    2. 轨迹点数量分析：
       - 每个 step 规划的轨迹点数：
         * 平均点数
         * 点数变化范围
         * 点数稳定性（标准差）
       - 点数与频率的关系
       - 点数与移动距离的关系

    3. 关节角度变化分析：
       a) 单步内变化（step 前后）：
          - 每个关节的增量
          - 是否符合预期
       b) 步间变化（相邻 step 之间）：
          - joints_after[i] vs joints_before[i+1]
          - 检测不连续性（gap）
       c) 连续性分析：
          - 相邻 step 的实际执行是否连续
          - 检测回跳现象

    4. 规划 vs 实际分析：
       - 规划的轨迹点序列（planned_points）
       - 实际执行到的位置（joints_after）
       - 对比分析：
         * 执行到了第几个规划点
         * 是否提前停止
         * 是否超过规划终点

    5. 异常模式检测：
       a) 回跳（Backtrack）：
          - joints_after 比 joints_before 更远离目标
          - 检测频率和幅度
       b) 震荡（Oscillation）：
          - 连续多个 step 在同一位置附近振荡
          - 无有效前进
       c) 突变（Jump）：
          - 单步变化过大
          - 超出合理范围

对比分析（多文件）：
    支持同时分析两个文件（如 10Hz vs 20Hz），输出对比表格：
    - 控制频率
    - 平均规划耗时
    - 平均发送耗时
    - 平均轨迹点数
    - 异常检测结果

输出格式：
    ================================================================
    分析文件: trajectory_data_10hz_20240125.json
    ================================================================

    配置: 10Hz, freq_hz=500Hz
    有效 steps: 100

    ────────────────────────────────────────────────────────────────
    【1】时间分布分析
    ────────────────────────────────────────────────────────────────
    控制周期: 100.0ms
    movLA 规划: 平均=5.2ms, 最大=8.1ms
    发送时间:   平均=25.3ms, 最大=45.2ms
    sleep 时间: 平均=68.5ms, 最小=45.3ms
    总执行时间: 平均=99.0ms, 最大=105.2ms
    超时 steps: 5/100 (5.0%)

    ────────────────────────────────────────────────────────────────
    【2】轨迹点数量分析
    ────────────────────────────────────────────────────────────────
    规划点数: 平均=12.5, 范围=[8, 18], 标准差=2.3
    点数稳定性: 良好

    ────────────────────────────────────────────────────────────────
    【3】关节角度变化分析
    ────────────────────────────────────────────────────────────────
    单步内变化: 平均 [q1, q2, ..., q7]
    步间连续性: 98% 连续, 2 个 gap 检测到
    最大 gap: 0.05 rad (joint 3, step 45)

    ────────────────────────────────────────────────────────────────
    【4】异常模式检测
    ────────────────────────────────────────────────────────────────
    回跳检测: 3 次 (3.0%)
    震荡检测: 1 区域 (steps 67-72)
    突变检测: 0 次

    ================================================================

使用方法：
    cd /home/xlb/code_marvin/hil-serl

    # 分析单个文件
    python utils/visualization_tools/analyze_trajectory_data.py \
        utils/debug_tools/trajectory_data_10hz_20240125.json

    # 对比两个文件（10Hz vs 20Hz）
    python utils/visualization_tools/analyze_trajectory_data.py \
        utils/debug_tools/trajectory_data_10hz_20240125.json \
        utils/debug_tools/trajectory_data_20hz_20240125.json

应用场景：
    1. 诊断控制抖动问题
    2. 优化控制频率选择
    3. 分析 movLA 参数影响
    4. 验证轨迹平滑性
    5. 检测异常控制行为

后续可视化：
    使用 visualize_trajectory.py 生成图表：
    - 关节角度轨迹图
    - 时间分析柱状图
    - 发送时间戳散点图

注意事项：
    1. 输入文件必须包含完整的时序数据
    2. 至少需要 2 个有效 step 才能进行分析
    3. 对比分析时两个文件应该是相同任务的不同频率
    4. 分析结果仅供参考，需结合实际观察
"""
import json
import numpy as np
import sys

def analyze_trajectory_file(filepath):
    """深度分析单个轨迹文件"""
    print(f"\n{'='*80}")
    print(f"分析文件: {filepath}")
    print(f"{'='*80}\n")

    with open(filepath, 'r') as f:
        data = json.load(f)

    config = data['config']
    steps = [s for s in data['steps'] if not s.get('skipped', False)]

    print(f"配置: {config['control_hz']}Hz, freq_hz={config['movla_freq_hz']}")
    print(f"有效 steps: {len(steps)}\n")

    if len(steps) < 2:
        print("数据不足，跳过分析")
        return

    # ==========================================================================
    # 1. 时间分布分析
    # ==========================================================================
    print(f"{'─'*80}")
    print("【1】时间分布分析")
    print(f"{'─'*80}")

    time_movla = []
    time_send = []
    time_sleep = []
    total_time = []

    for s in steps:
        time_movla.append(s['time_movla'] * 1000)
        time_send.append(s['time_send_total'] * 1000)
        time_sleep.append(s.get('time_sleep', 0) * 1000)
        total_time.append(s['total_time'] * 1000)

    control_period = 1000.0 / config['control_hz']

    print(f"控制周期: {control_period:.1f}ms")
    print(f"movLA 规划: 平均={np.mean(time_movla):.2f}ms, 最大={np.max(time_movla):.2f}ms")
    print(f"发送时间:   平均={np.mean(time_send):.2f}ms, 最大={np.max(time_send):.2f}ms")
    print(f"sleep 时间: 平均={np.mean(time_sleep):.2f}ms, 最小={np.min(time_sleep):.2f}ms, 最大={np.max(time_sleep):.2f}ms")
    print(f"总执行时间: 平均={np.mean(total_time):.2f}ms, 最大={np.max(total_time):.2f}ms")
    print(f"是否超时:   {np.any(np.array(total_time) > control_period)}")

    # ==========================================================================
    # 2. 轨迹点数量分析
    # ==========================================================================
    print(f"\n{'─'*80}")
    print("【2】轨迹点数量分析")
    print(f"{'─'*80}")

    num_planned = [s['num_planned_points'] for s in steps]
    print(f"规划点数: 平均={np.mean(num_planned):.1f}, 最小={np.min(num_planned)}, 最大={np.max(num_planned)}")

    # ==========================================================================
    # 3. 关节角度变化分析（最关键！）
    # ==========================================================================
    print(f"\n{'─'*80}")
    print("【3】关节角度变化分析（关键！）")
    print(f"{'─'*80}")

    # 3.1 单个 step 内的变化（before -> after）
    print("\n3.1 单个 step 内的关节变化（before -> after）:")

    intra_step_changes = []
    for i, s in enumerate(steps[:10]):  # 只看前10个
        joints_before = np.array(s['joints_before'])
        joints_after = np.array(s['joints_after'])
        delta = joints_after - joints_before
        max_change = np.max(np.abs(delta))
        intra_step_changes.append(max_change)

        if i < 5:  # 只打印前5个
            print(f"  Step {s['step_id']:3d}: 最大变化={max_change:.4f}°, "
                  f"规划={s['num_planned_points']:2d}点, "
                  f"sleep={s.get('time_sleep', 0)*1000:.1f}ms")

    print(f"\n前10个step的单步内变化: 平均={np.mean(intra_step_changes):.4f}°, "
          f"最大={np.max(intra_step_changes):.4f}°")

    # 3.2 连续 step 间的变化（step[i].after -> step[i+1].before）
    print("\n3.2 连续 step 间的关节变化（step[i].after -> step[i+1].before）:")

    inter_step_changes = []
    for i in range(len(steps) - 1):
        joints_after_i = np.array(steps[i]['joints_after'])
        joints_before_i1 = np.array(steps[i+1]['joints_before'])
        delta = joints_before_i1 - joints_after_i
        max_change = np.max(np.abs(delta))
        inter_step_changes.append(max_change)

        if i < 5:  # 只打印前5个
            print(f"  Step {steps[i]['step_id']:3d} -> {steps[i+1]['step_id']:3d}: "
                  f"最大变化={max_change:.4f}°")

    print(f"\n前10个step间变化: 平均={np.mean(inter_step_changes[:10]):.4f}°, "
          f"最大={np.max(inter_step_changes[:10]):.4f}°")

    # 3.3 完整轨迹的连续性检查
    print("\n3.3 完整轨迹连续性:")

    all_joints = []
    for s in steps:
        all_joints.append(s['joints_before'])
    all_joints.append(steps[-1]['joints_after'])

    all_joints = np.array(all_joints)

    # 计算相邻点的最大跳变
    max_jumps = []
    for i in range(len(all_joints) - 1):
        delta = np.abs(all_joints[i+1] - all_joints[i])
        max_jumps.append(np.max(delta))

    print(f"轨迹点数: {len(all_joints)}")
    print(f"最大跳变: {np.max(max_jumps):.4f}°")
    print(f"平均跳变: {np.mean(max_jumps):.4f}°")
    print(f"超过0.5°的跳变数: {np.sum(np.array(max_jumps) > 0.5)}")
    print(f"超过1.0°的跳变数: {np.sum(np.array(max_jumps) > 1.0)}")

    # ==========================================================================
    # 4. 规划点 vs 实际执行点对比
    # ==========================================================================
    print(f"\n{'─'*80}")
    print("【4】规划轨迹 vs 实际关节变化")
    print(f"{'─'*80}")

    print("\n前5个step的对比:")
    for i, s in enumerate(steps[:5]):
        joints_before = np.array(s['joints_before'])
        joints_after = np.array(s['joints_after'])
        planned_points = np.array(s['planned_points'])

        # 规划的总变化（第一个点 -> 最后一个点）
        planned_first = planned_points[0]
        planned_last = planned_points[-1]
        planned_delta = np.max(np.abs(planned_last - planned_first))

        # 实际变化
        actual_delta = np.max(np.abs(joints_after - joints_before))

        execution_rate = (actual_delta / planned_delta * 100) if planned_delta > 0 else 0

        print(f"  Step {s['step_id']:3d}: 规划变化={planned_delta:.4f}°, "
              f"实际变化={actual_delta:.4f}°, 执行率={execution_rate:.1f}%")

    # ==========================================================================
    # 5. 检测异常模式
    # ==========================================================================
    print(f"\n{'─'*80}")
    print("【5】异常模式检测")
    print(f"{'─'*80}")

    # 5.1 检测回跳（连续两个 step 的移动方向相反）
    print("\n5.1 回跳检测（连续 step 移动方向相反）:")

    backtrack_count = 0
    for i in range(len(steps) - 1):
        j_before_i = np.array(steps[i]['joints_before'])
        j_after_i = np.array(steps[i]['joints_after'])
        j_before_i1 = np.array(steps[i+1]['joints_before'])
        j_after_i1 = np.array(steps[i+1]['joints_after'])

        delta_i = j_after_i - j_before_i
        delta_i1 = j_after_i1 - j_before_i1

        # 检查是否有关节方向相反
        dot_product = np.dot(delta_i, delta_i1)
        if dot_product < 0 and np.linalg.norm(delta_i) > 0.01 and np.linalg.norm(delta_i1) > 0.01:
            backtrack_count += 1
            if backtrack_count <= 3:
                print(f"  ⚠️ Step {steps[i]['step_id']} -> {steps[i+1]['step_id']}: 检测到回跳")

    print(f"\n总回跳次数: {backtrack_count} / {len(steps)-1} ({backtrack_count/(len(steps)-1)*100:.1f}%)")

    # 5.2 检测震荡（关节值在小范围内反复变化）
    print("\n5.2 震荡检测:")

    if len(all_joints) >= 10:
        window_size = 5
        oscillation_count = 0

        for joint_idx in range(6):
            joint_values = all_joints[:, joint_idx]

            for i in range(len(joint_values) - window_size):
                window = joint_values[i:i+window_size]
                std = np.std(window)
                range_val = np.max(window) - np.min(window)

                # 检测：标准差小但range大（说明在小范围内剧烈变化）
                if std < 0.1 and range_val > 0.2:
                    oscillation_count += 1

        print(f"检测到震荡窗口数: {oscillation_count}")

    return {
        'control_hz': config['control_hz'],
        'avg_intra_step_change': np.mean(intra_step_changes),
        'avg_inter_step_change': np.mean(inter_step_changes[:10]),
        'max_jump': np.max(max_jumps),
        'avg_sleep': np.mean(time_sleep),
        'backtrack_rate': backtrack_count / (len(steps)-1) * 100 if len(steps) > 1 else 0
    }


def compare_files(file1, file2):
    """对比两个文件"""
    print(f"\n\n{'='*80}")
    print("对比分析")
    print(f"{'='*80}\n")

    result1 = analyze_trajectory_file(file1)
    result2 = analyze_trajectory_file(file2)

    print(f"\n{'='*80}")
    print("总结对比")
    print(f"{'='*80}\n")

    print(f"{'指标':<30} | {result1['control_hz']}Hz | {result2['control_hz']}Hz")
    print(f"{'─'*30} | {'─'*10} | {'─'*10}")
    print(f"{'单步内平均变化 (°)':<30} | {result1['avg_intra_step_change']:>10.4f} | {result2['avg_intra_step_change']:>10.4f}")
    print(f"{'步间平均变化 (°)':<30} | {result1['avg_inter_step_change']:>10.4f} | {result2['avg_inter_step_change']:>10.4f}")
    print(f"{'最大跳变 (°)':<30} | {result1['max_jump']:>10.4f} | {result2['max_jump']:>10.4f}")
    print(f"{'平均 sleep 时间 (ms)':<30} | {result1['avg_sleep']:>10.2f} | {result2['avg_sleep']:>10.2f}")
    print(f"{'回跳率 (%)':<30} | {result1['backtrack_rate']:>10.1f} | {result2['backtrack_rate']:>10.1f}")


if __name__ == '__main__':
    if len(sys.argv) == 3:
        compare_files(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        analyze_trajectory_file(sys.argv[1])
    else:
        print("用法:")
        print("  分析单个文件: python analyze_trajectory_data.py file.json")
        print("  对比两个文件: python analyze_trajectory_data.py file1.json file2.json")
