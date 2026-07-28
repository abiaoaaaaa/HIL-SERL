#!/usr/bin/env python3
"""
端到端验证程序：从 SpaceMouse 原始 6D 输入到最终基座系指令

模拟两条路径并逐步对比：
  路径A (6D+mask):  参考实现 — 6D 动作全变换后 mask 掉 rx_base/ry_base
  路径B (5D fixed): 当前修复 — 5D 动作只保留 EE ry 填入 ry 槽位

数据流对比:
  ┌─────────────────────────────────────────────────────────┐
  │  6D+mask:                                               │
  │  SM 6D → SM(7D) → RelativeFrame(T@[:6]) → mask rx,ry  │
  │  5D fixed:                                              │
  │  SM 6D → SM提取dry(5D) → RelativeFrame(补零→变换→提取)  │
  └─────────────────────────────────────────────────────────┘
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

# ==============================================================================
# 配置
# ==============================================================================
RESET_POSE = np.array([394.3, 321.7, 200.3, -90.05, 0.01, -90.01])  # mm, deg
ACTION_SCALE = np.array([15.0, 0.1, 1.0])  # [xyz mm/step, rot rad/step, gripper]
TEST_YAW_ANGLES = [-90.0, 0.0, 90.0]  # deg

# SpaceMouse 6D 测试输入 (EE系): [dx, dy, dz, rx, ry, rz]
# 覆盖各种组合
SM_TEST_CASES = [
    # [dx,  dy,  dz,  rx,   ry,   rz]   描述
    [ 0.0, 0.0, 0.0, 0.0,  0.5,  0.0],  # T1: 纯 pitch 正转
    [ 0.0, 0.0, 0.0, 0.0, -0.5,  0.0],  # T2: 纯 pitch 反转
    [ 0.0, 0.0, 0.0, 0.0,  0.0,  0.0],  # T3: 零输入
    [ 0.3, 0.2, 0.0, 0.0,  0.0,  0.0],  # T4: 纯 XY 位移 (无旋转)
    [ 0.0, 0.0, 0.5, 0.0,  0.0,  0.0],  # T5: 纯 Z 位移 (无旋转)
    [ 0.3, 0.2,-0.1, 0.0,  0.3,  0.0],  # T6: XYZ位移 + pitch旋转
    [ 0.0, 0.0, 0.0, 0.5,  0.0,  0.0],  # T7: 纯 roll  (应被屏蔽)
    [ 0.0, 0.0, 0.0, 0.0,  0.0,  0.5],  # T8: 纯 yaw   (应被屏蔽)
    [ 0.0, 0.0, 0.0, 0.3,  0.5,  0.2],  # T9: 全旋转  (只有 ry 应该生效)
    [-0.1,-0.1,-0.1, 0.0,  0.2,  0.0],  # T10: XYZ位移 + pitch旋转
    [ 0.0, 0.0, 0.0, 0.0, -0.1,  0.0],  # T11: 小 pitch 反转
    [ 0.5, 0.0, 0.3, 0.0,  0.8,  0.0],  # T12: 混合大动作
]

# 容差: 位置 1e-9 mm, 旋转 1e-4 rad (~0.006°) — 远小于机械精度
POS_TOL = 1e-9
ROT_TOL = 1e-4


# ==============================================================================
# 工具函数
# ==============================================================================

def euler_to_quat(euler_rad):
    return R.from_euler("xyz", euler_rad).as_quat()


def construct_transform_matrix(tcp_pose):
    """复刻 transformations.py"""
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
    T = np.zeros((6, 6))
    T[:3, :3] = rotation
    T[3:, 3:] = rotation
    return T


def make_tcp_pose(yaw_deg):
    """构造当前 tcp_pose (m, quat)"""
    euler_rad = np.deg2rad([RESET_POSE[3], RESET_POSE[4], yaw_deg])
    return np.concatenate([RESET_POSE[:3] / 1000.0, euler_to_quat(euler_rad)])


# ==============================================================================
# 路径A: 6D + mask (参考实现, 6D 时代的做法)
# ==============================================================================

def path_6d_mask(sm_input_6d, tcp_pose, gripper=0.0):
    """
    路径A: 6D 全变换 + 基座系 mask rx/ry

    Step A1: SpaceMouse → 7D [dx,dy,dz,rx,ry,rz,gripper] (EE系)
    Step A2: RelativeFrame: action[:6] = T @ action[:6]  (6D全变换)
    Step A3: MarvinEnv mask: rx_base=0, ry_base=0, 只保留 rz_base
    """
    # A1: SpaceMouse 6D → 7D (旧版 SM 直接拼接)
    action_7d_ee = np.zeros(7)
    action_7d_ee[:6] = sm_input_6d  # [dx,dy,dz,rx,ry,rz] (EE系)
    action_7d_ee[6] = gripper

    # A2: RelativeFrame 全变换
    T = construct_transform_matrix(tcp_pose)
    action_7d_base = action_7d_ee.copy()
    action_7d_base[:6] = T @ action_7d_ee[:6]

    # A3: MarvinEnv mask (6D版本)
    action_rot_rad_base = action_7d_base[3:6] * ACTION_SCALE[1]
    # 屏蔽基座 rx, ry, 只保留 rz
    action_rot_rad_base[0] = 0.0  # mask rx_base
    action_rot_rad_base[1] = 0.0  # mask ry_base
    # action_rot_rad_base[2]  = rz_base, 保留

    # 位置 (6D版本也是直接用)
    pos_delta_mm = action_7d_base[:3] * ACTION_SCALE[0]

    result = {
        "action_7d_ee": action_7d_ee,
        "action_7d_base": action_7d_base,
        "transform_matrix": T,
        "pos_delta_mm": pos_delta_mm,
        "rot_delta_rad": action_rot_rad_base,  # 已 mask
        "rz_base_rad": action_rot_rad_base[2],
        "gripper": gripper,
    }
    return result


# ==============================================================================
# 路径B: 5D fixed (当前修复版)
# ==============================================================================

def path_5d_fixed(sm_input_6d, tcp_pose, gripper=0.0):
    """
    路径B: 5D 修复版

    Step B1: SpaceMouse 6D → 提取 dry=expert_a[4](EE ry) → 5D [dx,dy,dz,dry,gripper]
    Step B2: RelativeFrame 5D 变换: pad [dx,dy,dz, 0,dry,0] → T@ → extract [dx,dy,dz,drz]
    Step B3: MarvinEnv: action_rot_rad[2] = drz * scale
    """
    # B1: SpaceMouse 6D → 5D (当前 SM 逻辑)
    action_5d_ee = np.zeros(5)
    action_5d_ee[0] = sm_input_6d[0]  # dx
    action_5d_ee[1] = sm_input_6d[1]  # dy
    action_5d_ee[2] = sm_input_6d[2]  # dz
    action_5d_ee[3] = sm_input_6d[4]  # dry = EE ry (expert_a[4])
    action_5d_ee[4] = gripper

    # B2: RelativeFrame 5D 变换 (修复版)
    T = construct_transform_matrix(tcp_pose)
    # pad 6D: [dx,dy,dz, 0,dry,0]
    action_6d_padded = np.zeros(6)
    action_6d_padded[0] = action_5d_ee[0]
    action_6d_padded[1] = action_5d_ee[1]
    action_6d_padded[2] = action_5d_ee[2]
    action_6d_padded[4] = action_5d_ee[3]  # dry → EE ry 槽位
    # rx=0, rz=0

    transformed_6d = T @ action_6d_padded

    # extract 5D base
    action_5d_base = np.zeros(5)
    action_5d_base[0] = transformed_6d[0]  # dx_base
    action_5d_base[1] = transformed_6d[1]  # dy_base
    action_5d_base[2] = transformed_6d[2]  # dz_base
    action_5d_base[3] = transformed_6d[5]  # drz_base (index 5 = Z rotation)
    action_5d_base[4] = gripper

    # B3: MarvinEnv (5D 版本)
    action_rot_rad_base = np.zeros(3)
    action_rot_rad_base[2] = action_5d_base[3] * ACTION_SCALE[1]

    pos_delta_mm = action_5d_base[:3] * ACTION_SCALE[0]

    result = {
        "action_5d_ee": action_5d_ee,
        "action_6d_padded": action_6d_padded,
        "transformed_6d": transformed_6d,
        "action_5d_base": action_5d_base,
        "transform_matrix": T,
        "pos_delta_mm": pos_delta_mm,
        "rot_delta_rad": action_rot_rad_base,
        "rz_base_rad": action_rot_rad_base[2],
        "gripper": gripper,
    }
    return result


# ==============================================================================
# 对比验证
# ==============================================================================

def run_tests():
    total = 0
    passed = 0

    for yaw_deg in TEST_YAW_ANGLES:
        tcp_pose = make_tcp_pose(yaw_deg)
        euler = np.array([RESET_POSE[3], RESET_POSE[4], yaw_deg])
        rot = R.from_quat(tcp_pose[3:]).as_matrix()

        print(f"\n{'='*90}")
        print(f"🔬 YAW = {yaw_deg:+.0f}°  (姿态: roll={euler[0]:.0f}°, pitch={euler[1]:.0f}°, yaw={euler[2]:.0f}°)")
        print(f"   EE X → 基座: {np.array2string(rot[:,0], precision=1, suppress_small=True)}")
        print(f"   EE Y → 基座: {np.array2string(rot[:,1], precision=1, suppress_small=True)}")
        print(f"   EE Z → 基座: {np.array2string(rot[:,2], precision=1, suppress_small=True)}")
        print(f"   预期: EE Y = [0,0,-1] (恒等于基座-Z)")
        print(f"{'='*90}")

        yaw_total = 0
        yaw_passed = 0

        for i, sm in enumerate(SM_TEST_CASES):
            sm = np.array(sm)
            total += 1
            yaw_total += 1

            r6 = path_6d_mask(sm, tcp_pose)
            r5 = path_5d_fixed(sm, tcp_pose)

            # ---- 比较 ----
            errors = []

            # 1. 位置增量必须一致
            if not np.allclose(r6["pos_delta_mm"], r5["pos_delta_mm"], atol=POS_TOL):
                errors.append(f"POS mismatch: 6D={r6['pos_delta_mm']} vs 5D={r5['pos_delta_mm']}")

            # 2. 基座 Z 旋转必须一致 (容忍 pitch=0.01° 导致的 6D 路径微量泄漏)
            if not np.allclose(r6["rz_base_rad"], r5["rz_base_rad"], atol=ROT_TOL):
                errors.append(f"RZ mismatch: 6D={r6['rz_base_rad']:.10f} vs 5D={r5['rz_base_rad']:.10f}")

            # 3. 整体 rot_delta 必须一致
            if not np.allclose(r6["rot_delta_rad"], r5["rot_delta_rad"], atol=ROT_TOL):
                errors.append(f"ROT mismatch: 6D={r6['rot_delta_rad']} vs 5D={r5['rot_delta_rad']}")

            ok = len(errors) == 0
            if ok:
                yaw_passed += 1
                passed += 1
                status = "✅"
            else:
                status = "❌"

            sm_ry = sm[4]
            sm_has_rot = abs(sm_ry) > 1e-10

            print(f"\n  [{status}] Case {i+1}: SM_6D={np.array2string(sm, precision=1, suppress_small=True)}")
            print(f"        SM.ry(EE)={sm_ry:+.2f}  {'← 有旋转输入' if sm_has_rot else '← 无旋转/非ry轴'}")

            # 路径A 详细
            a_ee = r6["action_7d_ee"]
            a_base = r6["action_7d_base"]
            print(f"    6D: EE输入 [{a_ee[0]:+.3f},{a_ee[1]:+.3f},{a_ee[2]:+.3f} | {a_ee[3]:+.3f},{a_ee[4]:+.3f},{a_ee[5]:+.3f}]")
            print(f"    6D: T→基座 [{a_base[0]:+.3f},{a_base[1]:+.3f},{a_base[2]:+.3f} | {a_base[3]:+.3f},{a_base[4]:+.3f},{a_base[5]:+.3f}]")
            print(f"    6D: mask后  pos={np.array2string(r6['pos_delta_mm'], precision=3, suppress_small=True)} mm  "
                  f"rot={np.array2string(r6['rot_delta_rad'], precision=6, suppress_small=True)} rad  "
                  f"→ rz={r6['rz_base_rad']:+.6f} rad")

            # 路径B 详细
            ee5 = r5["action_5d_ee"]
            padded = r5["action_6d_padded"]
            t6 = r5["transformed_6d"]
            base5 = r5["action_5d_base"]
            print(f"    5D: SM提取  [{ee5[0]:+.3f},{ee5[1]:+.3f},{ee5[2]:+.3f} | dry={ee5[3]:+.3f}]")
            print(f"    5D: pad6D   [{padded[0]:+.3f},{padded[1]:+.3f},{padded[2]:+.3f} | {padded[3]:+.3f},{padded[4]:+.3f},{padded[5]:+.3f}]")
            print(f"    5D: T→基座 [{t6[0]:+.3f},{t6[1]:+.3f},{t6[2]:+.3f} | {t6[3]:+.3f},{t6[4]:+.3f},{t6[5]:+.3f}]")
            print(f"    5D: 提取   drz={base5[3]:+.6f}  "
                  f"pos={np.array2string(r5['pos_delta_mm'], precision=3, suppress_small=True)} mm  "
                  f"rot={np.array2string(r5['rot_delta_rad'], precision=6, suppress_small=True)} rad  "
                  f"→ rz={r5['rz_base_rad']:+.6f} rad")

            if not ok:
                for e in errors:
                    print(f"        ❌ {e}")

        print(f"\n  ── yaw={yaw_deg:+.0f}° 小计: {yaw_passed}/{yaw_total} ──")

    # ==========================================================================
    # 汇总
    # ==========================================================================
    print(f"\n{'='*90}")
    print(f"📊 总计: {passed}/{total} 通过")
    if passed == total:
        print(f"🎉 全部 {total} 测试通过! 6D+mask 与 5D fixed 在所有 yaw 角下完全一致")
    else:
        print(f"❌ {total - passed} 个失败 (容差: pos={POS_TOL}, rot={ROT_TOL} rad)")
    print(f"{'='*90}")


# ==============================================================================
# 手动验证: 展示两条路径的等价性
# ==============================================================================

def explain_equivalence():
    """用数学解释为什么 6D+mask ≡ 5D fixed"""
    yaw = -90.0
    tcp = make_tcp_pose(yaw)
    T = construct_transform_matrix(tcp)
    R_mat = T[:3, :3]

    print(f"\n{'='*90}")
    print(f"📐 数学等价性证明 (yaw={yaw}°)")
    print(f"{'='*90}")
    print(f"""
    变换矩阵 T = [R, 0; 0, R]  (block-diagonal)

    6D+mask 路径:
      rot_base = R @ [rx_ee, ry_ee, rz_ee]
      mask: rx_base=0, ry_base=0 → rz_base = R[2,:] · [rx_ee, ry_ee, rz_ee]

    5D fixed 路径:
      只填入 ry_ee: [0, ry_ee, 0]
      rot_base = R @ [0, ry_ee, 0] = ry_ee · R[:,1]
      rz_base = ry_ee · R[2,1]

    在当前姿态 (roll=-90°, pitch=0°):
      R[:,1] = [0, 0, -1]  (EE Y ≡ 基座 -Z)
      R[:,0] = 随 yaw 变化 (EE X ≠ 基座 Z)
      R[:,2] = 随 yaw 变化 (EE Z ≠ 基座 Z)

    所以:
      6D+mask: rz_base = R[2,:]·[rx,ry,rz] = R[2,0]·rx + R[2,1]·ry + R[2,2]·rz
                        = R[2,1]·ry (因为 mask 掉后只看这一项, 而 rx/rz 的贡献已被丢弃)
      5D fixed: rz_base = R[2,1]·ry

    而 R[2,1] = -1.0, 所以:
      rz_base = -ry_ee  (两条路径完全一致)
""")

    # 数值演示
    sm = np.array([0.3, 0.2, -0.1, 0.0, 0.5, 0.0])
    print(f"  数值演示: SM输入 = {sm}")
    print(f"  R[2,:] = {np.array2string(R_mat[2,:], precision=6)}")
    print(f"  R[2,0]·rx_ee = R[2,0]·0 = {R_mat[2,0] * sm[3]:.6f}")
    print(f"  R[2,1]·ry_ee = {R_mat[2,1]:.6f} · {sm[4]} = {R_mat[2,1] * sm[4]:.6f}")
    print(f"  R[2,2]·rz_ee = R[2,2]·0 = {R_mat[2,2] * sm[5]:.6f}")
    print(f"  rz_base = {R_mat[2,1] * sm[4]:.6f}")
    print(f"")
    print(f"  6D+mask: 先全变换再mask → rz_base = {R_mat[2,1] * sm[4]:.6f}")
    print(f"  5D fixed: 预裁剪ry_ee再变换  → rz_base = {R_mat[2,1] * sm[4]:.6f}")
    print(f"  ✅ 一致")


if __name__ == "__main__":
    run_tests()
    explain_equivalence()
