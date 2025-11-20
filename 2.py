#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试脚本：测试单个位姿是否可达
"""
import sys
import torch
import numpy as np
from pathlib import Path

# 添加路径
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "edf_interface"))
sys.path.insert(0, str(PROJECT_ROOT / "edf_interface/examples"))

from edf_interface.data import SE3
from edf_interface.modules.robot import RobotInterface


def test_single_pose():
    """测试失败的目标位姿"""
    
    # 从错误日志中提取的目标位姿
    target_pos = [-0.04940994083881378, 0.32348474860191345, 0.16627947986125946]
    target_quat_xyzw = [-0.24454217, 0.9668187, 0.040123098, -0.062055662]
    
    print("=" * 60)
    print("🎯 目标位姿信息")
    print("=" * 60)
    print(f"位置 (x,y,z): {target_pos}")
    print(f"四元数 (x,y,z,w): {target_quat_xyzw}")
    print()
    
    # 检查四元数归一化
    quat_norm = np.linalg.norm(target_quat_xyzw)
    print(f"📐 四元数模长: {quat_norm:.6f}")
    if abs(quat_norm - 1.0) > 0.01:
        print(f"⚠️  警告: 四元数未归一化! (应为1.0)")
        target_quat_xyzw = (np.array(target_quat_xyzw) / quat_norm).tolist()
        print(f"✓ 归一化后: {target_quat_xyzw}")
    else:
        print(f"✓ 四元数已归一化")
    print()
    
    # 检查位置范围
    print("📏 位置分析:")
    print(f"  X: {target_pos[0]:.4f} m")
    print(f"  Y: {target_pos[1]:.4f} m")
    print(f"  Z: {target_pos[2]:.4f} m (高度)")
    
    if target_pos[2] < 0.10:
        print(f"  ⚠️  Z 高度过低! 可能碰撞工作台")
    elif target_pos[2] < 0.15:
        print(f"  ⚠️  Z 高度偏低，需注意安全")
    else:
        print(f"  ✓ Z 高度合理")
    print()
    
    # 计算到基座的距离
    dist_xy = np.sqrt(target_pos[0]**2 + target_pos[1]**2)
    dist_3d = np.sqrt(target_pos[0]**2 + target_pos[1]**2 + target_pos[2]**2)
    print(f"📍 到基座距离:")
    print(f"  XY 平面: {dist_xy:.4f} m")
    print(f"  3D 空间: {dist_3d:.4f} m")
    
    # UR5e 工作半径约 850mm
    if dist_xy > 0.85:
        print(f"  ⚠️  超出 UR5e 工作半径 (850mm)!")
    else:
        print(f"  ✓ 在工作半径内")
    print()
    
    # 连接机器人
    print("=" * 60)
    print("🤖 连接机器人")
    print("=" * 60)
    robot = RobotInterface(robot_ip="192.168.56.101")
    print("✓ 连接成功")
    print()
    
    # 获取当前位姿
    print("=" * 60)
    print("📍 当前位姿")
    print("=" * 60)
    current_pos, current_quat = robot.get_current_pose()
    print(f"位置: {current_pos}")
    print(f"四元数: {current_quat}")
    print()
    
    # 计算位移距离
    pos_diff = np.array(target_pos) - np.array(current_pos)
    move_dist = np.linalg.norm(pos_diff)
    print(f"📏 需要移动的距离: {move_dist:.4f} m")
    print(f"   ΔX: {pos_diff[0]:+.4f} m")
    print(f"   ΔY: {pos_diff[1]:+.4f} m")
    print(f"   ΔZ: {pos_diff[2]:+.4f} m")
    print()
    
    # 尝试移动
    print("=" * 60)
    print("🚀 尝试移动到目标位姿")
    print("=" * 60)
    
    velocity = 0.1  # 降低速度
    acceleration = 0.5  # 降低加速度
    
    print(f"速度: {velocity} rad/s")
    print(f"加速度: {acceleration} rad/s²")
    print()
    
    print("发送指令...")
    success = robot.move_to_pose(
        position=target_pos,
        quaternion=target_quat_xyzw,
        velocity=velocity,
        acceleration=acceleration,
        wait=True
    )
    
    print()
    if success:
        print("✅ 移动成功!")
        final_pos, final_quat = robot.get_current_pose()
        print(f"最终位置: {final_pos}")
        print(f"最终四元数: {final_quat}")
        
        # 计算误差
        pos_error = np.linalg.norm(np.array(final_pos) - np.array(target_pos))
        print(f"\n位置误差: {pos_error*1000:.2f} mm")
    else:
        print("❌ 移动失败!")
        print("\n可能的原因:")
        print("  1. 目标位姿超出工作空间")
        print("  2. 运动学奇异点")
        print("  3. 碰撞检测触发")
        print("  4. 关节限位")
        print("\n请在示教器上查看详细错误信息")
    
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_single_pose()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()