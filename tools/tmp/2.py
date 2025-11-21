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