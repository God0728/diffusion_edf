#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四元数转欧拉角工具
"""
import numpy as np
from scipy.spatial.transform import Rotation


def quaternion_to_euler(quat, input_format='wxyz', output_unit='radians'):
    """
    将四元数转换为欧拉角（XYZ顺序）
    
    Args:
        quat: 四元数，列表或数组
        input_format: 'wxyz' 或 'xyzw'
        output_unit: 'radians' 或 'degrees'
    
    Returns:
        (rx, ry, rz): 欧拉角
    """
    # 转换为 scipy 格式 [x, y, z, w]
    if input_format == 'wxyz':
        quat_xyzw = [quat[1], quat[2], quat[3], quat[0]]
    elif input_format == 'xyzw':
        quat_xyzw = quat
    else:
        raise ValueError(f"Unknown format: {input_format}")
    
    # 转换为旋转对象
    R = Rotation.from_quat(quat_xyzw)
    
    # 转换为欧拉角（XYZ顺序）
    euler_xyz = R.as_euler('xyz', degrees=(output_unit == 'degrees'))
    
    return euler_xyz


# 你的四元数
quat_xyzw = [0.03484342008409241, 0.4199214537853403, -0.9017416976659595, -0.09650813135774128]

print("输入四元数 [x, y, z, w]:", quat_xyzw)
print("="*60)

# 转换为欧拉角（弧度）
euler_rad = quaternion_to_euler(quat_xyzw, input_format='wxyz', output_unit='radians')
print(f"\n欧拉角 (弧度):")
print(f"  rx = {euler_rad[0]:.6f} rad")
print(f"  ry = {euler_rad[1]:.6f} rad")
print(f"  rz = {euler_rad[2]:.6f} rad")

# 转换为欧拉角（角度）
euler_deg = quaternion_to_euler(quat_xyzw, input_format='wxyz', output_unit='degrees')
print(f"\n欧拉角 (角度):")
print(f"  rx = {euler_deg[0]:.3f}°")
print(f"  ry = {euler_deg[1]:.3f}°")
print(f"  rz = {euler_deg[2]:.3f}°")

# 验证：转换回四元数
print(f"\n验证（转换回四元数）:")
R_verify = Rotation.from_euler('xyz', euler_rad, degrees=False)
quat_verify = R_verify.as_quat()  # [x, y, z, w]
print(f"  原始: {quat_xyzw}")
print(f"  验证: {[quat_verify[0], quat_verify[1], quat_verify[2], quat_verify[3]]}")
print(f"  误差: {np.linalg.norm(np.array(quat_xyzw) - quat_verify):.2e}")