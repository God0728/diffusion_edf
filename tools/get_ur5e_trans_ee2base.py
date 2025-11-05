#!/usr/bin/env python
import rtde_receive
import numpy as np
import json
import time
from scipy.spatial.transform import Rotation as R

class UR5eKinematics:
    """UR5e机器人正运动学计算"""
    
    def __init__(self):
        # UR5e DH参数 (标准DH参数)
        # [a, alpha, d, theta_offset]
        self.dh_params = np.array([
            [0,         np.pi/2,    0.1625,   0],      # Joint 1
            [-0.425,    0,          0,        0],      # Joint 2  
            [-0.39225,  0,          0,        0],      # Joint 3
            [0,         np.pi/2,    0.1333,   0],      # Joint 4
            [0,         -np.pi/2,   0.0997,   0],      # Joint 5
            [0,         0,          0.0996,   0],      # Joint 6
        ])
        
        print("UR5e DH参数表:")
        print("Joint |    a     |  alpha   |    d     | theta_offset")
        print("------|----------|----------|----------|-------------")
        for i, params in enumerate(self.dh_params):
            print(f"  {i+1}   | {params[0]:8.5f} | {params[1]:8.5f} | {params[2]:8.5f} | {params[3]:8.5f}")
    
    def dh_transform(self, a, alpha, d, theta):
        """
        计算单个关节的DH变换矩阵
        
        Args:
            a: 连杆长度
            alpha: 连杆扭转角
            d: 关节偏移
            theta: 关节角度
        
        Returns:
            4x4变换矩阵
        """
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        T = np.array([
            [ct,    -st*ca,   st*sa,    a*ct],
            [st,     ct*ca,  -ct*sa,    a*st],
            [0,      sa,      ca,       d   ],
            [0,      0,       0,        1   ]
        ])
        
        return T
    
    def forward_kinematics(self, joint_angles):
        """
        计算正运动学，从关节角度到末端执行器位姿
        
        Args:
            joint_angles: [q1, q2, q3, q4, q5, q6] 关节角度 (弧度)
        
        Returns:
            T_ee_base: 4x4变换矩阵 (末端执行器相对于基座)
        """
        if len(joint_angles) != 6:
            raise ValueError("需要6个关节角度")
        
        # 初始化为单位矩阵
        T = np.eye(4)
        
        print("\n=== 正运动学计算过程 ===")
        print(f"输入关节角度 (度): {[np.degrees(q) for q in joint_angles]}")
        print(f"输入关节角度 (弧度): {joint_angles}")
        
        # 逐个关节计算变换矩阵
        for i in range(6):
            a, alpha, d, theta_offset = self.dh_params[i]
            theta = joint_angles[i] + theta_offset
            
            T_i = self.dh_transform(a, alpha, d, theta)
            T = T @ T_i
            
            print(f"\n关节 {i+1} 变换矩阵 T_{i+1}:")
            print(f"  DH参数: a={a:.5f}, alpha={alpha:.5f}, d={d:.5f}, theta={theta:.5f}")
            print(T_i)
            
            print(f"\n累积变换矩阵 T_0^{i+1}:")
            print(T)
        
        return T
    
    def extract_pose_from_matrix(self, T):
        """
        从变换矩阵提取位置和姿态
        
        Args:
            T: 4x4变换矩阵
        
        Returns:
            position: [x, y, z] 位置
            quaternion: [qx, qy, qz, qw] 四元数
            rotvec: [rx, ry, rz] 旋转向量
        """
        # 提取位置
        position = T[:3, 3]
        
        # 提取旋转矩阵并转换为四元数和旋转向量
        rotation_matrix = T[:3, :3]
        rotation = R.from_matrix(rotation_matrix)
        quaternion = rotation.as_quat()  # [qx, qy, qz, qw]
        rotvec = rotation.as_rotvec()    # [rx, ry, rz]
        
        return position, quaternion, rotvec

def get_joint_angles_from_robot(robot_ip="192.168.56.101"):
    """
    从UR5e机器人获取当前关节角度
    
    Args:
        robot_ip: 机器人IP地址
    
    Returns:
        joint_angles: [q1, q2, q3, q4, q5, q6] 关节角度 (弧度)
        tcp_pose: [x, y, z, rx, ry, rz] TCP姿态 (用于对比)
    """
    try:
        rtde = rtde_receive.RTDEReceiveInterface(robot_ip)
        print(f"✓ 连接UR5e成功: {robot_ip}")
        
        # 获取关节角度
        joint_angles = rtde.getActualQ()
        
        # 获取TCP姿态 (用于对比)
        tcp_pose = rtde.getActualTCPPose()
        
        rtde.disconnect()
        
        if joint_angles is None:
            raise ValueError("无法获取关节角度")
        
        print(f"✓ 获取关节角度成功")
        print(f"关节角度 (度): {[np.degrees(q) for q in joint_angles]}")
        print(f"关节角度 (弧度): {joint_angles}")
        
        if tcp_pose is not None:
            print(f"实际TCP姿态: {tcp_pose}")
        
        return list(joint_angles), list(tcp_pose) if tcp_pose else None
        
    except Exception as e:
        print(f"❌ 获取关节角度失败: {e}")
        return None, None

def compare_forward_kinematics_with_actual(robot_ip="192.168.56.101"):
    """
    对比正运动学计算结果与实际TCP姿态
    """
    print("=== 正运动学验证 ===")
    
    # 1. 获取关节角度和实际TCP姿态
    joint_angles, actual_tcp = get_joint_angles_from_robot(robot_ip)
    
    if joint_angles is None:
        print("❌ 无法获取机器人数据")
        return
    
    # 2. 初始化运动学对象
    ur5e = UR5eKinematics()
    
    # 3. 计算正运动学
    T_ee_base = ur5e.forward_kinematics(joint_angles)
    
    # 4. 提取计算得到的位姿
    calc_position, calc_quaternion, calc_rotvec = ur5e.extract_pose_from_matrix(T_ee_base)
    
    print(f"\n=== 结果对比 ===")
    print("正运动学计算结果:")
    print(f"  位置: [{calc_position[0]:.6f}, {calc_position[1]:.6f}, {calc_position[2]:.6f}]")
    print(f"  旋转向量: [{calc_rotvec[0]:.6f}, {calc_rotvec[1]:.6f}, {calc_rotvec[2]:.6f}]")
    print(f"  四元数: [{calc_quaternion[0]:.6f}, {calc_quaternion[1]:.6f}, {calc_quaternion[2]:.6f}, {calc_quaternion[3]:.6f}]")
    
    if actual_tcp is not None:
        actual_position = actual_tcp[:3]
        actual_rotvec = actual_tcp[3:]
        actual_quaternion = R.from_rotvec(actual_rotvec).as_quat()
        
        print("\n机器人实际TCP姿态:")
        print(f"  位置: [{actual_position[0]:.6f}, {actual_position[1]:.6f}, {actual_position[2]:.6f}]")
        print(f"  旋转向量: [{actual_rotvec[0]:.6f}, {actual_rotvec[1]:.6f}, {actual_rotvec[2]:.6f}]")
        print(f"  四元数: [{actual_quaternion[0]:.6f}, {actual_quaternion[1]:.6f}, {actual_quaternion[2]:.6f}, {actual_quaternion[3]:.6f}]")
        
        # 计算误差
        pos_error = np.linalg.norm(np.array(calc_position) - np.array(actual_position))
        rot_error = np.linalg.norm(np.array(calc_rotvec) - np.array(actual_rotvec))
        
        print(f"\n误差分析:")
        print(f"  位置误差: {pos_error:.6f} m")
        print(f"  旋转误差: {rot_error:.6f} rad ({np.degrees(rot_error):.3f} deg)")
        
        if pos_error < 0.001 and rot_error < 0.01:
            print("✅ 正运动学计算准确!")
        else:
            print("⚠️ 正运动学计算存在误差，可能需要校准DH参数")
    
    print(f"\n最终变换矩阵 (末端执行器 → 基座):")
    print(T_ee_base)
    
    return T_ee_base, calc_position, calc_quaternion

def save_kinematics_transform(robot_ip="192.168.56.101", output_file="ee_to_base_kinematics.json"):
    """
    使用正运动学计算并保存变换矩阵
    """
    print("=== 基于正运动学的变换矩阵计算 ===")
    
    # 获取关节角度
    joint_angles, _ = get_joint_angles_from_robot(robot_ip)
    
    if joint_angles is None:
        print("❌ 无法获取关节角度")
        return
    
    # 计算正运动学
    ur5e = UR5eKinematics()
    T_ee_base = ur5e.forward_kinematics(joint_angles)
    position, quaternion, rotvec = ur5e.extract_pose_from_matrix(T_ee_base)
    
    # 保存到JSON
    transform_data = {
        "position": position.tolist(),
        "quaternion": quaternion.tolist(),
        "rotvec": rotvec.tolist(),
        "joint_angles": joint_angles,
        "timestamp": time.time(),
        "method": "forward_kinematics",
        "description": "UR5e end-effector to base transform computed by forward kinematics"
    }
    
    with open(output_file, 'w') as f:
        json.dump(transform_data, f, indent=4)
    
    print(f"✓ 正运动学变换已保存到: {output_file}")
    
    return T_ee_base

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="UR5e正运动学计算工具")
    parser.add_argument("--ip", default="192.168.56.101", help="UR5e机器人IP地址")
    parser.add_argument("--action", choices=['compare', 'save', 'test'], default='compare',
                       help="操作类型: compare(对比验证), save(计算并保存), test(测试DH参数)")
    parser.add_argument("--output", default="ee_to_base_kinematics.json", help="输出文件名")
    
    args = parser.parse_args()
    
    if args.action == 'compare':
        compare_forward_kinematics_with_actual(args.ip)
        
    elif args.action == 'save':
        save_kinematics_transform(args.ip, args.output)
        
    elif args.action == 'test':
        # 测试DH参数
        ur5e = UR5eKinematics()
        
        # 使用零位测试
        zero_angles = [0, 0, 0, 0, 0, 0]
        print("\n=== 零位测试 ===")
        T_zero = ur5e.forward_kinematics(zero_angles)
        pos, quat, rotvec = ur5e.extract_pose_from_matrix(T_zero)
        print(f"零位末端位置: {pos}")

"""
使用示例:

1. 对比正运动学与实际姿态:
   python ur5e_forward_kinematics.py --action compare

2. 计算并保存变换矩阵:
   python ur5e_forward_kinematics.py --action save

3. 测试DH参数:
   python ur5e_forward_kinematics.py --action test

4. 指定机器人IP:
   python ur5e_forward_kinematics.py --ip 192.168.1.100 --action compare
"""