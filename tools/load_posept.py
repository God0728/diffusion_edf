import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def read_and_analyze_poses(poses_path):
    """
    读取并分析poses.pt文件中的6D姿态数据
    
    Args:
        poses_path: poses.pt文件路径
    """
    # 读取poses文件
    poses = torch.load(poses_path).cpu().numpy()
    
    print("=== Poses 数据分析 ===")
    print(f"数据类型: {type(poses)}")
    print(f"数据形状: {poses.shape}")
    print(f"数据范围: {poses.min():.6f} 到 {poses.max():.6f}")
    print(f"\n原始数据:")
    print(poses)
    
    # 根据数据形状判断可能的格式
    if len(poses.shape) == 1:
        if poses.shape[0] == 6:
            print("\n可能的格式: 单个6D pose [x, y, z, rx, ry, rz]")
            analyze_single_6d_pose(poses)
        elif poses.shape[0] == 7:
            print("\n格式: 单个7D pose [qx, qy, qz, qw, x, y, z]")
            analyze_single_7d_pose(poses)
        else:
            print(f"\n未知格式: 1D数组，长度为 {poses.shape[0]}")
            
    elif len(poses.shape) == 2:
        if poses.shape[1] == 6:
            print(f"\n可能的格式: {poses.shape[0]}个6D poses, 每个包含 [x, y, z, rx, ry, rz]")
            analyze_multiple_6d_poses(poses)
        elif poses.shape[1] == 7:
            print(f"\n格式: {poses.shape[0]}个7D poses, 每个包含 [qx, qy, qz, qw, x, y, z]")
            analyze_multiple_7d_poses(poses)
        elif poses.shape[1] == 3:
            print(f"\n可能的格式: {poses.shape[0]}个3D位置 [x, y, z]")
            analyze_positions(poses)
        else:
            print(f"\n未知格式: 2D数组，形状为 {poses.shape}")
            
    elif len(poses.shape) == 3:
        if poses.shape[1:] == (4, 4):
            print(f"\n可能的格式: {poses.shape[0]}个4x4变换矩阵")
            analyze_transformation_matrices(poses)
        else:
            print(f"\n未知格式: 3D数组，形状为 {poses.shape}")
    
    return poses

def analyze_single_6d_pose(pose):
    """分析单个6D姿态"""
    position = pose[:3]
    rotation = pose[3:]
    
    print(f"位置 (x, y, z): {position}")
    print(f"旋转 (rx, ry, rz): {rotation}")
    
    # 假设旋转是欧拉角（弧度）
    try:
        rot_matrix = R.from_euler('xyz', rotation).as_matrix()
        print(f"旋转矩阵:\n{rot_matrix}")
    except:
        print("无法解析为欧拉角")

def analyze_single_7d_pose(pose):
    """分析单个7D姿态（四元数+位置）格式: [qx, qy, qz, qw, x, y, z]"""
    quaternion = pose[:4]  # 前4个是四元数
    position = pose[4:]    # 后3个是位置
    
    print(f"四元数 (qx, qy, qz, qw): {quaternion}")
    print(f"位置 (x, y, z): {position}")
    
    try:
        # 检查四元数是否归一化
        quat_norm = np.linalg.norm(quaternion)
        print(f"四元数模长: {quat_norm:.6f}")
        
        if abs(quat_norm - 1.0) > 0.1:
            print("警告: 四元数可能未归一化")
        
        rot_matrix = R.from_quat(quaternion).as_matrix()
        euler = R.from_quat(quaternion).as_euler('xyz', degrees=True)
        print(f"旋转矩阵:\n{rot_matrix}")
        print(f"欧拉角 (度): {euler}")
        
        # 构建4x4变换矩阵
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rot_matrix
        transform_matrix[:3, 3] = position
        print(f"\n4x4变换矩阵:\n{transform_matrix}")
        
    except Exception as e:
        print(f"无法解析四元数: {e}")

def analyze_multiple_6d_poses(poses):
    """分析多个6D姿态"""
    positions = poses[:, :3]
    rotations = poses[:, 3:]
    
    print(f"\n位置统计:")
    print(f"  X范围: {positions[:, 0].min():.6f} 到 {positions[:, 0].max():.6f}")
    print(f"  Y范围: {positions[:, 1].min():.6f} 到 {positions[:, 1].max():.6f}")
    print(f"  Z范围: {positions[:, 2].min():.6f} 到 {positions[:, 2].max():.6f}")
    
    print(f"\n旋转统计:")
    print(f"  RX范围: {rotations[:, 0].min():.6f} 到 {rotations[:, 0].max():.6f}")
    print(f"  RY范围: {rotations[:, 1].min():.6f} 到 {rotations[:, 1].max():.6f}")
    print(f"  RZ范围: {rotations[:, 2].min():.6f} 到 {rotations[:, 2].max():.6f}")
    
    # 显示每个pose
    for i, pose in enumerate(poses):
        print(f"\nPose {i}: 位置{pose[:3]}, 旋转{pose[3:]}")

def analyze_multiple_7d_poses(poses):
    """分析多个7D姿态 格式: [qx, qy, qz, qw, x, y, z]"""
    quaternions = poses[:, :4]  # 前4列是四元数
    positions = poses[:, 4:]    # 后3列是位置
    
    print(f"\n四元数统计:")
    quat_norms = np.linalg.norm(quaternions, axis=1)
    print(f"  四元数模长范围: {quat_norms.min():.6f} 到 {quat_norms.max():.6f}")
    print(f"  平均模长: {quat_norms.mean():.6f}")
    
    print(f"\n位置统计:")
    print(f"  X范围: {positions[:, 0].min():.6f} 到 {positions[:, 0].max():.6f}")
    print(f"  Y范围: {positions[:, 1].min():.6f} 到 {positions[:, 1].max():.6f}")
    print(f"  Z范围: {positions[:, 2].min():.6f} 到 {positions[:, 2].max():.6f}")
    
    # 显示每个pose的详细信息
    for i, pose in enumerate(poses):
        quaternion = pose[:4]
        position = pose[4:]
        print(f"\nPose {i}:")
        print(f"  四元数(qx,qy,qz,qw): {quaternion}")
        print(f"  位置(x,y,z): {position}")
        
        try:
            euler = R.from_quat(quaternion).as_euler('xyz', degrees=True)
            print(f"  欧拉角(度): {euler}")
        except:
            print(f"  无法解析四元数")

def analyze_positions(positions):
    """分析3D位置数据"""
    print(f"\n位置统计:")
    print(f"  X范围: {positions[:, 0].min():.6f} 到 {positions[:, 0].max():.6f}")
    print(f"  Y范围: {positions[:, 1].min():.6f} 到 {positions[:, 1].max():.6f}")
    print(f"  Z范围: {positions[:, 2].min():.6f} 到 {positions[:, 2].max():.6f}")
    
    for i, pos in enumerate(positions):
        print(f"位置 {i}: {pos}")

def analyze_transformation_matrices(matrices):
    """分析变换矩阵"""
    print(f"\n{len(matrices)}个4x4变换矩阵:")
    
    for i, matrix in enumerate(matrices):
        print(f"\n变换矩阵 {i}:")
        print(matrix)
        
        # 提取位置和旋转
        position = matrix[:3, 3]
        rotation_matrix = matrix[:3, :3]
        
        print(f"位置: {position}")
        try:
            euler = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)
            print(f"欧拉角 (度): {euler}")
        except:
            print("无法解析旋转矩阵")

def visualize_poses_3d(poses):
    """3D可视化姿态数据"""
    if len(poses.shape) == 2 and poses.shape[1] >= 3:
        # 根据数据格式选择位置
        if poses.shape[1] == 7:
            # [qx, qy, qz, qw, x, y, z] 格式，位置在后3列
            positions = poses[:, 4:]
        else:
            # 假设位置在前3列
            positions = poses[:, :3]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                  c=range(len(positions)), cmap='viridis', s=100)
        
        # 添加标号
        for i, pos in enumerate(positions):
            ax.text(pos[0], pos[1], pos[2], f'  {i}', fontsize=8)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Poses 3D Visualization')
        
        # 显示颜色条
        scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                           c=range(len(positions)), cmap='viridis', s=100)
        plt.colorbar(scatter)
        plt.show()
    elif len(poses.shape) == 1 and poses.shape[0] == 7:
        # 单个pose的情况
        position = poses[4:]
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(position[0], position[1], position[2], c='red', s=200)
        ax.text(position[0], position[1], position[2], '  Pose', fontsize=12)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Single Pose 3D Visualization')
        plt.show()

if __name__ == "__main__":
    # 指定poses.pt文件路径
    poses_path = "/home/hkcrc/diffusion_edfs/diffusion_edf/demo/rebar_grasping/data/demo_1/step_0/target_poses/poses.pt"
    
    # 读取和分析poses数据
    poses_data = read_and_analyze_poses(poses_path)
    
    # 如果有数据，进行3D可视化
    try:
        print("\n是否显示3D可视化? (y/n): ", end="")
        if input().lower() == 'y':
            visualize_poses_3d(poses_data)
    except:
        print("无法进行3D可视化")