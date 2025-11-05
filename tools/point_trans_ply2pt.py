import numpy as np
import torch
import open3d as o3d
import json
import os
from scipy.spatial.transform import Rotation as R

def pose_to_homogeneous_matrix(position, quaternion):
    """位置+四元数 → 4x4变换矩阵"""
    R_mat = R.from_quat(quaternion).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = position
    return T

def load_transform_json(json_file):
    """从JSON文件加载变换参数"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    position = data['position']  # [x, y, z]
    quaternion = data['quaternion']  # [qx, qy, qz, qw]
    return position, quaternion

def get_current_ee_to_base_transform(robot_ip="192.168.56.101"):
    """
    从机器人获取当前的末端执行器到基座变换
    """
    try:
        import rtde_receive
        
        print(f"   连接机器人: {robot_ip}")
        rtde = rtde_receive.RTDEReceiveInterface(robot_ip)
        
        # 获取当前TCP姿态
        tcp_pose = rtde.getActualTCPPose()
        if tcp_pose is None:
            raise ValueError("无法获取TCP姿态")
        
        position = tcp_pose[:3]
        rotvec = tcp_pose[3:]
        quaternion = R.from_rotvec(rotvec).as_quat()
        
        rtde.disconnect()
        
        print(f"   ✓ 获取当前末端姿态: {position}")
        return list(position), list(quaternion)
        
    except ImportError:
        print("   ❌ 缺少rtde_receive模块，无法连接机器人")
        return None, None
    except Exception as e:
        print(f"   ❌ 连接机器人失败: {e}")
        return None, None

def transform_to_ee_coordinates(input_ply, output_dir, cam_to_ee_file):
    """
    变换到末端执行器坐标系 (离线操作，不需要连接机器人)
    
    Args:
        input_ply: 输入PLY文件
        output_dir: 输出目录
        cam_to_ee_file: 相机到末端执行器的标定文件
    """
    print("=== 变换到末端执行器坐标系 ===")
    print("模式: 离线变换 (无需连接机器人)")
    
    # 1. 加载点云
    print("\n1. 加载PLY点云...")
    pcd = o3d.io.read_point_cloud(input_ply)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if len(pcd.colors) > 0 else np.ones((len(points), 3))
    
    print(f"   点数量: {len(points)}")
    print(f"   原始范围: X[{points[:,0].min():.3f}, {points[:,0].max():.3f}] "
          f"Y[{points[:,1].min():.3f}, {points[:,1].max():.3f}] "
          f"Z[{points[:,2].min():.3f}, {points[:,2].max():.3f}]")
    
    # 2. 加载相机到末端执行器的标定
    print(f"\n2. 加载手眼标定文件: {cam_to_ee_file}")
    if not os.path.exists(cam_to_ee_file):
        raise FileNotFoundError(f"标定文件不存在: {cam_to_ee_file}")
    
    cam_pos, cam_quat = load_transform_json(cam_to_ee_file)
    T_cam_ee = pose_to_homogeneous_matrix(cam_pos, cam_quat)
    
    print(f"   相机到末端执行器变换:")
    print(f"   位置: {cam_pos}")
    print(f"   四元数: {cam_quat}")
    
    # 3. 应用变换
    print("\n3. 应用坐标变换...")
    points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
    points_ee = (T_cam_ee @ points_homo.T).T[:, :3]
    
    print(f"   变换后范围: X[{points_ee[:,0].min():.3f}, {points_ee[:,0].max():.3f}] "
          f"Y[{points_ee[:,1].min():.3f}, {points_ee[:,1].max():.3f}] "
          f"Z[{points_ee[:,2].min():.3f}, {points_ee[:,2].max():.3f}]")
    
    # 4. 保存结果
    print(f"\n4. 保存到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    points_tensor = torch.from_numpy(points_ee.astype(np.float32))
    colors_tensor = torch.from_numpy(colors.astype(np.float32))
    
    torch.save(points_tensor, f"{output_dir}/points.pt")
    torch.save(colors_tensor, f"{output_dir}/colors.pt")
    
    # 保存变换信息
    transform_info = {
        "source_file": input_ply,
        "target_frame": "end_effector",
        "transform_chain": "camera -> end_effector",
        "T_cam_ee": T_cam_ee.tolist(),
        "timestamp": time.time()
    }
    
    with open(f"{output_dir}/transform_info.json", 'w') as f:
        json.dump(transform_info, f, indent=4)
    
    print(f"   ✓ points.pt: {points_tensor.shape}")
    print(f"   ✓ colors.pt: {colors_tensor.shape}")
    print(f"   ✓ transform_info.json")
    
    return points_ee, colors

def transform_to_baselink_coordinates(input_ply, output_dir, cam_to_ee_file, robot_ip="192.168.56.101", ee_to_base_file=None):
    """
    变换到基座坐标系 (在线操作，需要连接机器人或提供ee_to_base文件)
    
    Args:
        input_ply: 输入PLY文件
        output_dir: 输出目录
        cam_to_ee_file: 相机到末端执行器的标定文件
        robot_ip: 机器人IP (如果要实时获取姿态)
        ee_to_base_file: 末端到基座的变换文件 (可选，如果提供则不连接机器人)
    """
    print("=== 变换到基座坐标系 ===")
    
    # 1. 加载点云
    print("\n1. 加载PLY点云...")
    pcd = o3d.io.read_point_cloud(input_ply)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if len(pcd.colors) > 0 else np.ones((len(points), 3))
    
    print(f"   点数量: {len(points)}")
    
    # 2. 加载相机到末端执行器的标定
    print(f"\n2. 加载手眼标定文件: {cam_to_ee_file}")
    if not os.path.exists(cam_to_ee_file):
        raise FileNotFoundError(f"标定文件不存在: {cam_to_ee_file}")
    
    cam_pos, cam_quat = load_transform_json(cam_to_ee_file)
    T_cam_ee = pose_to_homogeneous_matrix(cam_pos, cam_quat)
    
    # 3. 获取末端执行器到基座的变换
    print(f"\n3. 获取末端执行器到基座变换...")
    
    if ee_to_base_file and os.path.exists(ee_to_base_file):
        # 方式1: 从文件加载 (离线模式)
        print(f"   模式: 从文件加载 ({ee_to_base_file})")
        ee_pos, ee_quat = load_transform_json(ee_to_base_file)
        transform_source = "file"
    else:
        # 方式2: 从机器人实时获取 (在线模式)
        print(f"   模式: 从机器人实时获取 ({robot_ip})")
        ee_pos, ee_quat = get_current_ee_to_base_transform(robot_ip)
        if ee_pos is None:
            raise RuntimeError("无法获取机器人姿态，请检查连接或提供ee_to_base文件")
        transform_source = "robot_realtime"
    
    T_ee_base = pose_to_homogeneous_matrix(ee_pos, ee_quat)
    
    print(f"   末端执行器到基座变换:")
    print(f"   位置: {ee_pos}")
    print(f"   四元数: {ee_quat}")
    
    # 4. 计算完整变换链
    print(f"\n4. 计算变换链: 相机 → 末端 → 基座")
    T_cam_base = T_ee_base @ T_cam_ee  # 相机坐标系在基座中的位姿
    print(f"   T_cam_base shape: {T_cam_base.shape}")

    # ⭐ 新增：计算点变换矩阵
    T_base_cam = np.linalg.inv(T_cam_base)  # 基座坐标系在相机中的位姿（用于点变换）
    print(f"   T_base_cam (逆矩阵) shape: {T_base_cam.shape}")

    # 5. 应用变换
    print(f"\n5. 应用坐标变换...")
    points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
    points_base = (T_base_cam @ points_homo.T).T[:, :3]  # ⭐ 使用逆矩阵变换点
    print(f"   变换后范围: X[{points_base[:,0].min():.3f}, {points_base[:,0].max():.3f}] "
          f"Y[{points_base[:,1].min():.3f}, {points_base[:,1].max():.3f}] "
          f"Z[{points_base[:,2].min():.3f}, {points_base[:,2].max():.3f}]")
    
    # 6. 保存结果
    print(f"\n6. 保存到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    points_tensor = torch.from_numpy(points_base.astype(np.float32))
    colors_tensor = torch.from_numpy(colors.astype(np.float32))
    
    torch.save(points_tensor, f"{output_dir}/points.pt")
    torch.save(colors_tensor, f"{output_dir}/colors.pt")
    
    # 保存详细的变换信息
    import time
    transform_info = {
        "source_file": input_ply,
        "target_frame": "base_link",
        "transform_chain": "camera -> end_effector -> base_link",
        "transform_source": transform_source,
        "robot_ip": robot_ip if transform_source == "robot_realtime" else None,
        "transforms": {
            "T_cam_ee": T_cam_ee.tolist(),
            "T_ee_base": T_ee_base.tolist(),
            "T_cam_base": T_cam_base.tolist()
        },
        "poses": {
            "cam_to_ee": {"position": cam_pos, "quaternion": cam_quat},
            "ee_to_base": {"position": ee_pos, "quaternion": ee_quat}
        },
        "timestamp": time.time()
    }
    
    with open(f"{output_dir}/transform_info.json", 'w') as f:
        json.dump(transform_info, f, indent=4)
    
    print(f"   ✓ points.pt: {points_tensor.shape}")
    print(f"   ✓ colors.pt: {colors_tensor.shape}")
    print(f"   ✓ transform_info.json")
    
    return points_base, colors

if __name__ == "__main__":
    import argparse
    import time
    
    # 默认路径
    INPUT_PLY = "/home/hkcrc/DCIM/rs1105/cloud.ply"  
    OUTPUT_DIR_EE = "/home/hkcrc/diffusion_edfs/diffusion_edf/dataset/1105_ee_pt/grasp0/raw" 
    OUTPUT_DIR_BASE = "/home/hkcrc/diffusion_edfs/diffusion_edf/dataset/1105_baselink_pt/grasp0/raw"
    CAM_TO_EE_FILE = "cam_to_ee_lmz.json"#需要更新
    EE_TO_BASE_FILE = "ee_to_base.json"
    ROBOT_IP = "192.168.56.101"
    
    parser = argparse.ArgumentParser(description="PLY点云坐标变换工具 (增强版)")
    parser.add_argument("--target", choices=['ee', 'baselink'], required=True,
                       help="目标坐标系: ee(末端执行器) 或 baselink(基座)")
    parser.add_argument("--input", default=INPUT_PLY, help="输入PLY文件路径")
    parser.add_argument("--output", help="输出目录 (默认根据target自动选择)")
    parser.add_argument("--cam-to-ee", default=CAM_TO_EE_FILE, help="相机到末端执行器标定文件")
    parser.add_argument("--ee-to-base", default=EE_TO_BASE_FILE, help="末端到基座变换文件 (可选)")
    parser.add_argument("--robot-ip", default=ROBOT_IP, help="机器人IP地址")
    
    args = parser.parse_args()
    
    
    # 设置输出目录
    if args.output:
        output_dir = args.output
    else:
        output_dir = OUTPUT_DIR_EE if args.target == 'ee' else OUTPUT_DIR_BASE
    
    try:
        if args.target == 'ee':
            # 转换到末端执行器坐标系 (离线)
            transform_to_ee_coordinates(
                input_ply=args.input,
                output_dir=output_dir,
                cam_to_ee_file=args.cam_to_ee
            )
            
        elif args.target == 'baselink':
            # 转换到基座坐标系 (在线或离线)
            transform_to_baselink_coordinates(
                input_ply=args.input,
                output_dir=output_dir,
                cam_to_ee_file=args.cam_to_ee,
                robot_ip=args.robot_ip,
                ee_to_base_file=args.ee_to_base if os.path.exists(args.ee_to_base) else None
            )
        
        print(f"\n✅ 转换完成!")
        print(f"输出目录: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ 转换失败: {e}")

"""
使用示例:

1. 创建示例文件:
   python point_trans_ply2pt.py --create-examples

2. 转换到末端执行器坐标系 (离线，仅需标定文件):
   python point_trans_ply2pt.py --target ee

3. 转换到基座坐标系 (在线，连接机器人):
   python point_trans_ply2pt.py --target baselink 

4. 转换到基座坐标系 (离线，使用保存的姿态文件):
   python point_trans_ply2pt.py --target baselink --ee-to-base saved_pose.json

5. 指定输入输出路径:
   python point_trans_ply2pt.py --target baselink --input my_cloud.ply --output my_output_dir
"""