#!/usr/bin/env python3
# filepath: /home/hkcrc/diffusion_edfs/diffusion_edf/edf_interface/tools/check_contect.py
import argparse
from pathlib import Path
import torch
import numpy as np

from edf_interface.data import PointCloud, SE3
from edf_interface.modules.transform import TransformManager


def parse_target_pose(pose_str: str) -> tuple:
    """
    解析目标位姿字符串
    
    Args:
        pose_str: "x, y, z, rx, ry, rz" 格式的字符串
    
    Returns:
        (position, rotation_xyz): 位置和欧拉角（弧度）
    """
    values = [float(x.strip()) for x in pose_str.split(",")]
    if len(values) != 6:
        raise ValueError(f"Expected 6 values (x,y,z,rx,ry,rz), got {len(values)}")
    
    position = values[:3]
    rotation_xyz = values[3:]
    return position, rotation_xyz


def euler_to_quaternion(rx: float, ry: float, rz: float) -> list:
    """
    将欧拉角（XYZ顺序）转换为四元数 [w, x, y, z]
    
    Args:
        rx, ry, rz: 绕 X, Y, Z 轴的旋转角度（弧度）
    
    Returns:
        [w, x, y, z] 格式的四元数
    """
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler('xyz', [rx, ry, rz], degrees=False)
    quat_xyzw = R.as_quat()  # scipy 返回 [x, y, z, w]
    quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]  # 转为 [w, x, y, z]
    return quat_wxyz


def load_pointcloud(path: str) -> PointCloud:
    """
    加载点云文件（支持 .pt 和 .ply 格式）
    
    Args:
        path: 点云文件路径或包含点云的目录
    
    Returns:
        PointCloud 对象
    """
    path = Path(path)
    
    if path.is_dir():
        # 检查是否包含 points.pt 和 colors.pt
        points_file = path / "points.pt"
        colors_file = path / "colors.pt"
        
        if points_file.exists():
            print(f"Loading point cloud from directory: {path}")
            
            # 加载 points
            points_data = torch.load(str(points_file))
            if isinstance(points_data, np.ndarray):
                points = torch.from_numpy(points_data).float()
            elif isinstance(points_data, torch.Tensor):
                points = points_data.float()
            else:
                raise ValueError(f"Unsupported points data type: {type(points_data)}")
            
            # 加载 colors（如果存在）
            if colors_file.exists():
                colors_data = torch.load(str(colors_file))
                if isinstance(colors_data, np.ndarray):
                    colors = torch.from_numpy(colors_data).float()
                elif isinstance(colors_data, torch.Tensor):
                    colors = colors_data.float()
                else:
                    raise ValueError(f"Unsupported colors data type: {type(colors_data)}")
            else:
                print("  No colors.pt found, using default gray color")
                colors = torch.ones_like(points) * 0.5
            
            pcd = PointCloud(points=points, colors=colors)
            print(f"  Loaded {pcd.points.shape[0]} points")
            return pcd
        else:
            # 查找单个 .ply 文件
            ply_files = list(path.glob("*.ply"))
            if ply_files:
                path = ply_files[0]
            else:
                raise FileNotFoundError(f"No points.pt or .ply files found in {path}")
    
    # 处理单个文件
    print(f"Loading point cloud file: {path}")
    
    if path.suffix == ".ply":
        import open3d as o3d
        pcd_o3d = o3d.io.read_point_cloud(str(path))
        pcd = PointCloud.from_o3d(pcd_o3d)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    print(f"  Loaded {pcd.points.shape[0]} points")
    return pcd


def main():
    parser = argparse.ArgumentParser(
        description="将抓取点云变换到目标位姿并与场景点云拼接"
    )
    parser.add_argument(
        "--grasp_pcd",
        type=str,
        default="/home/hkcrc/diffusion_edfs/diffusion_edf/edf_interface/run_sessions/grasp/20251113_154123/grasp_pcd",
        help="抓取点云路径",
    )   
    parser.add_argument(
        "--scene_pcd",
        type=str,
        default="/home/hkcrc/diffusion_edfs/diffusion_edf/edf_interface/run_sessions/scene/20251113_170007/scene_pcd",
        help="场景点云路径",
    )
    parser.add_argument(
        "--target_poses",
        type=str,
        default="-0.047, 0.368, 0.177, 2.936, 0.018, -2.27",
        help="目标位姿 (x, y, z, rx, ry, rz)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="输出目录",
    )
    args = parser.parse_args()

    # 1. 加载点云
    print("\n========== Step 1: 加载点云 ==========")
    grasp_pcd = load_pointcloud(args.grasp_pcd)
    scene_pcd = load_pointcloud(args.scene_pcd)

    # 2. 解析目标位姿
    print("\n========== Step 2: 解析目标位姿 ==========")
    position, rotation_xyz = parse_target_pose(args.target_poses)
    print(f"Target position: {position}")
    print(f"Target rotation (xyz, rad): {rotation_xyz}")
    
    # 转换为四元数 [w, x, y, z]
    quat_wxyz = euler_to_quaternion(*rotation_xyz)
    print(f"Target quaternion (wxyz): {quat_wxyz}")

    # 3. 构建变换矩阵 SE3
    print("\n========== Step 3: 变换抓取点云 ==========")
    T_target = SE3(torch.tensor([*quat_wxyz, *position], dtype=torch.float32))
    print(f"Transform SE3: {T_target.poses}")
    
    # 变换抓取点云
    grasp_pcd_transformed = grasp_pcd.transformed(T_target, squeeze=True)
    print(f"  Transformed grasp point cloud: {grasp_pcd_transformed.points.shape[0]} points")

    # 4. 拼接点云（保留原始颜色）
    print("\n========== Step 4: 拼接点云 ==========")
    combined_points = torch.cat([scene_pcd.points, grasp_pcd_transformed.points], dim=0)
    combined_colors = torch.cat([scene_pcd.colors, grasp_pcd_transformed.colors], dim=0)
    combined_pcd = PointCloud(points=combined_points, colors=combined_colors)
    print(f"Combined point cloud: {combined_pcd.points.shape[0]} points")
    print(f"  - Scene: {scene_pcd.points.shape[0]} points")
    print(f"  - Grasp: {grasp_pcd_transformed.points.shape[0]} points")

    # 5. 保存结果
    print("\n========== Step 5: 保存结果 ==========")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_pcd.save(str(output_dir))
    print(f"Saved to: {output_dir}")

    # 6. 可视化
    print("\n========== Step 6: 可视化 ==========")
    try:
        fig = combined_pcd.show(
            point_size=2.0,
            name="Combined Point Cloud (Original Colors)"
        )
        fig.show()
        print("Visualization opened (close the window to continue)")
    except Exception as e:
        print(f"Visualization failed: {e}")

    print("\n========== 完成 ==========")


if __name__ == "__main__":
    main()