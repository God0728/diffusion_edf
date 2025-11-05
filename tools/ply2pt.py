import torch
import open3d as o3d
import numpy as np
import os

def convert_ply_to_pt(ply_file_path, output_dir):
    """
    将PLY文件转换为points.pt和colors.pt格式
    
    Args:
        ply_file_path: PLY文件路径
        output_dir: 输出目录路径
    """
    # 读取PLY文件
    pcd = o3d.io.read_point_cloud(ply_file_path)
    
    # 检查点云是否为空
    if len(pcd.points) == 0:
        print("错误：PLY文件中没有点云数据")
        return
    
    # 提取点坐标
    points = np.asarray(pcd.points)
    print(f"点云包含 {len(points)} 个点")
    print(f"坐标范围: X({points[:, 0].min():.3f}, {points[:, 0].max():.3f}) "
          f"Y({points[:, 1].min():.3f}, {points[:, 1].max():.3f}) "
          f"Z({points[:, 2].min():.3f}, {points[:, 2].max():.3f})")
    
    # 提取颜色信息
    if len(pcd.colors) > 0:
        colors = np.asarray(pcd.colors)
        print(f"颜色范围: R({colors[:, 0].min():.3f}, {colors[:, 0].max():.3f}) "
              f"G({colors[:, 1].min():.3f}, {colors[:, 1].max():.3f}) "
              f"B({colors[:, 2].min():.3f}, {colors[:, 2].max():.3f})")
    else:
        # 如果没有颜色信息，创建默认颜色（白色）
        colors = np.ones((len(points), 3))
        print("警告：PLY文件中没有颜色信息，使用白色作为默认颜色")
    
    # 确保颜色值在0-1范围内
    if colors.max() > 1.0:
        colors = colors / 255.0
        print("颜色值已从0-255范围转换为0-1范围")
    
    # 转换为torch张量
    points_tensor = torch.from_numpy(points.astype(np.float32))
    colors_tensor = torch.from_numpy(colors.astype(np.float32))
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为.pt文件
    points_path = os.path.join(output_dir, "points.pt")
    colors_path = os.path.join(output_dir, "colors.pt")
    
    torch.save(points_tensor, points_path)
    torch.save(colors_tensor, colors_path)
    
    print(f"转换完成！")
    print(f"Points 保存到: {points_path}")
    print(f"Colors 保存到: {colors_path}")
    
    return points_tensor, colors_tensor

# 使用示例
if __name__ == "__main__":
    # 修改这些路径为你的实际路径
    ply_file = "/home/hkcrc/DCIM/rs1105_2/cloud.ply"  # 你的PLY文件路径
    output_directory = "output/"  # 输出目录
    
    # 转换文件
    points_tensor, colors_tensor = convert_ply_to_pt(ply_file, output_directory)
    
    # 验证转换结果（可选）
    print("\n=== 验证转换结果 ===")
    print(f"Points tensor shape: {points_tensor.shape}")
    print(f"Colors tensor shape: {colors_tensor.shape}")
    
    # 使用你现有的可视化代码验证
    pcd_verify = o3d.geometry.PointCloud()
    pcd_verify.points = o3d.utility.Vector3dVector(points_tensor.numpy())
    pcd_verify.colors = o3d.utility.Vector3dVector(colors_tensor.numpy())
    
    print("显示转换后的点云...")
    o3d.visualization.draw_geometries([pcd_verify])