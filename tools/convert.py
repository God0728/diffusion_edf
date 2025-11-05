import numpy as np
import open3d as o3d
import os

# 路径设置
base_dir = "/home/hkcrc/diffusion_edfs/diffusion_edf/demo/panda_bottle_on_shelf/data/demo_0/step_0/grasp_pcd"
points_path = os.path.join(base_dir, "points.pt")
colors_path = os.path.join(base_dir, "colors.pt")

# 读取纯文本 (ASCII)
points = np.loadtxt(points_path)
colors = np.loadtxt(colors_path)

print(f"✅ 已读取 {points.shape[0]} 个点")
print("前5个点坐标:\n", points[:5])

# 检查颜色范围，如果是 0–255 则归一化
if colors.max() > 1.0:
    colors = colors / 255.0

# 转为 Open3D 点云
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# 保存为 .ply
output_path = os.path.join(base_dir, "converted_ascii_cloud.ply")
o3d.io.write_point_cloud(output_path, pcd)
print(f"✅ 已保存为 {output_path}")