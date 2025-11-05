import torch
import open3d as o3d

# 读取 .pt 文件
path = "/home/hkcrc/diffusion_edfs/diffusion_edf/dataset/1103_baselink_pt/grasp0/raw/"
#path = "/home/hkcrc/diffusion_edfs/diffusion_edf/demo/panda_bottle_on_shelf/data/demo_0/step_1/scene_pcd/points.pt"
points = torch.load(path+"points.pt").cpu().numpy()
colors = torch.load(path+"colors.pt").cpu().numpy()
#colors = torch.load("/home/hkcrc/diffusion_edfs/diffusion_edf/demo/panda_bowl_on_dish_test/data/demo_0/step_1/scene_pcd/colors.pt").cpu().numpy()

# 查看变量的值和形状
print("Points:", points)
print("Points shape:", points.shape if hasattr(points, 'shape') else len(points))
print("Points type:", type(points))

print("Color:", colors)
print("Color shape:", colors.shape if hasattr(colors, 'shape') else len(colors))
print("Color type:", type(colors))
# 转为 open3d 点云
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
# 可视化
o3d.visualization.draw_geometries([pcd])


# import numpy as np

# points1 = torch.load('points_1.pt').cpu().numpy()
# points2 = torch.load('points_2.pt').cpu().numpy()

# print("Scene 1 range:", np.min(points1, axis=0), "→", np.max(points1, axis=0))
# print("Scene 2 range:", np.min(points2, axis=0), "→", np.max(points2, axis=0))
