# import torch
# points = torch.load("demo/panda_bottle_on_shelf/data/demo_0/step_0/target_poses/poses.pt")
# print(points.shape)
import torch

# 加载 poses.pt 文件
path = "/home/hkcrc/diffusion_edfs/diffusion_edf/demo/panda_bottle_on_shelf/data/demo_1/step_0/scene_pcd/points.pt"
poses = torch.load(path)

print("✅ 文件加载成功")
print("数据类型:", type(poses))
print("张量形状:", poses.shape)

# 打印全部数据内容
print("\n具体数值:")
print(poses)

# 如果想单独分开显示位置和姿态：
pose = poses[0]
position = pose[:3]
quaternion = pose[3:]
print("\n位置 (x, y, z):", position.tolist())
print("姿态 (qx, qy, qz, qw):", quaternion.tolist())
