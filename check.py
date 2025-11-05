import torch
import numpy as np
import os
import sys
sys.path.append('/home/hkcrc/diffusion_edfs/diffusion_edf')
from torch_cluster import radius

def simple_voxel_downsample(points, voxel_size):
    """简单的体素下采样实现"""
    if voxel_size <= 0:
        return points
    
    # 量化坐标到体素网格
    voxel_coords = np.floor(points[:, :3] / voxel_size).astype(np.int32)
    
    # 找到唯一的体素
    unique_voxels, inverse_indices = np.unique(
        voxel_coords, axis=0, return_inverse=True
    )
    
    # 对每个体素计算平均位置
    downsampled_points = []
    for i in range(len(unique_voxels)):
        mask = (inverse_indices == i)
        avg_point = points[mask].mean(axis=0)
        downsampled_points.append(avg_point)
    
    return np.array(downsampled_points)

def check_first_demo():
    """只检查第一个demo"""
   # demo_path = "demo/rebar_grasping/data/demo_0/step_0"
    demo_path = "demo/panda_mug_on_hanger/data/demo_0/step_0"    
    # 加载原始数据
    scene_pcd = torch.load(os.path.join(demo_path, "scene_pcd/points.pt"))
    grasp_pcd = torch.load(os.path.join(demo_path, "grasp_pcd/points.pt"))

    print("="*60)
    print("检查 demo_0/step_0 的点云数据")
    print("="*60)
    
    print(f"\n原始数据:")
    print(f"场景点云: {scene_pcd.shape}")
    print(f"抓取点云: {grasp_pcd.shape}")
    
    # 打印点云范围
    print(f"\n场景点云范围:")
    print(f"  X: [{scene_pcd[:, 0].min():.3f}, {scene_pcd[:, 0].max():.3f}]")
    print(f"  Y: [{scene_pcd[:, 1].min():.3f}, {scene_pcd[:, 1].max():.3f}]")
    print(f"  Z: [{scene_pcd[:, 2].min():.3f}, {scene_pcd[:, 2].max():.3f}]")
    
    print(f"\n抓取点云范围:")
    print(f"  X: [{grasp_pcd[:, 0].min():.3f}, {grasp_pcd[:, 0].max():.3f}]")
    print(f"  Y: [{grasp_pcd[:, 1].min():.3f}, {grasp_pcd[:, 1].max():.3f}]")
    print(f"  Z: [{grasp_pcd[:, 2].min():.3f}, {grasp_pcd[:, 2].max():.3f}]")
    
    # 测试不同的配置
    voxel_sizes = [0.005, 0.01]
    contact_radii = [0.02, 0.03, 0.05, 0.1]
    
    print("\n" + "="*60)
    print("测试不同参数组合:")
    print("="*60)
    
    for voxel_size in voxel_sizes:
        # 下采样
        scene_down = simple_voxel_downsample(scene_pcd.numpy(), voxel_size)
        grasp_down = simple_voxel_downsample(grasp_pcd.numpy(), voxel_size)
        
        scene_down = torch.tensor(scene_down[:, :3], dtype=torch.float32)
        grasp_down = torch.tensor(grasp_down[:, :3], dtype=torch.float32)
        
        print(f"\n体素大小 = {voxel_size}m:")
        print(f"  下采样后: 场景 {len(scene_down)} 点, 抓取 {len(grasp_down)} 点")
        
        for contact_radius in contact_radii:
            # 计算接触点
            edge_dst, edge_src = radius(x=scene_down, y=grasp_down, r=contact_radius)
            
            if len(edge_dst) > 0:
                print(f"  radius={contact_radius:.2f}m: ✓ 找到 {len(edge_dst)} 条边")
            else:
                print(f"  radius={contact_radius:.2f}m: ✗ 没有接触点")
                
                # 计算最小距离
                from scipy.spatial.distance import cdist
                dist = cdist(grasp_down.numpy(), scene_down.numpy())
                min_dist = dist.min()
                print(f"    最小距离: {min_dist:.4f}m")

# 运行检查
check_first_demo()