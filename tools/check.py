import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def visualize_pointclouds():
    """可视化两个数据集的点云"""
    
    # 加载数据
    rebar_scene = torch.load("/home/hkcrc/diffusion_edfs/diffusion_edf/demo/rebar_grasping/data/demo_0/step_0/scene_pcd/points.pt")
    rebar_grasp = torch.load("/home/hkcrc/diffusion_edfs/diffusion_edf/demo/rebar_grasping/data/demo_0/step_0/grasp_pcd/points.pt")
    
    bowl_scene = torch.load("/home/hkcrc/diffusion_edfs/diffusion_edf/demo/panda_bowl_on_dish/data/demo_0/step_0/scene_pcd/points.pt")
    bowl_grasp = torch.load("/home/hkcrc/diffusion_edfs/diffusion_edf/demo/panda_bowl_on_dish/data/demo_0/step_0/grasp_pcd/points.pt")
    
    # 下采样
    voxel_size = 0.01
    rebar_scene_down = simple_voxel_downsample(rebar_scene.numpy(), voxel_size)
    rebar_grasp_down = simple_voxel_downsample(rebar_grasp.numpy(), voxel_size)
    bowl_scene_down = simple_voxel_downsample(bowl_scene.numpy(), voxel_size)
    bowl_grasp_down = simple_voxel_downsample(bowl_grasp.numpy(), voxel_size)
    
    # 创建图形
    fig = plt.figure(figsize=(20, 10))
    
    # Rebar原始数据
    ax1 = fig.add_subplot(241, projection='3d')
    ax1.scatter(rebar_scene[::100, 0], rebar_scene[::100, 1], rebar_scene[::100, 2], 
                c='blue', s=1, alpha=0.3, label='Scene')
    ax1.scatter(rebar_grasp[::50, 0], rebar_grasp[::50, 1], rebar_grasp[::50, 2], 
                c='red', s=2, alpha=0.6, label='Grasp')
    ax1.set_title(f'Rebar Original\nScene:{rebar_scene.shape[0]}, Grasp:{rebar_grasp.shape[0]}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    
    # Rebar下采样后
    ax2 = fig.add_subplot(242, projection='3d')
    ax2.scatter(rebar_scene_down[::10, 0], rebar_scene_down[::10, 1], rebar_scene_down[::10, 2], 
                c='blue', s=1, alpha=0.3, label='Scene')
    ax2.scatter(rebar_grasp_down[:, 0], rebar_grasp_down[:, 1], rebar_grasp_down[:, 2], 
                c='red', s=3, alpha=0.8, label='Grasp')
    ax2.set_title(f'Rebar Downsampled (voxel={voxel_size})\nScene:{len(rebar_scene_down)}, Grasp:{len(rebar_grasp_down)}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    
    # Bowl原始数据
    ax3 = fig.add_subplot(243, projection='3d')
    ax3.scatter(bowl_scene[::100, 0], bowl_scene[::100, 1], bowl_scene[::100, 2], 
                c='blue', s=1, alpha=0.3, label='Scene')
    ax3.scatter(bowl_grasp[::50, 0], bowl_grasp[::50, 1], bowl_grasp[::50, 2], 
                c='red', s=2, alpha=0.6, label='Grasp')
    ax3.set_title(f'Bowl Original\nScene:{bowl_scene.shape[0]}, Grasp:{bowl_grasp.shape[0]}')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()
    
    # Bowl下采样后
    ax4 = fig.add_subplot(244, projection='3d')
    ax4.scatter(bowl_scene_down[::10, 0], bowl_scene_down[::10, 1], bowl_scene_down[::10, 2], 
                c='blue', s=1, alpha=0.3, label='Scene')
    ax4.scatter(bowl_grasp_down[:, 0], bowl_grasp_down[:, 1], bowl_grasp_down[:, 2], 
                c='red', s=3, alpha=0.8, label='Grasp')
    ax4.set_title(f'Bowl Downsampled (voxel={voxel_size})\nScene:{len(bowl_scene_down)}, Grasp:{len(bowl_grasp_down)}')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.legend()
    
    # 俯视图 - Rebar
    ax5 = fig.add_subplot(245)
    ax5.scatter(rebar_scene_down[::10, 0], rebar_scene_down[::10, 1], 
                c='blue', s=1, alpha=0.3, label='Scene')
    ax5.scatter(rebar_grasp_down[:, 0], rebar_grasp_down[:, 1], 
                c='red', s=5, alpha=0.8, label='Grasp')
    ax5.set_title('Rebar Top View (X-Y)')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.axis('equal')
    
    # 侧视图 - Rebar
    ax6 = fig.add_subplot(246)
    ax6.scatter(rebar_scene_down[::10, 0], rebar_scene_down[::10, 2], 
                c='blue', s=1, alpha=0.3, label='Scene')
    ax6.scatter(rebar_grasp_down[:, 0], rebar_grasp_down[:, 2], 
                c='red', s=5, alpha=0.8, label='Grasp')
    ax6.set_title('Rebar Side View (X-Z)')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Z')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 俯视图 - Bowl
    ax7 = fig.add_subplot(247)
    ax7.scatter(bowl_scene_down[::10, 0], bowl_scene_down[::10, 1], 
                c='blue', s=1, alpha=0.3, label='Scene')
    ax7.scatter(bowl_grasp_down[:, 0], bowl_grasp_down[:, 1], 
                c='red', s=5, alpha=0.8, label='Grasp')
    ax7.set_title('Bowl Top View (X-Y)')
    ax7.set_xlabel('X')
    ax7.set_ylabel('Y')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.axis('equal')
    
    # 侧视图 - Bowl
    ax8 = fig.add_subplot(248)
    ax8.scatter(bowl_scene_down[::10, 0], bowl_scene_down[::10, 2], 
                c='blue', s=1, alpha=0.3, label='Scene')
    ax8.scatter(bowl_grasp_down[:, 0], bowl_grasp_down[:, 2], 
                c='red', s=5, alpha=0.8, label='Grasp')
    ax8.set_title('Bowl Side View (X-Z)')
    ax8.set_xlabel('X')
    ax8.set_ylabel('Z')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pointcloud_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 打印统计信息
    print("="*60)
    print("点云统计信息")
    print("="*60)
    
    # 计算最小距离
    from scipy.spatial.distance import cdist
    
    print("\nRebar数据集:")
    dist_rebar = cdist(rebar_grasp_down[:, :3], rebar_scene_down[:, :3])
    min_dist_rebar = dist_rebar.min()
    print(f"  下采样后最小距离: {min_dist_rebar:.4f}m")
    print(f"  Scene范围: X[{rebar_scene_down[:, 0].min():.3f}, {rebar_scene_down[:, 0].max():.3f}]")
    print(f"  Grasp范围: X[{rebar_grasp_down[:, 0].min():.3f}, {rebar_grasp_down[:, 0].max():.3f}]")
    
    print("\nBowl数据集:")
    dist_bowl = cdist(bowl_grasp_down[:, :3], bowl_scene_down[:, :3])
    min_dist_bowl = dist_bowl.min()
    print(f"  下采样后最小距离: {min_dist_bowl:.4f}m")
    print(f"  Scene范围: X[{bowl_scene_down[:, 0].min():.3f}, {bowl_scene_down[:, 0].max():.3f}]")
    print(f"  Grasp范围: X[{bowl_grasp_down[:, 0].min():.3f}, {bowl_grasp_down[:, 0].max():.3f}]")
    
    # 检查重叠区域
    print("\n空间重叠分析:")
    rebar_x_overlap = (rebar_grasp_down[:, 0].min() < rebar_scene_down[:, 0].max()) and \
                      (rebar_grasp_down[:, 0].max() > rebar_scene_down[:, 0].min())
    rebar_y_overlap = (rebar_grasp_down[:, 1].min() < rebar_scene_down[:, 1].max()) and \
                      (rebar_grasp_down[:, 1].max() > rebar_scene_down[:, 1].min())
    rebar_z_overlap = (rebar_grasp_down[:, 2].min() < rebar_scene_down[:, 2].max()) and \
                      (rebar_grasp_down[:, 2].max() > rebar_scene_down[:, 2].min())
    
    bowl_x_overlap = (bowl_grasp_down[:, 0].min() < bowl_scene_down[:, 0].max()) and \
                     (bowl_grasp_down[:, 0].max() > bowl_scene_down[:, 0].min())
    bowl_y_overlap = (bowl_grasp_down[:, 1].min() < bowl_scene_down[:, 1].max()) and \
                     (bowl_grasp_down[:, 1].max() > bowl_scene_down[:, 1].min())
    bowl_z_overlap = (bowl_grasp_down[:, 2].min() < bowl_scene_down[:, 2].max()) and \
                     (bowl_grasp_down[:, 2].max() > bowl_scene_down[:, 2].min())
    
    print(f"Rebar - X重叠:{rebar_x_overlap}, Y重叠:{rebar_y_overlap}, Z重叠:{rebar_z_overlap}")
    print(f"Bowl  - X重叠:{bowl_x_overlap}, Y重叠:{bowl_y_overlap}, Z重叠:{bowl_z_overlap}")

if __name__ == "__main__":
    visualize_pointclouds()