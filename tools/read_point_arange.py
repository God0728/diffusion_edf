import torch
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# 读取 .pt 文件
path = "/home/hkcrc/diffusion_edfs/diffusion_edf/tools/"
points = torch.load(path+"points.pt").cpu().numpy()
colors = torch.load(path+"colors.pt").cpu().numpy()

# 基本信息
print("=== 基本信息 ===")
print("Points:", points)
print("Points shape:", points.shape if hasattr(points, 'shape') else len(points))
print("Points type:", type(points))
print("Color:", colors)
print("Color shape:", colors.shape if hasattr(colors, 'shape') else len(colors))
print("Color type:", type(colors))

# 详细的坐标分布分析
print("\n=== 坐标分布分析 ===")
print(f"点云总数: {len(points)}")

# X, Y, Z 轴的统计信息
x_coords = points[:, 0]
y_coords = points[:, 1]
z_coords = points[:, 2]

print(f"\nX轴坐标分布:")
print(f"  范围: {x_coords.min():.6f} 到 {x_coords.max():.6f}")
print(f"  均值: {x_coords.mean():.6f}")
print(f"  标准差: {x_coords.std():.6f}")
print(f"  中位数: {np.median(x_coords):.6f}")

print(f"\nY轴坐标分布:")
print(f"  范围: {y_coords.min():.6f} 到 {y_coords.max():.6f}")
print(f"  均值: {y_coords.mean():.6f}")
print(f"  标准差: {y_coords.std():.6f}")
print(f"  中位数: {np.median(y_coords):.6f}")

print(f"\nZ轴坐标分布:")
print(f"  范围: {z_coords.min():.6f} 到 {z_coords.max():.6f}")
print(f"  均值: {z_coords.mean():.6f}")
print(f"  标准差: {z_coords.std():.6f}")
print(f"  中位数: {np.median(z_coords):.6f}")

# 计算点云的边界框
print(f"\n=== 边界框信息 ===")
min_bound = points.min(axis=0)
max_bound = points.max(axis=0)
size = max_bound - min_bound
center = (min_bound + max_bound) / 2

print(f"最小边界: [{min_bound[0]:.6f}, {min_bound[1]:.6f}, {min_bound[2]:.6f}]")
print(f"最大边界: [{max_bound[0]:.6f}, {max_bound[1]:.6f}, {max_bound[2]:.6f}]")
print(f"尺寸大小: [{size[0]:.6f}, {size[1]:.6f}, {size[2]:.6f}]")
print(f"中心点: [{center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}]")

# 计算点云密度
volume = size[0] * size[1] * size[2]
if volume > 0:
    density = len(points) / volume
    print(f"点云密度: {density:.2f} 点/立方单位")

# 距离原点的分布
distances_from_origin = np.linalg.norm(points, axis=1)
print(f"\n=== 距离分布 ===")
print(f"距离原点范围: {distances_from_origin.min():.6f} 到 {distances_from_origin.max():.6f}")
print(f"平均距离: {distances_from_origin.mean():.6f}")

# 绘制坐标分布直方图
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(x_coords, bins=50, alpha=0.7, color='red')
plt.title('X坐标分布')
plt.xlabel('X值')
plt.ylabel('频次')

plt.subplot(1, 3, 2)
plt.hist(y_coords, bins=50, alpha=0.7, color='green')
plt.title('Y坐标分布')
plt.xlabel('Y值')
plt.ylabel('频次')

plt.subplot(1, 3, 3)
plt.hist(z_coords, bins=50, alpha=0.7, color='blue')
plt.title('Z坐标分布')
plt.xlabel('Z值')
plt.ylabel('频次')

plt.tight_layout()
plt.show()

# 3D散点图显示坐标分布（如果点不太多）
if len(points) <= 120000:  # 避免太多点导致卡顿
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 随机采样一些点来显示
    if len(points) > 5000:
        indices = np.random.choice(len(points), 5000, replace=False)
        sample_points = points[indices]
        sample_colors = colors[indices]
    else:
        sample_points = points
        sample_colors = colors
    
    scatter = ax.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2], 
                        c=sample_colors, s=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('点云3D分布')
    plt.show()

# 转为 open3d 点云并可视化
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# 添加坐标轴
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

print("\n=== Open3D可视化 ===")
print("红色轴: X轴")
print("绿色轴: Y轴") 
print("蓝色轴: Z轴")

# 可视化
o3d.visualization.draw_geometries([pcd, coordinate_frame], 
                                window_name="点云可视化 - 带坐标轴",
                                width=800, height=600)

# 检查坐标是否在合理范围内
print(f"\n=== 坐标范围检查 ===")
if np.all(np.abs(points) < 10):
    print("✓ 坐标在合理范围内 (< 10)")
elif np.all(np.abs(points) < 100):
    print("⚠ 坐标较大但可接受 (< 100)")
else:
    print("❌ 坐标可能过大，需要归一化")

# 检查是否有异常值
q1 = np.percentile(points, 25, axis=0)
q3 = np.percentile(points, 75, axis=0)
iqr = q3 - q1
outlier_threshold = 1.5

outliers_low = points < (q1 - outlier_threshold * iqr)
outliers_high = points > (q3 + outlier_threshold * iqr)
outlier_mask = np.any(outliers_low | outliers_high, axis=1)
num_outliers = np.sum(outlier_mask)

print(f"异常值检测: {num_outliers} 个点可能是异常值 ({num_outliers/len(points)*100:.2f}%)")