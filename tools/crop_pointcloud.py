import numpy as np
import torch
import open3d as o3d
import os

def load_pointcloud_pt(input_dir):
    """
    从目录加载points.pt和colors.pt文件
    
    Args:
        input_dir: 包含points.pt和colors.pt的目录
    
    Returns:
        points: Nx3 numpy数组
        colors: Nx3 numpy数组
    """
    points_file = os.path.join(input_dir, "points.pt")
    colors_file = os.path.join(input_dir, "colors.pt")
    
    if not os.path.exists(points_file):
        raise FileNotFoundError(f"未找到文件: {points_file}")
    
    # 加载点云坐标
    points = torch.load(points_file).cpu().numpy()
    print(f"✓ 加载points.pt: {points.shape}")
    
    # 加载颜色
    if os.path.exists(colors_file):
        colors = torch.load(colors_file).cpu().numpy()
        print(f"✓ 加载colors.pt: {colors.shape}")
    else:
        print("⚠ 未找到colors.pt，使用默认白色")
        colors = np.ones((len(points), 3))
    
    return points, colors


def remove_outliers(points, colors):
    """
    去除点云中的离散点/噪声点
    支持统计滤波和半径滤波两种方法
    
    Args:
        points: Nx3 点云坐标
        colors: Nx3 点云颜色
    
    Returns:
        filtered_points, filtered_colors: 过滤后的点云和颜色
    """
    print("\n=== 去除离散点/噪声点 ===")
    print(f"原始点数: {len(points)}")
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print("\n选择过滤方法:")
    print("1. 统计滤波 (Statistical Outlier Removal) - 推荐")
    print("   - 基于邻域点的距离统计")
    print("   - 适合去除稀疏的噪声点")
    print("2. 半径滤波 (Radius Outlier Removal)")
    print("   - 基于半径内的邻居数量")
    print("   - 适合去除孤立的点簇")
    
    method = input("请选择方法 (1/2, 默认1): ").strip()
    if not method:
        method = '1'
    
    try:
        if method == '1':
            # 统计滤波
            print("\n统计滤波参数设置:")
            print("- nb_neighbors: 考虑的邻居点数量 (推荐20-50)")
            print("- std_ratio: 标准差倍数阈值 (推荐1.0-2.0，越小越严格)")
            
            nb_neighbors_input = input("邻居点数量 (默认20): ").strip()
            nb_neighbors = int(nb_neighbors_input) if nb_neighbors_input else 20
            
            std_ratio_input = input("标准差倍数 (默认2.0): ").strip()
            std_ratio = float(std_ratio_input) if std_ratio_input else 2.0
            
            print(f"\n执行统计滤波 (邻居数: {nb_neighbors}, 标准差: {std_ratio})...")
            
            # 应用统计滤波
            filtered_pcd, inlier_indices = pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )
            
        else:
            # 半径滤波
            print("\n半径滤波参数设置:")
            print("- radius: 搜索半径 (米)")
            print("- min_neighbors: 半径内最少邻居数")
            
            radius_input = input("搜索半径 (米, 默认0.05): ").strip()
            radius = float(radius_input) if radius_input else 0.05
            
            min_neighbors_input = input("最少邻居数 (默认10): ").strip()
            min_neighbors = int(min_neighbors_input) if min_neighbors_input else 10
            
            print(f"\n执行半径滤波 (半径: {radius}m, 最少邻居: {min_neighbors})...")
            
            # 应用半径滤波
            filtered_pcd, inlier_indices = pcd.remove_radius_outlier(
                nb_points=min_neighbors,
                radius=radius
            )
        
        # 提取过滤后的点和颜色
        filtered_points = np.asarray(filtered_pcd.points)
        filtered_colors = np.asarray(filtered_pcd.colors)
        
        removed_count = len(points) - len(filtered_points)
        removed_ratio = removed_count / len(points) * 100
        
        print(f"\n✓ 过滤完成:")
        print(f"  原始点数: {len(points)}")
        print(f"  过滤后点数: {len(filtered_points)}")
        print(f"  移除点数: {removed_count}")
        print(f"  移除比例: {removed_ratio:.1f}%")
        
        if len(filtered_points) == 0:
            print("⚠ 警告: 过滤后点云为空！请调整参数")
            return points, colors
        
        if removed_ratio < 0.1:
            print("⚠ 提示: 移除的点很少，可能需要调整参数")
        elif removed_ratio > 50:
            print("⚠ 警告: 移除了超过50%的点，可能参数过于严格")
        
        return filtered_points, filtered_colors
        
    except Exception as e:
        print(f"❌ 过滤失败: {e}")
        import traceback
        traceback.print_exc()
        return points, colors


def visualize_pointcloud(points, colors, title="点云可视化", coordinate_size=None, show_grid=False):
    """
    可视化点云
    
    Args:
        points: Nx3 点云坐标
        colors: Nx3 点云颜色
        title: 窗口标题
        coordinate_size: 坐标轴大小，如果为None则自动计算
        show_grid: 是否显示网格
    """
    print(f"\n{title}...")
    
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 自动计算坐标轴大小
    if coordinate_size is None:
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        z_range = points[:, 2].max() - points[:, 2].min()
        coordinate_size = max(x_range, y_range, z_range) * 0.15  # 15%的比例
        coordinate_size = max(coordinate_size, 0.05)  # 最小5cm
    
    # 添加坐标轴
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coordinate_size)
    
    # 创建可视化对象列表
    vis_objects = [pcd, coordinate_frame]
    
    # 添加网格 (如果需要)
    if show_grid:
        print("添加以原点为中心的3D网格...")
        
        # 计算点云边界
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        
        # 计算合适的网格步长
        x_range, y_range, z_range = x_max - x_min, y_max - y_min, z_max - z_min
        max_range = max(x_range, y_range, z_range)
        
        # 根据范围选择网格步长
        grid_step = 0.1  # 10cm网格
        
        print(f"- 网格步长: {grid_step:.2f}m ({grid_step*100:.0f}cm)")
        
        # 确定网格范围 (以原点为中心，覆盖点云范围)
        grid_x_min = np.floor(x_min / grid_step) * grid_step
        grid_x_max = np.ceil(x_max / grid_step) * grid_step
        grid_y_min = np.floor(y_min / grid_step) * grid_step
        grid_y_max = np.ceil(y_max / grid_step) * grid_step
        grid_z_min = np.floor(z_min / grid_step) * grid_step
        grid_z_max = np.ceil(z_max / grid_step) * grid_step
        
        # 创建网格线
        grid_lines = []
        grid_colors = []
        
        # XY平面网格 (在Z=0处，或者最接近0的Z平面)
        z_plane = 0.0 if grid_z_min <= 0 <= grid_z_max else grid_z_min
        
        # X方向线 (平行于X轴)
        y = grid_y_min
        while y <= grid_y_max:
            grid_lines.extend([[grid_x_min, y, z_plane], [grid_x_max, y, z_plane]])
            # 主轴用深色，其他用浅色
            if abs(y) < grid_step/2:  # Y=0轴
                grid_colors.extend([[0.3, 0.7, 0.3], [0.3, 0.7, 0.3]])  # 深绿
            else:
                grid_colors.extend([[0.7, 0.7, 0.7], [0.7, 0.7, 0.7]])  # 浅灰
            y += grid_step
        
        # Y方向线 (平行于Y轴)
        x = grid_x_min
        while x <= grid_x_max:
            grid_lines.extend([[x, grid_y_min, z_plane], [x, grid_y_max, z_plane]])
            # 主轴用深色，其他用浅色
            if abs(x) < grid_step/2:  # X=0轴
                grid_colors.extend([[0.7, 0.3, 0.3], [0.7, 0.3, 0.3]])  # 深红
            else:
                grid_colors.extend([[0.7, 0.7, 0.7], [0.7, 0.7, 0.7]])  # 浅灰
            x += grid_step
        
        # XZ平面网格 (在Y=0处，或者最接近0的Y平面)
        y_plane = 0.0 if grid_y_min <= 0 <= grid_y_max else grid_y_min
        
        # X方向线
        z = grid_z_min
        while z <= grid_z_max:
            if abs(z - z_plane) > grid_step/2:  # 避免与XY平面重复
                grid_lines.extend([[grid_x_min, y_plane, z], [grid_x_max, y_plane, z]])
                if abs(z) < grid_step/2:  # Z=0轴
                    grid_colors.extend([[0.3, 0.3, 0.7], [0.3, 0.3, 0.7]])  # 深蓝
                else:
                    grid_colors.extend([[0.6, 0.8, 0.6], [0.6, 0.8, 0.6]])  # 浅绿
            z += grid_step
        
        # Z方向线
        x = grid_x_min
        while x <= grid_x_max:
            if abs(x) > grid_step/2 or abs(y_plane) > grid_step/2:  # 避免与主轴重复
                grid_lines.extend([[x, y_plane, grid_z_min], [x, y_plane, grid_z_max]])
                grid_colors.extend([[0.6, 0.8, 0.6], [0.6, 0.8, 0.6]])  # 浅绿
            x += grid_step
        
        # YZ平面网格 (在X=0处，或者最接近0的X平面)
        x_plane = 0.0 if grid_x_min <= 0 <= grid_x_max else grid_x_min
        
        # Y方向线
        z = grid_z_min
        while z <= grid_z_max:
            if abs(z - z_plane) > grid_step/2:  # 避免重复
                grid_lines.extend([[x_plane, grid_y_min, z], [x_plane, grid_y_max, z]])
                grid_colors.extend([[0.8, 0.6, 0.6], [0.8, 0.6, 0.6]])  # 浅红
            z += grid_step
        
        # Z方向线
        y = grid_y_min
        while y <= grid_y_max:
            if abs(y) > grid_step/2 or abs(x_plane) > grid_step/2:  # 避免重复
                grid_lines.extend([[x_plane, y, grid_z_min], [x_plane, y, grid_z_max]])
                grid_colors.extend([[0.8, 0.6, 0.6], [0.8, 0.6, 0.6]])  # 浅红
            y += grid_step
        
        # 创建线集对象
        if len(grid_lines) >= 2:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(grid_lines)
            line_set.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(0, len(grid_lines)-1, 2)])
            line_set.colors = o3d.utility.Vector3dVector(grid_colors[::2])  # 每条线取一个颜色
            vis_objects.append(line_set)
        
        # 添加数值标注 (在关键位置)
        text_points = []
        text_values = []
        
        # X轴标注
        x = grid_x_min
        while x <= grid_x_max:
            if abs(x) >= grid_step and abs(x % (grid_step * 2)) < grid_step/2:  # 每两个网格标注一次
                text_points.append([x, grid_y_max + grid_step*0.1, z_plane])
                text_values.append(f"X{x:.2f}")
            x += grid_step
        
        # Y轴标注
        y = grid_y_min
        while y <= grid_y_max:
            if abs(y) >= grid_step and abs(y % (grid_step * 2)) < grid_step/2:
                text_points.append([grid_x_max + grid_step*0.1, y, z_plane])
                text_values.append(f"Y{y:.2f}")
            y += grid_step
        
        # Z轴标注
        z = grid_z_min
        while z <= grid_z_max:
            if abs(z) >= grid_step and abs(z % (grid_step * 2)) < grid_step/2:
                text_points.append([grid_x_max + grid_step*0.1, y_plane, z])
                text_values.append(f"Z{z:.2f}")
            z += grid_step
        
        print(f"- 网格范围: X[{grid_x_min:.2f}, {grid_x_max:.2f}] Y[{grid_y_min:.2f}, {grid_y_max:.2f}] Z[{grid_z_min:.2f}, {grid_z_max:.2f}]")
        print(f"- 添加了 {len(text_values)} 个数值标注")
    
    print("- 红色轴: X轴")
    print("- 绿色轴: Y轴")  
    print("- 蓝色轴: Z轴")
    print(f"- 坐标轴大小: {coordinate_size:.3f}m")
    if show_grid:
        print("- 深色线条: 主轴 (X=0, Y=0, Z=0)")
        print("- 浅色线条: 网格线")
        print("- 数值标注显示坐标值")
    print("关闭窗口继续...")
    
    # 可视化
    o3d.visualization.draw_geometries(vis_objects, 
                                    window_name=title,
                                    width=1200, height=900)

def crop_with_bounding_box(points, colors):
    """
    基于边界框的点云切割 (观察模式)
    先可视化点云让用户观察，然后通过命令行输入边界框参数
    
    Args:
        points: Nx3 点云坐标
        colors: Nx3 点云颜色
    
    Returns:
        cropped_points, cropped_colors: 切割后的点云和颜色
    """
    print("\n=== 边界框切割模式 ===")
    print("此模式结合可视化观察和精确数值输入")
    print("1. 观察点云的空间分布和坐标轴")
    print("2. 3D网格帮助判断具体尺度")
    print("3. 根据观察结果输入精确的边界框参数")
    
    # 显示当前点云范围
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    
    print(f"\n当前点云范围:")
    print(f"  X: [{x_min:.3f}, {x_max:.3f}] (红色轴)")
    print(f"  Y: [{y_min:.3f}, {y_max:.3f}] (绿色轴)")
    print(f"  Z: [{z_min:.3f}, {z_max:.3f}] (蓝色轴)")
    print(f"  总点数: {len(points)}")
    
    # 询问是否显示网格
    show_grid = input("\n是否显示3D网格帮助观察尺度? (y/n, 默认y): ").strip().lower()
    show_grid = show_grid != 'n'
    
    # 可选：再次显示点云供用户观察
    show_again = input("是否重新显示点云进行观察? (y/n, 默认y): ").strip().lower()
    if show_again != 'n':
        # 计算合适的坐标轴大小
        point_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        coordinate_size = point_range * 0.1  # 坐标轴大小为点云范围的10%
        
        print("\n显示点云及坐标轴:")
        print("- 红色轴: X轴")
        print("- 绿色轴: Y轴")
        print("- 蓝色轴: Z轴")
        if show_grid:
            print("- 网格帮助判断距离和尺度")
        
        visualize_pointcloud(points, colors, "边界框切割 - 观察点云", coordinate_size, show_grid)
    
    # 用户输入切割范围
    try:
        print("\n请输入切割边界框 (直接按回车保持当前值):")
        
        x_min_input = input(f"X最小值 (红色轴, 当前: {x_min:.3f}): ").strip()
        x_min_crop = float(x_min_input) if x_min_input else x_min
        
        x_max_input = input(f"X最大值 (红色轴, 当前: {x_max:.3f}): ").strip()
        x_max_crop = float(x_max_input) if x_max_input else x_max
        
        y_min_input = input(f"Y最小值 (绿色轴, 当前: {y_min:.3f}): ").strip()
        y_min_crop = float(y_min_input) if y_min_input else y_min
        
        y_max_input = input(f"Y最大值 (绿色轴, 当前: {y_max:.3f}): ").strip()
        y_max_crop = float(y_max_input) if y_max_input else y_max
        
        z_min_input = input(f"Z最小值 (蓝色轴, 当前: {z_min:.3f}): ").strip()
        z_min_crop = float(z_min_input) if z_min_input else z_min
        
        z_max_input = input(f"Z最大值 (蓝色轴, 当前: {z_max:.3f}): ").strip()
        z_max_crop = float(z_max_input) if z_max_input else z_max
        
        # 验证范围
        if x_min_crop >= x_max_crop or y_min_crop >= y_max_crop or z_min_crop >= z_max_crop:
            print("❌ 错误: 最小值必须小于最大值")
            return points, colors
        
    except ValueError:
        print("❌ 输入格式错误，使用原始范围")
        return points, colors
    
    # 执行切割
    print(f"\n执行边界框切割...")
    print(f"切割范围:")
    print(f"  X: [{x_min_crop:.3f}, {x_max_crop:.3f}]")
    print(f"  Y: [{y_min_crop:.3f}, {y_max_crop:.3f}]")
    print(f"  Z: [{z_min_crop:.3f}, {z_max_crop:.3f}]")
    
    mask = ((points[:, 0] >= x_min_crop) & (points[:, 0] <= x_max_crop) &
            (points[:, 1] >= y_min_crop) & (points[:, 1] <= y_max_crop) &
            (points[:, 2] >= z_min_crop) & (points[:, 2] <= z_max_crop))
    
    cropped_points = points[mask]
    cropped_colors = colors[mask]
    
    print(f"\n✓ 边界框切割完成:")
    print(f"  原始点数: {len(points)}")
    print(f"  切割后点数: {len(cropped_points)}")
    print(f"  保留比例: {len(cropped_points)/len(points)*100:.1f}%")
    
    if len(cropped_points) == 0:
        print("⚠ 警告: 切割后点云为空，请检查范围设置")
        return points, colors
    
    return cropped_points, cropped_colors

def crop_pointcloud_interactive(points, colors):
    """
    交互式点云切割 (命令行版本)
    """
    print("\n=== 交互式点云切割 ===")
    print("请输入要保留的点云范围 (留空使用默认值)")
    
    # 显示当前点云范围
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    
    print(f"\n当前点云范围:")
    print(f"  X: [{x_min:.3f}, {x_max:.3f}]")
    print(f"  Y: [{y_min:.3f}, {y_max:.3f}]")
    print(f"  Z: [{z_min:.3f}, {z_max:.3f}]")
    print(f"  总点数: {len(points)}")
    
    # 用户输入切割范围
    try:
        print("\n请输入切割范围 (直接按回车保持当前值):")
        
        x_min_input = input(f"X最小值 (当前: {x_min:.3f}): ").strip()
        x_min_crop = float(x_min_input) if x_min_input else x_min
        
        x_max_input = input(f"X最大值 (当前: {x_max:.3f}): ").strip()
        x_max_crop = float(x_max_input) if x_max_input else x_max
        
        y_min_input = input(f"Y最小值 (当前: {y_min:.3f}): ").strip()
        y_min_crop = float(y_min_input) if y_min_input else y_min
        
        y_max_input = input(f"Y最大值 (当前: {y_max:.3f}): ").strip()
        y_max_crop = float(y_max_input) if y_max_input else y_max
        
        z_min_input = input(f"Z最小值 (当前: {z_min:.3f}): ").strip()
        z_min_crop = float(z_min_input) if z_min_input else z_min
        
        z_max_input = input(f"Z最大值 (当前: {z_max:.3f}): ").strip()
        z_max_crop = float(z_max_input) if z_max_input else z_max
        
        # 验证范围
        if x_min_crop >= x_max_crop or y_min_crop >= y_max_crop or z_min_crop >= z_max_crop:
            print("❌ 错误: 最小值必须小于最大值")
            return points, colors
        
    except ValueError:
        print("❌ 输入格式错误，使用原始范围")
        return points, colors
    
    # 执行切割
    print(f"\n执行切割...")
    print(f"切割范围: X[{x_min_crop:.3f}, {x_max_crop:.3f}] "
          f"Y[{y_min_crop:.3f}, {y_max_crop:.3f}] "
          f"Z[{z_min_crop:.3f}, {z_max_crop:.3f}]")
    
    mask = ((points[:, 0] >= x_min_crop) & (points[:, 0] <= x_max_crop) &
            (points[:, 1] >= y_min_crop) & (points[:, 1] <= y_max_crop) &
            (points[:, 2] >= z_min_crop) & (points[:, 2] <= z_max_crop))
    
    cropped_points = points[mask]
    cropped_colors = colors[mask]
    
    print(f"\n✓ 切割完成:")
    print(f"  原始点数: {len(points)}")
    print(f"  切割后点数: {len(cropped_points)}")
    print(f"  保留比例: {len(cropped_points)/len(points)*100:.1f}%")
    
    if len(cropped_points) == 0:
        print("⚠ 警告: 切割后点云为空，请检查范围设置")
        return points, colors
    
    return cropped_points, cropped_colors


def crop_pointcloud_visual_interactive(points, colors):
    """
    可视化界面中的交互式点云切割 (增强版)
    支持多次操作、反向选择、凸包裁剪
    
    Args:
        points: Nx3 点云坐标
        colors: Nx3 点云颜色
    
    Returns:
        cropped_points, cropped_colors: 切割后的点云和颜色
    """
    print("\n=== 可视化交互式点云切割 (增强版) ===")
    print("功能:")
    print("- 支持多次选择操作")
    print("- 可以选择保留或删除选中区域")
    print("- 支持凸包裁剪（非矩形）")
    print("- 支持精确点选择")
    
    current_points = points.copy()
    current_colors = colors.copy()
    operation_count = 0
    
    while True:
        operation_count += 1
        print(f"\n--- 第 {operation_count} 次操作 ---")
        print(f"当前点数: {len(current_points)}")
        
        # 显示当前点云范围（调试信息）
        print(f"当前点云范围:")
        print(f"  X: [{current_points[:, 0].min():.3f}, {current_points[:, 0].max():.3f}]")
        print(f"  Y: [{current_points[:, 1].min():.3f}, {current_points[:, 1].max():.3f}]")
        print(f"  Z: [{current_points[:, 2].min():.3f}, {current_points[:, 2].max():.3f}]")
        
        # 创建点云对象（确保数据是连续的副本）
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(current_points))
        pcd.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(current_colors))
        
        # 验证点云对象
        print(f"点云对象验证:")
        print(f"  - 点数量: {len(pcd.points)}")
        print(f"  - 颜色数量: {len(pcd.colors)}")
        print(f"  - 是否为空: {pcd.is_empty()}")
        
        if pcd.is_empty():
            print("❌ 错误: 点云对象为空！")
            return points, colors
        
        # 添加坐标轴（确保大小合理）
        x_range = current_points[:, 0].max() - current_points[:, 0].min()
        y_range = current_points[:, 1].max() - current_points[:, 1].min()
        z_range = current_points[:, 2].max() - current_points[:, 2].min()
        coordinate_size = max(x_range, y_range, z_range) * 0.1
        coordinate_size = max(coordinate_size, 0.01)  # 最小1cm，避免太小
        
        print(f"坐标轴大小: {coordinate_size:.3f}m")
        
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coordinate_size)
        
        # 使用Open3D的可视化工具进行交互式选择
        print("\n正在打开可视化窗口...")
        print("=" * 60)
        print("【交互式选择说明】")
        print("1. 按住 Shift + 鼠标左键 拖拽来框选点")
        print("2. 可以多次框选来选择多个区域")
        print("3. 按 'Ctrl/Cmd + 鼠标左键' 可以取消选择")
        print("4. 选择完成后，按 Q 键或关闭窗口")
        print("=" * 60)
        
        # 确保之前的窗口已关闭
        import time
        time.sleep(0.2)  # 延迟确保资源释放
        
        vis = o3d.visualization.VisualizerWithEditing()
        picked_points = []
        
        try:
            success = vis.create_window(
                window_name=f"交互式点云切割 - 操作 {operation_count}", 
                width=1200, 
                height=900
            )
            
            if not success:
                print("❌ 错误: 无法创建可视化窗口")
                return points, colors
            
            vis.add_geometry(pcd)
            vis.add_geometry(coordinate_frame)
            
            # 设置相机视角
            ctr = vis.get_view_control()
            ctr.set_zoom(0.8)
            
            # 设置渲染选项 - 减小点大小，避免显示为圆形
            opt = vis.get_render_option()
            opt.point_size = 1.0  # 使用较小的点大小
            opt.background_color = np.asarray([0.1, 0.1, 0.1])  # 深色背景
            
            # 禁用一些可能导致圆形显示的选项
            opt.point_show_normal = False
            
            print("\n✓ 窗口已打开，等待用户选择...")
            print("  提示: 如果无法选择点，请尝试:")
            print("  1. 确保窗口处于焦点状态")
            print("  2. 先按一次 Shift 键")
            print("  3. 然后按住 Shift + 鼠标左键拖拽框选")
            
            # 运行可视化器
            vis.run()
            
            # 获取选择的点的索引
            picked_points = vis.get_picked_points()
            print(f"\n获取到的选择点数量: {len(picked_points)}")
            
        except Exception as e:
            print(f"❌ 可视化过程出错: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # 确保窗口被正确关闭
            try:
                vis.destroy_window()
            except:
                pass
            del vis
            time.sleep(0.2)  # 确保资源完全释放
        
        if len(picked_points) == 0:
            print("❌ 没有选择任何点")
            choice = input("是否继续操作? (y/n, 默认n): ").strip().lower()
            if choice != 'y':
                break
            else:
                continue
        
        print(f"✓ 选择了 {len(picked_points)} 个点")
        
        # 选择操作类型
        print("\n选择操作:")
        print("1. 保留选中区域（删除其他）")
        print("2. 删除选中区域（保留其他）")
        operation_type = input("请选择 (1/2): ").strip()
        
        # 选择裁剪方法
        print("\n选择裁剪方法:")
        print("1. 精确选择（仅操作选中的点）")
        print("2. 边界框（矩形框）")
        print("3. 3D凸包（紧贴选中点的3D多边形）")
        print("4. 2D投影凸包（在平面上投影，保留高度/深度）⭐推荐")
        print("5. 球形邻域（以选中点为中心的球形区域）")
        crop_method = input("请选择 (1/2/3/4/5, 默认4): ").strip()
        if not crop_method:
            crop_method = '4'
        
        selected_indices = np.array(picked_points)
        
        # 根据方法生成mask
        if crop_method == '1':
            # 方法1: 精确选择
            mask = np.zeros(len(current_points), dtype=bool)
            mask[selected_indices] = True
            print(f"精确选择: {np.sum(mask)} 个点")
            
        elif crop_method == '2':
            # 方法2: 边界框
            selected_points = current_points[selected_indices]
            
            x_min, x_max = selected_points[:, 0].min(), selected_points[:, 0].max()
            y_min, y_max = selected_points[:, 1].min(), selected_points[:, 1].max()
            z_min, z_max = selected_points[:, 2].min(), selected_points[:, 2].max()
            
            # 询问是否扩展
            expand = input(f"是否扩展边界框? (y/n, 默认y): ").strip().lower()
            if expand != 'n':
                try:
                    margin_input = input("扩展距离 (米, 默认0.02): ").strip()
                    margin = float(margin_input) if margin_input else 0.02
                except ValueError:
                    margin = 0.02
                
                x_min -= margin
                x_max += margin
                y_min -= margin
                y_max += margin
                z_min -= margin
                z_max += margin
            
            mask = ((current_points[:, 0] >= x_min) & (current_points[:, 0] <= x_max) &
                    (current_points[:, 1] >= y_min) & (current_points[:, 1] <= y_max) &
                    (current_points[:, 2] >= z_min) & (current_points[:, 2] <= z_max))
            
            print(f"边界框范围: X[{x_min:.3f}, {x_max:.3f}] Y[{y_min:.3f}, {y_max:.3f}] Z[{z_min:.3f}, {z_max:.3f}]")
            print(f"包含 {np.sum(mask)} 个点")
            
        elif crop_method == '3':
            # 方法3: 3D凸包
            selected_points = current_points[selected_indices]
            
            # 询问扩展距离
            try:
                margin_input = input("安全边距 (米, 默认0.01): ").strip()
                margin = float(margin_input) if margin_input else 0.01
            except ValueError:
                margin = 0.01
            
            # 使用scipy的凸包
            from scipy.spatial import ConvexHull
            
            try:
                hull = ConvexHull(selected_points)
                hull_points = selected_points[hull.vertices]
                
                # 对每个点检查是否在凸包内
                from scipy.spatial import Delaunay
                delaunay = Delaunay(hull_points)
                mask = delaunay.find_simplex(current_points) >= 0
                
                # 扩展区域
                if margin > 0:
                    # 找到凸包内的点，然后找它们的邻域
                    hull_indices = np.where(mask)[0]
                    if len(hull_indices) > 0:
                        # 计算所有点到凸包内点的最小距离
                        from scipy.spatial.distance import cdist
                        distances = cdist(current_points, current_points[hull_indices])
                        min_distances = distances.min(axis=1)
                        mask = min_distances <= margin
                
                print(f"3D凸包裁剪 (边距: {margin}m): {np.sum(mask)} 个点")
                
            except Exception as e:
                print(f"凸包计算失败: {e}")
                print("回退到边界框方法...")
                selected_points = current_points[selected_indices]
                x_min, x_max = selected_points[:, 0].min() - margin, selected_points[:, 0].max() + margin
                y_min, y_max = selected_points[:, 1].min() - margin, selected_points[:, 1].max() + margin
                z_min, z_max = selected_points[:, 2].min() - margin, selected_points[:, 2].max() + margin
                mask = ((current_points[:, 0] >= x_min) & (current_points[:, 0] <= x_max) &
                        (current_points[:, 1] >= y_min) & (current_points[:, 1] <= y_max) &
                        (current_points[:, 2] >= z_min) & (current_points[:, 2] <= z_max))
        
        elif crop_method == '4':
            # 方法4: 2D投影凸包（保留高度/深度）⭐ 推荐用于处理垂直结构
            selected_points = current_points[selected_indices]
            
            # 选择投影平面
            print("\n选择投影平面:")
            print("1. XY平面 (俯视图，保留整个Z轴/高度)")
            print("2. XZ平面 (前视图，保留整个Y轴/深度)")
            print("3. YZ平面 (侧视图，保留整个X轴/宽度)")
            plane_choice = input("请选择 (1/2/3, 默认1): ").strip()
            if not plane_choice:
                plane_choice = '1'
            
            # 询问扩展距离
            try:
                margin_input = input("安全边距 (米, 默认0=不扩展，避免圆形痕迹): ").strip()
                margin = float(margin_input) if margin_input else 0.0
            except ValueError:
                margin = 0.0
            
            # 根据选择的平面进行投影
            if plane_choice == '1':
                # XY平面投影 (保留整个Z轴)
                proj_dims = [0, 1]  # X, Y
                preserve_dim = 2  # Z
                plane_name = "XY平面"
                preserve_name = "Z轴(高度)"
            elif plane_choice == '2':
                # XZ平面投影 (保留整个Y轴)
                proj_dims = [0, 2]  # X, Z
                preserve_dim = 1  # Y
                plane_name = "XZ平面"
                preserve_name = "Y轴(深度)"
            else:
                # YZ平面投影 (保留整个X轴)
                proj_dims = [1, 2]  # Y, Z
                preserve_dim = 0  # X
                plane_name = "YZ平面"
                preserve_name = "X轴(宽度)"
            
            try:
                from scipy.spatial import ConvexHull, Delaunay
                
                # 显示选择点的信息
                print(f"\n调试信息 - 选择的点:")
                print(f"  选择点数: {len(selected_points)}")
                print(f"  X范围: [{selected_points[:, 0].min():.3f}, {selected_points[:, 0].max():.3f}]")
                print(f"  Y范围: [{selected_points[:, 1].min():.3f}, {selected_points[:, 1].max():.3f}]")
                print(f"  Z范围: [{selected_points[:, 2].min():.3f}, {selected_points[:, 2].max():.3f}]")
                
                # 在2D平面上计算凸包
                selected_2d = selected_points[:, proj_dims]
                print(f"\n2D投影后选择的点:")
                print(f"  投影点数: {len(selected_2d)}")
                for i, dim in enumerate(proj_dims):
                    dim_name = ['X', 'Y', 'Z'][dim]
                    print(f"  {dim_name}范围: [{selected_2d[:, i].min():.3f}, {selected_2d[:, i].max():.3f}]")
                
                hull_2d = ConvexHull(selected_2d)
                hull_points_2d = selected_2d[hull_2d.vertices]
                print(f"  2D凸包顶点数: {len(hull_points_2d)}")
                
                # ===== 关键修改：在2D平面上判断，保留该区域所有Z高度的点 =====
                # 检查所有点的2D投影是否在凸包内
                current_2d = current_points[:, proj_dims]
                delaunay_2d = Delaunay(hull_points_2d)
                mask_2d = delaunay_2d.find_simplex(current_2d) >= 0
                
                print(f"\n2D凸包内的点（所有{preserve_name}）: {np.sum(mask_2d)}")
                
                # 扩展区域（仅在2D投影平面上扩展，保留所有preserve_dim的值）
                if margin > 0:
                    print(f"正在扩展边界 (边距: {margin}m)...")
                    
                    # 不扩展，避免边缘产生圆形痕迹
                    # 如果用户需要包含边缘附近的点，建议：
                    # 1. 选点时多选一些边缘点
                    # 2. 或者使用边界框方法后手动扩展
                    print(f"⚠ 提示: 为避免边缘圆形痕迹，不进行边距扩展")
                    print(f"  如需包含更多边缘点，请在选择时多选一些边缘区域的点")
                    mask = mask_2d
                else:
                    mask = mask_2d
                
                print(f"\n=== 2D投影凸包裁剪结果 ===")
                print(f"  投影平面: {plane_name}")
                print(f"  保留维度: {preserve_name} (所有高度/深度)")
                print(f"  安全边距: {margin}m")
                print(f"  包含点数: {np.sum(mask)}")
                
                # 显示保留维度的完整范围
                preserved_points = current_points[mask]
                if len(preserved_points) > 0:
                    preserve_min = preserved_points[:, preserve_dim].min()
                    preserve_max = preserved_points[:, preserve_dim].max()
                    preserve_range = preserve_max - preserve_min
                    
                    # 同时显示整个点云的该维度范围作为对比
                    total_preserve_min = current_points[:, preserve_dim].min()
                    total_preserve_max = current_points[:, preserve_dim].max()
                    total_preserve_range = total_preserve_max - total_preserve_min
                    
                    print(f"  {preserve_name}范围: [{preserve_min:.3f}, {preserve_max:.3f}] (跨度: {preserve_range:.3f}m)")
                    print(f"  原始{preserve_name}范围: [{total_preserve_min:.3f}, {total_preserve_max:.3f}] (跨度: {total_preserve_range:.3f}m)")
                    print(f"  保留了 {preserve_range/total_preserve_range*100:.1f}% 的{preserve_name}范围")
                    
                    # 额外显示投影平面的范围
                    print(f"  投影平面范围:")
                    for i, dim in enumerate(proj_dims):
                        dim_name = ['X', 'Y', 'Z'][dim]
                        dim_min = preserved_points[:, dim].min()
                        dim_max = preserved_points[:, dim].max()
                        dim_range = dim_max - dim_min
                        print(f"    {dim_name}轴: [{dim_min:.3f}, {dim_max:.3f}] (跨度: {dim_range:.3f}m)")
                
            except Exception as e:
                print(f"2D凸包计算失败: {e}")
                import traceback
                traceback.print_exc()
                print("回退到边界框方法...")
                selected_points = current_points[selected_indices]
                x_min, x_max = selected_points[:, 0].min() - margin, selected_points[:, 0].max() + margin
                y_min, y_max = selected_points[:, 1].min() - margin, selected_points[:, 1].max() + margin
                z_min, z_max = selected_points[:, 2].min() - margin, selected_points[:, 2].max() + margin
                mask = ((current_points[:, 0] >= x_min) & (current_points[:, 0] <= x_max) &
                        (current_points[:, 1] >= y_min) & (current_points[:, 1] <= y_max) &
                        (current_points[:, 2] >= z_min) & (current_points[:, 2] <= z_max))
        
        elif crop_method == '5':
            # 方法5: 球形邻域
            selected_points = current_points[selected_indices]
            center = selected_points.mean(axis=0)
            
            try:
                radius_input = input("球形半径 (米, 默认0.05): ").strip()
                radius = float(radius_input) if radius_input else 0.05
            except ValueError:
                radius = 0.05
            
            distances = np.linalg.norm(current_points - center, axis=1)
            mask = distances <= radius
            
            print(f"球形邻域 (中心: {center}, 半径: {radius}m): {np.sum(mask)} 个点")
        
        else:
            print("无效选择，使用精确选择")
            mask = np.zeros(len(current_points), dtype=bool)
            mask[selected_indices] = True
        
        # 应用操作
        if operation_type == '1':
            # 保留选中区域
            current_points = current_points[mask].copy()  # 确保创建副本
            current_colors = current_colors[mask].copy()
            print(f"✓ 保留选中区域: 剩余 {len(current_points)} 个点")
        else:
            # 删除选中区域
            current_points = current_points[~mask].copy()  # 确保创建副本
            current_colors = current_colors[~mask].copy()
            print(f"✓ 删除选中区域: 剩余 {len(current_points)} 个点")
        
        if len(current_points) == 0:
            print("⚠ 警告: 点云已为空！")
            return points, colors
        
        # 验证数据有效性
        if not np.isfinite(current_points).all():
            print("⚠ 警告: 点云数据包含无效值（NaN或Inf）")
            # 过滤无效点
            valid_mask = np.isfinite(current_points).all(axis=1)
            current_points = current_points[valid_mask].copy()
            current_colors = current_colors[valid_mask].copy()
            print(f"过滤后剩余 {len(current_points)} 个有效点")
            
            if len(current_points) == 0:
                print("❌ 错误: 所有点都无效！")
                return points, colors
        
        # 预览当前结果
        preview = input("\n是否预览当前结果? (y/n, 默认y): ").strip().lower()
        if preview != 'n':
            visualize_pointcloud(current_points, current_colors, f"当前结果 (操作{operation_count}次)")
        
        # 询问是否继续
        cont = input("\n是否继续操作? (y/n, 默认n): ").strip().lower()
        if cont != 'y':
            break
    
    print(f"\n✓ 交互式切割完成:")
    print(f"  原始点数: {len(points)}")
    print(f"  最终点数: {len(current_points)}")
    print(f"  保留比例: {len(current_points)/len(points)*100:.1f}%")
    print(f"  操作次数: {operation_count}")
    
    return current_points, current_colors



def save_pointcloud_pt(points, colors, output_dir):
    """
    保存点云为PT文件
    
    Args:
        points: Nx3 点云坐标
        colors: Nx3 点云颜色  
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    points_tensor = torch.from_numpy(points.astype(np.float32))
    colors_tensor = torch.from_numpy(colors.astype(np.float32))
    
    torch.save(points_tensor, os.path.join(output_dir, "points.pt"))
    torch.save(colors_tensor, os.path.join(output_dir, "colors.pt"))
    
    print(f"✓ 保存完成:")
    print(f"  - {output_dir}/points.pt ({points_tensor.shape})")
    print(f"  - {output_dir}/colors.pt ({colors_tensor.shape})")

def main_crop_workflow(input_dir, output_dir, visualize_before=True, visualize_after=True, crop_mode='bbox'):
    """
    完整的点云切割工作流程
    
    Args:
        input_dir: 输入目录 (包含points.pt和colors.pt)
        output_dir: 输出目录
        visualize_before: 是否在切割前可视化
        visualize_after: 是否在切割后可视化
        crop_mode: 切割模式 ('bbox'=边界框, 'interactive'=交互式选择, 'command'=命令行)
    """
    print("=== 点云切割工具 ===")
    
    # 1. 加载点云
    print("\n1. 加载点云数据...")
    points, colors = load_pointcloud_pt(input_dir)
    
    # 显示基本信息
    print(f"\n点云信息:")
    print(f"  点数量: {len(points)}")
    print(f"  坐标范围: X[{points[:,0].min():.3f}, {points[:,0].max():.3f}] "
          f"Y[{points[:,1].min():.3f}, {points[:,1].max():.3f}] "
          f"Z[{points[:,2].min():.3f}, {points[:,2].max():.3f}]")
    
    # 2. 切割前可视化（interactive模式默认跳过）
    if visualize_before and crop_mode != 'interactive':
        print("\n2. 可视化原始点云...")
        visualize_pointcloud(points, colors, "原始点云")
    elif crop_mode == 'interactive':
        print("\n2. 跳过原始点云可视化（interactive模式）")
    
    # 2.5 询问是否去除离散点
    remove_outlier_choice = input("\n是否去除离散点/噪声点? (y/n, 默认n): ").strip().lower()
    if remove_outlier_choice == 'y':
        points, colors = remove_outliers(points, colors)
        
        # 可视化去噪后的结果
        preview_choice = input("\n是否预览去噪后的点云? (y/n, 默认y): ").strip().lower()
        if preview_choice != 'n':
            visualize_pointcloud(points, colors, "去噪后的点云")
    
    # 3. 交互式切割
    print(f"\n3. 开始切割 (模式: {crop_mode})...")
    
    while True:
        choice = input("是否进行点云切割? (y/n): ").strip().lower()
        if choice == 'y':
            
            if crop_mode == 'bbox':
                cropped_points, cropped_colors = crop_with_bounding_box(points, colors)
            elif crop_mode == 'interactive':
                cropped_points, cropped_colors = crop_pointcloud_visual_interactive(points, colors)
            else:  # command
                cropped_points, cropped_colors = crop_pointcloud_interactive(points, colors)
            
            # 切割后可视化
            if visualize_after and len(cropped_points) > 0:
                print("\n4. 可视化切割结果...")
                visualize_pointcloud(cropped_points, cropped_colors, "切割后的点云")
            
            # 询问是否满意
            if len(cropped_points) > 0:
                satisfy = input("\n是否满意切割结果? (y/n): ").strip().lower()
                if satisfy == 'y':
                    points, colors = cropped_points, cropped_colors
                    break
                else:
                    print("重新进行切割...")
                    continue
            else:
                print("切割失败，重新尝试...")
                continue
                
        elif choice == 'n':
            print("跳过切割，使用原始点云")
            break
        else:
            print("请输入 y 或 n")
    
    # 5. 保存结果
    print("\n5. 保存点云...")
    save_pointcloud_pt(points, colors, output_dir)
    
    return points, colors

if __name__ == "__main__":
    import argparse
    
    # 默认路径配置
    INPUT_DIR = "/home/hkcrc/diffusion_edfs/diffusion_edf/dataset/1030_baselink_pt/grasp0/cropped"
    OUTPUT_DIR = "/home/hkcrc/diffusion_edfs/diffusion_edf/dataset/1030_baselink_pt/grasp0/cropped"
    
    parser = argparse.ArgumentParser(description="点云切割工具")
    parser.add_argument("--input", default=INPUT_DIR, 
                       help=f"输入目录 (默认: {INPUT_DIR})")
    parser.add_argument("--output", default=OUTPUT_DIR,
                       help=f"输出目录 (默认: {OUTPUT_DIR})")
    parser.add_argument("--mode", choices=['bbox', 'interactive', 'command'], default='bbox',
                       help="切割模式: bbox(边界框观察), interactive(交互选择), command(命令行)")
    parser.add_argument("--no-viz-before", action='store_true',
                       help="跳过切割前的可视化")
    parser.add_argument("--no-viz-after", action='store_true', 
                       help="跳过切割后的可视化")
    
    args = parser.parse_args()
    
    try:
        main_crop_workflow(
            input_dir=args.input,
            output_dir=args.output,
            visualize_before=not args.no_viz_before,
            visualize_after=not args.no_viz_after,
            crop_mode=args.mode
        )
        print("\n✅ 点云切割完成!")
        
    except Exception as e:
        print(f"\n❌ 切割失败: {e}")
        import traceback
        traceback.print_exc()

"""
使用示例:

1. 边界框模式 (推荐，可观察尺度):
   python crop_pointcloud.py --mode bbox

2. 交互式选择模式 (增强版，支持反向选择、凸包裁剪、多次操作):
   python crop_pointcloud.py --mode interactive

3. 命令行模式:
   python crop_pointcloud.py --mode command

4. 指定路径:
   python crop_pointcloud.py --input /path/to/input --output /path/to/output --mode interactive

功能特性:
【点云切割】
- 保留或删除选中区域
- 精确点选择
- 边界框裁剪（矩形）
- 3D凸包裁剪（紧贴选中点的3D多边形）
- 2D投影凸包裁剪（保留完整高度，推荐用于垂直结构）
- 球形邻域裁剪
- 支持多次操作，逐步精细化

【去除离散点】
- 统计滤波 (Statistical Outlier Removal)
  - 基于邻域点距离统计
  - 推荐参数: 邻居数20-50, 标准差倍数1.0-2.0
  - 适合去除稀疏噪声点
  
- 半径滤波 (Radius Outlier Removal)
  - 基于半径内邻居数量
  - 推荐参数: 半径0.02-0.10m, 最少邻居5-15
  - 适合去除孤立点簇

工作流程:
1. 加载点云
2. (可选) 去除离散点/噪声点
3. (可选) 空间裁剪
4. 保存结果
"""