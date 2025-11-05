import open3d as o3d
import numpy as np
import torch
import os

def check_file_structure(file_path):
    """
    检查文件的结构和包含的数据（支持PLY和PT格式）
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    print(f"=== 分析文件: {file_path} ===")
    print(f"文件类型: {file_ext}")
    
    if file_ext == '.ply':
        return check_ply_structure(file_path)
    elif file_ext == '.pt':
        return check_pt_structure(file_path)
    else:
        print(f"不支持的文件格式: {file_ext}")
        return None

def check_ply_structure(ply_file_path):
    """
    检查PLY文件的结构和包含的数据
    """
    try:
        import plyfile
        plydata = plyfile.PlyData.read(ply_file_path)
        print("\n--- PLY文件原始结构 ---")
        
        for element in plydata.elements:
            print(f"元素: {element.name}")
            print(f"数量: {element.count}")
            print("属性:")
            for prop in element.properties:
                print(f"  - {prop.name} ({prop.val_dtype})")
            print()
            
    except Exception as e:
        print(f"无法使用plyfile读取: {e}")
    
    # 使用Open3D读取
    try:
        pcd = o3d.io.read_point_cloud(ply_file_path)
        print("\n--- Open3D读取结果 ---")
        print(f"点数量: {len(pcd.points)}")
        print(f"包含颜色: {len(pcd.colors) > 0}")
        print(f"包含法向量: {len(pcd.normals) > 0}")
        
        result = {
            'points': np.asarray(pcd.points),
            'colors': np.asarray(pcd.colors) if len(pcd.colors) > 0 else None,
            'normals': np.asarray(pcd.normals) if len(pcd.normals) > 0 else None,
            'pcd': pcd
        }
        
        if len(pcd.normals) > 0:
            normals = np.asarray(pcd.normals)
            print(f"法向量数量: {len(normals)}")
            print(f"法向量范围: X({normals[:, 0].min():.3f}, {normals[:, 0].max():.3f}) "
                  f"Y({normals[:, 1].min():.3f}, {normals[:, 1].max():.3f}) "
                  f"Z({normals[:, 2].min():.3f}, {normals[:, 2].max():.3f})")
            
            # 检查法向量是否归一化
            norms = np.linalg.norm(normals, axis=1)
            print(f"法向量模长范围: {norms.min():.6f} 到 {norms.max():.6f}")
            if np.allclose(norms, 1.0, atol=1e-3):
                print("✓ 法向量已归一化")
            else:
                print("⚠ 法向量可能未归一化")
        else:
            print("❌ 不包含法向量")
            
        return result
            
    except Exception as e:
        print(f"Open3D读取失败: {e}")
        return None

def check_pt_structure(pt_file_path):
    """
    检查PT文件的结构和包含的数据
    """
    file_dir = os.path.dirname(pt_file_path)
    file_name = os.path.basename(pt_file_path)
    
    print("\n--- PT文件分析 ---")
    
    # 检查同目录下的相关文件
    related_files = {}
    possible_files = ['points.pt', 'colors.pt', 'normals.pt']
    
    for pf in possible_files:
        file_path = os.path.join(file_dir, pf)
        if os.path.exists(file_path):
            try:
                data = torch.load(file_path).cpu().numpy()
                related_files[pf] = data
                print(f"✓ {pf}: {data.shape} (范围: {data.min():.6f} 到 {data.max():.6f})")
            except Exception as e:
                print(f"❌ 无法读取 {pf}: {e}")
        else:
            print(f"❌ 未找到 {pf}")
    
    # 如果当前文件不是points.pt，也加载它
    if file_name not in related_files:
        try:
            data = torch.load(pt_file_path).cpu().numpy()
            related_files[file_name] = data
            print(f"✓ {file_name}: {data.shape} (范围: {data.min():.6f} 到 {data.max():.6f})")
        except Exception as e:
            print(f"❌ 无法读取 {file_name}: {e}")
    
    # 构建点云对象
    pcd = o3d.geometry.PointCloud()
    result = {'pcd': pcd}
    
    if 'points.pt' in related_files:
        points = related_files['points.pt']
        pcd.points = o3d.utility.Vector3dVector(points)
        result['points'] = points
        print(f"点数量: {len(points)}")
    
    if 'colors.pt' in related_files:
        colors = related_files['colors.pt']
        pcd.colors = o3d.utility.Vector3dVector(colors)
        result['colors'] = colors
        print(f"包含颜色: True")
    else:
        result['colors'] = None
        print(f"包含颜色: False")
    
    if 'normals.pt' in related_files:
        normals = related_files['normals.pt']
        pcd.normals = o3d.utility.Vector3dVector(normals)
        result['normals'] = normals
        print(f"包含法向量: True")
        print(f"法向量数量: {len(normals)}")
        print(f"法向量范围: X({normals[:, 0].min():.3f}, {normals[:, 0].max():.3f}) "
              f"Y({normals[:, 1].min():.3f}, {normals[:, 1].max():.3f}) "
              f"Z({normals[:, 2].min():.3f}, {normals[:, 2].max():.3f})")
        
        # 检查法向量是否归一化
        norms = np.linalg.norm(normals, axis=1)
        print(f"法向量模长范围: {norms.min():.6f} 到 {norms.max():.6f}")
        if np.allclose(norms, 1.0, atol=1e-3):
            print("✓ 法向量已归一化")
        else:
            print("⚠ 法向量可能未归一化")
    else:
        result['normals'] = None
        print(f"包含法向量: False")
    
    return result

def estimate_normals_if_needed(data_dict, search_radius=0.1, max_nn=30):
    """
    如果不包含法向量，则计算法向量
    """
    pcd = data_dict['pcd']
    
    if data_dict['normals'] is None:
        print("\n=== 计算法向量 ===")
        
        # 估计法向量
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=search_radius,    # 搜索半径
                max_nn=max_nn           # 最大邻居数
            )
        )
        
        # 统一法向量方向（可选）
        # pcd.orient_normals_consistent_tangent_plane(100)
        
        normals = np.asarray(pcd.normals)
        data_dict['normals'] = normals
        
        print(f"✓ 计算完成，法向量数量: {len(normals)}")
        print(f"法向量范围: X({normals[:, 0].min():.3f}, {normals[:, 0].max():.3f}) "
              f"Y({normals[:, 1].min():.3f}, {normals[:, 1].max():.3f}) "
              f"Z({normals[:, 2].min():.3f}, {normals[:, 2].max():.3f})")
        
        return True  # 表示计算了新的法向量
    else:
        print("✓ 已包含法向量，无需计算")
        return False  # 表示没有计算新的法向量

def save_data_as_pt(data_dict, output_dir, save_ply=False):
    """
    将数据保存为PT格式（和可选的PLY格式）
    """
    print(f"\n=== 保存数据到 {output_dir} ===")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存PT文件
    if data_dict['points'] is not None:
        points_tensor = torch.from_numpy(data_dict['points'].astype(np.float32))
        torch.save(points_tensor, os.path.join(output_dir, "points.pt"))
        print(f"✓ points.pt: {points_tensor.shape}")
    
    if data_dict['colors'] is not None:
        colors_tensor = torch.from_numpy(data_dict['colors'].astype(np.float32))
        torch.save(colors_tensor, os.path.join(output_dir, "colors.pt"))
        print(f"✓ colors.pt: {colors_tensor.shape}")
    
    if data_dict['normals'] is not None:
        normals_tensor = torch.from_numpy(data_dict['normals'].astype(np.float32))
        torch.save(normals_tensor, os.path.join(output_dir, "normals.pt"))
        print(f"✓ normals.pt: {normals_tensor.shape}")
    
    # 可选：保存PLY文件
    if save_ply:
        ply_path = os.path.join(output_dir, "pointcloud_with_normals.ply")
        o3d.io.write_point_cloud(ply_path, data_dict['pcd'])
        print(f"✓ PLY文件: {ply_path}")

def visualize_normals(pcd, normal_length=0.01, subsample_ratio=0.1):
    """
    可视化点云和法向量（支持子采样以提高性能）
    """
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    
    # 如果点太多，进行子采样
    if len(points) > 10000:
        indices = np.random.choice(len(points), 
                                 size=int(len(points) * subsample_ratio), 
                                 replace=False)
        points = points[indices]
        normals = normals[indices]
        print(f"为了可视化效果，子采样到 {len(points)} 个点")
    
    # 计算法向量终点
    normal_ends = points + normals * normal_length
    
    # 创建线段
    lines = []
    for i in range(len(points)):
        lines.append([i, i + len(points)])
    
    # 创建线段集合
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.vstack([points, normal_ends]))
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))])  # 红色法向量
    
    # 创建子采样的点云用于显示
    pcd_sub = o3d.geometry.PointCloud()
    pcd_sub.points = o3d.utility.Vector3dVector(points)
    if len(pcd.colors) > 0:
        colors = np.asarray(pcd.colors)
        if len(points) < len(colors):
            pcd_sub.colors = o3d.utility.Vector3dVector(colors[indices])
        else:
            pcd_sub.colors = o3d.utility.Vector3dVector(colors)
    
    # 可视化
    o3d.visualization.draw_geometries([pcd_sub, line_set], 
                                    window_name="点云与法向量",
                                    width=800, height=600)

if __name__ == "__main__":
    # 支持PLY和PT文件
    #file_path = "/home/hkcrc/DCIM/rs3/cloud.ply"  # 或者 "/path/to/points.pt"
    file_path = "/home/hkcrc/diffusion_edfs/diffusion_edf/output/test_scene_pcd_yzj/points.pt"
    # 检查文件结构
    data_dict = check_file_structure(file_path)
    
    if data_dict is None:
        print("文件读取失败")
        exit(1)
    
    # 如果需要，计算法向量
    output_directory = "output/output_with_normals"
    
    try:
        # 计算法向量（如果需要）
        computed_new_normals = estimate_normals_if_needed(data_dict)
        
        # 保存数据
        save_data_as_pt(data_dict, output_directory, save_ply=True)
        
        # 可选：可视化法向量
        print("\n是否可视化法向量? (y/n): ", end="")
        if input().lower() == 'y':
            visualize_normals(data_dict['pcd'])
            
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()