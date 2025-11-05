#!/usr/bin/env python3
"""
简化版点云裁剪工具 - 不依赖VisualizerWithEditing
使用可视化观察 + 命令行输入的方式
"""

import numpy as np
import torch
import open3d as o3d
import os
import sys

# 添加原crop_pointcloud.py的路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crop_pointcloud import (
    load_pointcloud_pt, 
    save_pointcloud_pt, 
    visualize_pointcloud,
    remove_outliers
)


def crop_with_visual_bbox(points, colors):
    """
    可视化观察 + 边界框输入的裁剪方法
    避免使用VisualizerWithEditing
    """
    print("\n=== 简化版裁剪模式 ===")
    print("1. 先观察点云，记录要保留的区域")
    print("2. 然后输入边界框参数进行裁剪")
    
    current_points = points.copy()
    current_colors = colors.copy()
    operation_count = 0
    
    while True:
        operation_count += 1
        print(f"\n--- 第 {operation_count} 次操作 ---")
        print(f"当前点数: {len(current_points)}")
        
        # 显示当前范围
        x_min, x_max = current_points[:, 0].min(), current_points[:, 0].max()
        y_min, y_max = current_points[:, 1].min(), current_points[:, 1].max()
        z_min, z_max = current_points[:, 2].min(), current_points[:, 2].max()
        
        print(f"\n当前点云范围:")
        print(f"  X: [{x_min:.3f}, {x_max:.3f}] (红色轴)")
        print(f"  Y: [{y_min:.3f}, {y_max:.3f}] (绿色轴)")
        print(f"  Z: [{z_min:.3f}, {z_max:.3f}] (蓝色轴)")
        
        # 可视化当前点云
        print("\n显示点云，请观察并记录要保留的区域范围...")
        visualize_pointcloud(current_points, current_colors, f"观察点云 - 操作{operation_count}")
        
        # 选择操作类型
        print("\n选择操作:")
        print("1. 保留指定区域（删除其他）")
        print("2. 删除指定区域（保留其他）")
        print("3. 完成裁剪")
        operation_type = input("请选择 (1/2/3): ").strip()
        
        if operation_type == '3':
            break
        
        if operation_type not in ['1', '2']:
            print("无效选择，请重新操作")
            continue
        
        # 输入边界框
        print("\n请输入要操作的区域边界框 (直接回车保持当前边界):")
        
        try:
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
                continue
                
        except ValueError:
            print("❌ 输入格式错误")
            continue
        
        # 创建mask
        mask = ((current_points[:, 0] >= x_min_crop) & (current_points[:, 0] <= x_max_crop) &
                (current_points[:, 1] >= y_min_crop) & (current_points[:, 1] <= y_max_crop) &
                (current_points[:, 2] >= z_min_crop) & (current_points[:, 2] <= z_max_crop))
        
        print(f"\n边界框范围:")
        print(f"  X: [{x_min_crop:.3f}, {x_max_crop:.3f}]")
        print(f"  Y: [{y_min_crop:.3f}, {y_max_crop:.3f}]")
        print(f"  Z: [{z_min_crop:.3f}, {z_max_crop:.3f}]")
        print(f"匹配点数: {np.sum(mask)}")
        
        # 应用操作
        if operation_type == '1':
            # 保留选中区域
            current_points = current_points[mask].copy()
            current_colors = current_colors[mask].copy()
            print(f"✓ 保留选中区域: 剩余 {len(current_points)} 个点")
        else:
            # 删除选中区域
            current_points = current_points[~mask].copy()
            current_colors = current_colors[~mask].copy()
            print(f"✓ 删除选中区域: 剩余 {len(current_points)} 个点")
        
        if len(current_points) == 0:
            print("⚠ 警告: 点云已为空！")
            return points, colors
        
        # 预览结果
        preview = input("\n是否预览当前结果? (y/n, 默认y): ").strip().lower()
        if preview != 'n':
            visualize_pointcloud(current_points, current_colors, f"当前结果 (操作{operation_count}次)")
    
    print(f"\n✓ 裁剪完成:")
    print(f"  原始点数: {len(points)}")
    print(f"  最终点数: {len(current_points)}")
    print(f"  保留比例: {len(current_points)/len(points)*100:.1f}%")
    print(f"  操作次数: {operation_count}")
    
    return current_points, current_colors


def main():
    """主函数"""
    import argparse
    
    # 默认路径
    INPUT_DIR = "/home/hkcrc/diffusion_edfs/diffusion_edf/output/output_pcd_baselink/"
    OUTPUT_DIR = "/home/hkcrc/diffusion_edfs/diffusion_edf/output/test_scene_pcd_yzj_cropped2"
    
    parser = argparse.ArgumentParser(description="简化版点云裁剪工具")
    parser.add_argument("--input", default=INPUT_DIR, help="输入目录")
    parser.add_argument("--output", default=OUTPUT_DIR, help="输出目录")
    
    args = parser.parse_args()
    
    try:
        print("=== 简化版点云裁剪工具 ===")
        print("（避免VisualizerWithEditing的兼容性问题）\n")
        
        # 1. 加载点云
        print("1. 加载点云数据...")
        points, colors = load_pointcloud_pt(args.input)
        
        print(f"\n点云信息:")
        print(f"  点数量: {len(points)}")
        print(f"  坐标范围: X[{points[:,0].min():.3f}, {points[:,0].max():.3f}] "
              f"Y[{points[:,1].min():.3f}, {points[:,1].max():.3f}] "
              f"Z[{points[:,2].min():.3f}, {points[:,2].max():.3f}]")
        
        # 2. 去除离散点
        remove_outlier = input("\n2. 是否去除离散点/噪声点? (y/n, 默认n): ").strip().lower()
        if remove_outlier == 'y':
            points, colors = remove_outliers(points, colors)
            
            preview = input("\n是否预览去噪后的点云? (y/n, 默认y): ").strip().lower()
            if preview != 'n':
                visualize_pointcloud(points, colors, "去噪后的点云")
        
        # 3. 裁剪
        crop_choice = input("\n3. 是否进行点云裁剪? (y/n, 默认y): ").strip().lower()
        if crop_choice != 'n':
            points, colors = crop_with_visual_bbox(points, colors)
        
        # 4. 最终预览
        final_preview = input("\n4. 是否预览最终结果? (y/n, 默认y): ").strip().lower()
        if final_preview != 'n':
            visualize_pointcloud(points, colors, "最终结果")
        
        # 5. 保存
        save_choice = input("\n5. 是否保存结果? (y/n, 默认y): ").strip().lower()
        if save_choice != 'n':
            save_pointcloud_pt(points, colors, args.output)
            print("\n✅ 处理完成!")
        else:
            print("\n取消保存")
            
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
