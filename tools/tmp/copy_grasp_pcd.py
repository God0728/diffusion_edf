#!/usr/bin/env python3
"""
复制grasp_pcd文件到demo目录

从 dataset/1030_baselink_pt/grasp0/cropped/ 
到 demo/rebar_grasping/data/demo_N/step_0/grasp_pcd/

从 dataset/1030_baselink_pt/grasp/cropped/
到 demo/rebar_grasping/data/demo_N/step_1/grasp_pcd/
"""

import os
import shutil
import yaml

def create_grasp_pcd_metadata(output_path):
    """创建grasp_pcd的metadata.yaml"""
    metadata = {
        '__type__': 'PointCloud',
        'name': '',
        'unit_length': '1 [m]'
    }
    
    metadata_file = os.path.join(output_path, 'metadata.yaml')
    with open(metadata_file, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    print(f"    ✓ 创建 metadata.yaml")


def copy_grasp_pcd(source_base, target_base):
    """
    复制grasp_pcd文件
    
    Args:
        source_base: 源目录基路径 (dataset/1030_baselink_pt)
        target_base: 目标目录基路径 (demo/rebar_grasping/data)
    """
    print("="*60)
    print("📋 Grasp PCD 文件复制工具")
    print("="*60)
    
    # 源文件路径
    grasp0_dir = os.path.join(source_base, 'grasp0', 'cropped')
    grasp_dir = os.path.join(source_base, 'grasp', 'cropped')
    
    grasp0_points = os.path.join(grasp0_dir, 'points.pt')
    grasp0_colors = os.path.join(grasp0_dir, 'colors.pt')
    grasp_points = os.path.join(grasp_dir, 'points.pt')
    grasp_colors = os.path.join(grasp_dir, 'colors.pt')
    
    # 检查源文件
    print(f"\n检查源文件...")
    files_ok = True
    
    if not os.path.exists(grasp0_points):
        print(f"❌ 源文件不存在: {grasp0_points}")
        files_ok = False
    else:
        print(f"✓ 找到: {grasp0_points}")
    
    if not os.path.exists(grasp0_colors):
        print(f"❌ 源文件不存在: {grasp0_colors}")
        files_ok = False
    else:
        print(f"✓ 找到: {grasp0_colors}")
    
    if not os.path.exists(grasp_points):
        print(f"❌ 源文件不存在: {grasp_points}")
        files_ok = False
    else:
        print(f"✓ 找到: {grasp_points}")
    
    if not os.path.exists(grasp_colors):
        print(f"❌ 源文件不存在: {grasp_colors}")
        files_ok = False
    else:
        print(f"✓ 找到: {grasp_colors}")
    
    if not files_ok:
        print("\n❌ 源文件检查失败，退出")
        return
    
    # 获取目标demo列表
    demo_dirs = []
    if os.path.exists(target_base):
        for item in os.listdir(target_base):
            if item.startswith('demo_') and os.path.isdir(os.path.join(target_base, item)):
                demo_num = item.replace('demo_', '')
                if demo_num.isdigit():
                    demo_dirs.append(int(demo_num))
    
    demo_dirs.sort()
    
    if not demo_dirs:
        print(f"\n⚠️  未找到任何demo目录在: {target_base}")
        print("提示: 请先运行 record_poses.py 创建demo目录")
        return
    
    print(f"\n找到 {len(demo_dirs)} 个demo目录: {demo_dirs}")
    print(f"\n将要执行的操作:")
    print(f"  源 (step_0): {grasp0_dir}")
    print(f"  源 (step_1): {grasp_dir}")
    print(f"  目标目录: {target_base}")
    
    for demo_num in demo_dirs:
        print(f"  - demo_{demo_num}/step_0/grasp_pcd ← grasp0/cropped")
        print(f"  - demo_{demo_num}/step_1/grasp_pcd ← grasp/cropped")
    
    confirm = input("\n确认开始复制? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ 取消操作")
        return
    
    # 复制文件
    success_count = 0
    error_count = 0
    
    for demo_num in demo_dirs:
        print(f"\n{'='*60}")
        print(f"处理 demo_{demo_num}")
        print(f"{'='*60}")
        
        # Step 0: grasp0 -> step_0/grasp_pcd
        print(f"\n  Step 0 (grasp_pcd):")
        step0_grasp_dir = os.path.join(target_base, f'demo_{demo_num}', 'step_0', 'grasp_pcd')
        os.makedirs(step0_grasp_dir, exist_ok=True)
        
        try:
            # 复制points.pt
            shutil.copy2(grasp0_points, os.path.join(step0_grasp_dir, 'points.pt'))
            print(f"    ✓ 复制 points.pt (from grasp0)")
            
            # 复制colors.pt
            shutil.copy2(grasp0_colors, os.path.join(step0_grasp_dir, 'colors.pt'))
            print(f"    ✓ 复制 colors.pt (from grasp0)")
            
            # 创建metadata.yaml
            create_grasp_pcd_metadata(step0_grasp_dir)
            
        except Exception as e:
            print(f"    ❌ Step 0 复制失败: {e}")
            error_count += 1
            continue
        
        # Step 1: grasp -> step_1/grasp_pcd
        print(f"\n  Step 1 (grasp_pcd):")
        step1_grasp_dir = os.path.join(target_base, f'demo_{demo_num}', 'step_1', 'grasp_pcd')
        os.makedirs(step1_grasp_dir, exist_ok=True)
        
        try:
            # 复制points.pt
            shutil.copy2(grasp_points, os.path.join(step1_grasp_dir, 'points.pt'))
            print(f"    ✓ 复制 points.pt (from grasp)")
            
            # 复制colors.pt
            shutil.copy2(grasp_colors, os.path.join(step1_grasp_dir, 'colors.pt'))
            print(f"    ✓ 复制 colors.pt (from grasp)")
            
            # 创建metadata.yaml
            create_grasp_pcd_metadata(step1_grasp_dir)
            
            success_count += 1
            
        except Exception as e:
            print(f"    ❌ Step 1 复制失败: {e}")
            error_count += 1
    
    # 总结
    print(f"\n{'='*60}")
    print(f"📊 复制完成")
    print(f"{'='*60}")
    print(f"✅ 成功: {success_count} 个demo")
    print(f"❌ 失败: {error_count} 个demo")
    print(f"📁 目标目录: {target_base}")
    print(f"\n说明:")
    print(f"  - step_0/grasp_pcd: 使用 grasp0/cropped (抓取时的物体点云)")
    print(f"  - step_1/grasp_pcd: 使用 grasp/cropped (放置时的物体点云)")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="复制grasp_pcd文件到demo目录")
    parser.add_argument("--source", 
                       default="/home/hkcrc/diffusion_edfs/diffusion_edf/dataset/1030_baselink_pt",
                       help="源目录路径")
    parser.add_argument("--target",
                       default="/home/hkcrc/diffusion_edfs/diffusion_edf/demo/rebar_grasping/data",
                       help="目标demo目录路径")
    
    args = parser.parse_args()
    
    copy_grasp_pcd(args.source, args.target)
