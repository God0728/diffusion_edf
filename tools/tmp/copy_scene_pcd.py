#!/usr/bin/env python3
"""
复制scene_pcd文件到demo目录
从 dataset/1030_baselink_pt/demoN/cropped/ 
到 demo/rebar_grasping/data/demo_N/step_0/scene_pcd/ 和 step_1/scene_pcd/
"""

import os
import shutil
import yaml

def create_scene_pcd_metadata(output_path):
    """创建scene_pcd的metadata.yaml"""
    metadata = {
        '__type__': 'PointCloud',
        'name': '',
        'unit_length': '1 [m]'
    }
    
    metadata_file = os.path.join(output_path, 'metadata.yaml')
    with open(metadata_file, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    print(f"  ✓ 创建 metadata.yaml")


def copy_scene_pcd(source_base, target_base, demo_count=None):
    """
    复制scene_pcd文件
    
    Args:
        source_base: 源目录基路径 (dataset/1030_baselink_pt)
        target_base: 目标目录基路径 (demo/rebar_grasping/data)
        demo_count: 要复制的demo数量，None则复制所有
    """
    print("="*60)
    print("📋 Scene PCD 文件复制工具")
    print("="*60)
    
    # 检查源目录
    if not os.path.exists(source_base):
        print(f"❌ 源目录不存在: {source_base}")
        return
    
    # 获取所有demo目录
    demo_dirs = []
    for item in os.listdir(source_base):
        if item.startswith('demo') and os.path.isdir(os.path.join(source_base, item)):
            # 提取demo编号
            demo_num = item.replace('demo', '')
            if demo_num.isdigit():
                demo_dirs.append((int(demo_num), item))
    
    demo_dirs.sort()  # 按编号排序
    
    if demo_count is not None:
        demo_dirs = demo_dirs[:demo_count]
    
    print(f"\n找到 {len(demo_dirs)} 个demo目录")
    print(f"源目录: {source_base}")
    print(f"目标目录: {target_base}")
    
    # 确认操作
    print(f"\n将要复制以下demo:")
    for num, name in demo_dirs:
        print(f"  - {name} → demo_{num}")
    
    confirm = input("\n确认开始复制? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ 取消操作")
        return
    
    # 复制文件
    success_count = 0
    error_count = 0
    
    for demo_num, demo_name in demo_dirs:
        print(f"\n{'='*60}")
        print(f"处理 {demo_name} → demo_{demo_num}")
        print(f"{'='*60}")
        
        # 源文件路径
        source_dir = os.path.join(source_base, demo_name, 'cropped')
        source_points = os.path.join(source_dir, 'points.pt')
        source_colors = os.path.join(source_dir, 'colors.pt')
        
        # 检查源文件
        if not os.path.exists(source_points):
            print(f"  ❌ 源文件不存在: {source_points}")
            error_count += 1
            continue
        
        if not os.path.exists(source_colors):
            print(f"  ❌ 源文件不存在: {source_colors}")
            error_count += 1
            continue
        
        # 复制到 step_0 和 step_1
        for step in [0, 1]:
            print(f"\n  Step {step}:")
            
            # 目标目录
            target_dir = os.path.join(target_base, f'demo_{demo_num}', f'step_{step}', 'scene_pcd')
            os.makedirs(target_dir, exist_ok=True)
            
            # 目标文件路径
            target_points = os.path.join(target_dir, 'points.pt')
            target_colors = os.path.join(target_dir, 'colors.pt')
            
            try:
                # 复制points.pt
                shutil.copy2(source_points, target_points)
                print(f"    ✓ 复制 points.pt")
                
                # 复制colors.pt
                shutil.copy2(source_colors, target_colors)
                print(f"    ✓ 复制 colors.pt")
                
                # 创建metadata.yaml
                create_scene_pcd_metadata(target_dir)
                
                if step == 1:
                    success_count += 1
                    
            except Exception as e:
                print(f"    ❌ 复制失败: {e}")
                if step == 1:
                    error_count += 1
    
    # 总结
    print(f"\n{'='*60}")
    print(f"📊 复制完成")
    print(f"{'='*60}")
    print(f"✅ 成功: {success_count} 个demo")
    print(f"❌ 失败: {error_count} 个demo")
    print(f"📁 目标目录: {target_base}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="复制scene_pcd文件到demo目录")
    parser.add_argument("--source", 
                       default="/home/hkcrc/diffusion_edfs/diffusion_edf/dataset/1030_baselink_pt",
                       help="源目录路径")
    parser.add_argument("--target",
                       default="/home/hkcrc/diffusion_edfs/diffusion_edf/demo/rebar_grasping/data",
                       help="目标demo目录路径")
    parser.add_argument("--count", type=int, default=None,
                       help="要复制的demo数量 (默认复制所有)")
    
    args = parser.parse_args()
    
    copy_scene_pcd(args.source, args.target, args.count)
