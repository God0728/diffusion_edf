#!/usr/bin/env python3
import os
import yaml
from pathlib import Path

def create_step_metadata(step_dir, step_idx):
    """创建 step 级别的 metadata.yaml"""
    metadata = {
        '__type__': 'TargetPoseDemo',
        'name': 'pick' if step_idx == 0 else 'place'
    }
    
    metadata_file = step_dir / 'metadata.yaml'
    with open(metadata_file, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
    print(f"  ✓ 创建: {metadata_file}")

def fix_demo_metadata(demo_dir):
    """修复单个demo的metadata"""
    demo_name = demo_dir.name
    print(f"\n处理 {demo_name}...")
    
    # step_0
    step_0_dir = demo_dir / 'step_0'
    if step_0_dir.exists():
        metadata_file = step_0_dir / 'metadata.yaml'
        if not metadata_file.exists():
            create_step_metadata(step_0_dir, 0)
        else:
            print(f"  - step_0/metadata.yaml 已存在")
    
    # step_1
    step_1_dir = demo_dir / 'step_1'
    if step_1_dir.exists():
        metadata_file = step_1_dir / 'metadata.yaml'
        if not metadata_file.exists():
            create_step_metadata(step_1_dir, 1)
        else:
            print(f"  - step_1/metadata.yaml 已存在")

def main():
    dataset_root = Path('/home/hkcrc/diffusion_edfs/diffusion_edf/demo/rebar_grasping/data')
    
    print("=" * 60)
    print("修复 rebar_grasping 数据集 metadata")
    print("=" * 60)
    
    if not dataset_root.exists():
        print(f"❌ 数据集目录不存在: {dataset_root}")
        return
    
    demo_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir() and d.name.startswith('demo_')])
    print(f"\n找到 {len(demo_dirs)} 个demo目录")
    
    for demo_dir in demo_dirs:
        fix_demo_metadata(demo_dir)
    
    print("\n" + "=" * 60)
    print("✅ 修复完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
