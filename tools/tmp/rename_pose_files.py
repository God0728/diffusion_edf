#!/usr/bin/env python3
"""
将所有 pose.pt 重命名为 poses.pt
"""
from pathlib import Path

dataset_root = Path('/home/hkcrc/diffusion_edfs/diffusion_edf/demo/rebar_grasping/data')

print("=" * 60)
print("重命名 pose.pt -> poses.pt")
print("=" * 60)

# 查找所有 pose.pt 文件
pose_files = list(dataset_root.rglob('pose.pt'))

print(f"\n找到 {len(pose_files)} 个 pose.pt 文件:")
for f in pose_files:
    print(f"  {f.relative_to(dataset_root.parent)}")

if not pose_files:
    print("\n✓ 没有需要重命名的文件")
    exit(0)

input("\n按 Enter 键开始重命名...")

# 重命名
renamed_count = 0
for old_path in pose_files:
    new_path = old_path.parent / 'poses.pt'
    try:
        old_path.rename(new_path)
        print(f"✓ {old_path.parent.name}/pose.pt -> poses.pt")
        renamed_count += 1
    except Exception as e:
        print(f"✗ 失败: {old_path} - {e}")

print("\n" + "=" * 60)
print(f"✅ 重命名完成！共 {renamed_count} 个文件")
print("=" * 60)
