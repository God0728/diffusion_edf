#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快速读取并打印 poses.pt 文件内容"""

import torch
from pathlib import Path

def read_poses(filepath: str):
    """读取并解析 poses.pt 文件
    
    Args:
        filepath: poses.pt 文件路径
    """
    path = Path(filepath)
    
    if not path.exists():
        print(f"❌ 文件不存在: {filepath}")
        return
    
    # 加载 tensor
    poses = torch.load(path)
    
    print(f"📄 文件路径: {filepath}")
    print(f"📊 Tensor shape: {poses.shape}")
    print(f"📊 Tensor dtype: {poses.dtype}")
    print(f"\n原始数据 (1x7):")
    print(poses)
    
    # 解析为四元数和位置
    if poses.shape == (1, 7):
        quat_wxyz = poses[0, :4].numpy()
        pos_xyz = poses[0, 4:].numpy()
        
        print(f"\n🔄 四元数 (w,x,y,z): {quat_wxyz}")
        print(f"📍 位置 (x,y,z): {pos_xyz}")
        
        # 转换为 xyzw 格式（UR 机器人使用）
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        print(f"\n🔄 四元数 (x,y,z,w) [UR格式]: {quat_xyzw}")
    else:
        print(f"\n⚠️  警告: 期望 shape (1, 7), 实际为 {poses.shape}")


if __name__ == "__main__":
    filepath = "/home/hkcrc/diffusion_edfs/diffusion_edf/demo/rebar_grasping1113/data/demo_0/step_0/target_poses/poses.pt"
    read_poses(filepath)