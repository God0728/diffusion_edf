#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# filepath: /home/hkcrc/diffusion_edfs/diffusion_edf/edf_interface/tools/generate_demo.py
"""
遍历 /home/hkcrc/diffusion_edfs/diffusion_edf/edf_interface/run_sessions/scene/ 下的目录
按顺序把每个子目录里的 /scene_pcd/ 目录复制到 
/home/hkcrc/diffusion_edfs/diffusion_edf/demo/rebar_grasping1113/data/demo_{demo_idx}/step_0 和 step_1 下
"""
import shutil
from pathlib import Path
from loguru import logger


def copy_scene_pcd_to_demo(
    source_root: str = "/home/hkcrc/diffusion_edfs/diffusion_edf/edf_interface/run_sessions/scene",
    target_root: str = "/home/hkcrc/diffusion_edfs/diffusion_edf/demo/rebar_grasping1113/data",
    steps_per_demo: list = [0, 1],  # 每个 demo 复制到哪些 step（step_0 和 step_1）
):
    """
    将 scene_pcd 目录组织为演示数据结构
    每个源 scene_pcd 会被复制到同一个 demo 的多个 step 中
    
    Args:
        source_root: 源目录（包含按时间命名的子目录）
        target_root: 目标 demo 数据目录
        steps_per_demo: 每个 demo 复制到哪些 step，默认 [0, 1]
    """
    source_path = Path(source_root)
    target_path = Path(target_root)
    
    # 获取所有子目录并按名称排序（时间戳命名会自动按时间排序）
    subdirs = sorted([d for d in source_path.iterdir() if d.is_dir()])
    
    if not subdirs:
        logger.error(f"No subdirectories found in {source_root}")
        return
    
    logger.info(f"Found {len(subdirs)} session directories")
    logger.info(f"Each scene_pcd will be copied to steps: {steps_per_demo}")
    
    demo_idx = 0
    
    for i, subdir in enumerate(subdirs):
        scene_pcd_dir = subdir / "scene_pcd"
        
        # 检查 scene_pcd 目录是否存在
        if not scene_pcd_dir.exists():
            logger.warning(f"[{i+1}/{len(subdirs)}] No scene_pcd in {subdir.name}, skipping")
            continue
        
        # 复制到多个 step
        for step_idx in steps_per_demo:
            target_demo_dir = target_path / f"demo_{demo_idx}" / f"step_{step_idx}"
            target_scene_pcd_dir = target_demo_dir / "scene_pcd"
            
            # 创建目标目录
            target_demo_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制 scene_pcd 目录
            if target_scene_pcd_dir.exists():
                logger.warning(f"Target exists, removing: {target_scene_pcd_dir}")
                shutil.rmtree(target_scene_pcd_dir)
            
            shutil.copytree(scene_pcd_dir, target_scene_pcd_dir)
            logger.success(
                f"[{i+1}/{len(subdirs)}] Copied: {subdir.name} -> demo_{demo_idx}/step_{step_idx}/scene_pcd"
            )
        
        # 每个源目录对应一个 demo
        demo_idx += 1
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Copy completed!")
    logger.info(f"Total demos: {demo_idx}")
    logger.info(f"Steps per demo: {steps_per_demo}")
    logger.info(f"Target directory: {target_path}")
    logger.info(f"{'='*50}")


def preview_copy_structure(source_root: str, target_root: str, steps_per_demo: list):
    """预览复制结构（dry run）"""
    source_path = Path(source_root)
    target_path = Path(target_root)
    
    subdirs = sorted([d for d in source_path.iterdir() if d.is_dir()])
    
    logger.info(f"\nPreview of copy operations:")
    logger.info(f"{'='*70}")
    
    demo_idx = 0
    
    for i, subdir in enumerate(subdirs):
        scene_pcd_dir = subdir / "scene_pcd"
        
        if not scene_pcd_dir.exists():
            logger.warning(f"[SKIP] {subdir.name} (no scene_pcd)")
            continue
        
        logger.info(f"[{i+1:2d}] {subdir.name}")
        
        for step_idx in steps_per_demo:
            target_dir = target_path / f"demo_{demo_idx}" / f"step_{step_idx}" / "scene_pcd"
            logger.info(f"     -> demo_{demo_idx}/step_{step_idx}/scene_pcd")
        
        demo_idx += 1
        logger.info("")
    
    logger.info(f"{'='*70}")
    logger.info(f"Total: {demo_idx} demos, each with {len(steps_per_demo)} steps")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Organize scene_pcd directories into demo structure"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="/home/hkcrc/diffusion_edfs/diffusion_edf/edf_interface/run_sessions/scene",
        help="Source directory containing session subdirectories"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="/home/hkcrc/diffusion_edfs/diffusion_edf/demo/rebar_grasping1113/data",
        help="Target demo data directory"
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="0,1",
        help="Steps to copy to (comma-separated), e.g., '0,1' or '0,1,2'"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run (do not copy, just show what would be done)"
    )
    
    args = parser.parse_args()
    
    # 解析 steps 参数
    steps_per_demo = [int(s.strip()) for s in args.steps.split(",")]
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be copied")
        preview_copy_structure(args.source, args.target, steps_per_demo)
    else:
        copy_scene_pcd_to_demo(
            source_root=args.source,
            target_root=args.target,
            steps_per_demo=steps_per_demo
        )


if __name__ == "__main__":
    main()