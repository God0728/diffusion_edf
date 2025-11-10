import argparse
import json
import os
from pathlib import Path

import numpy as np

# 依赖：torch, open3d, matplotlib(备用)
import torch
import open3d as o3d


def load_pt_tensor(path: Path):
    obj = torch.load(str(path), map_location="cpu")
    if isinstance(obj, dict):
        # 兼容常见格式：{'points': tensor} 或 {'xyz': tensor} 等
        for k in ("points", "point", "xyz", "data"):
            if k in obj:
                obj = obj[k]
                break
    arr = np.asarray(obj, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"文件 {path} 不是形状为 [N,3] 的张量")
    return arr


def auto_find_file(dirpath: Path, candidates):
    for name in candidates:
        p = dirpath / name
        if p.exists():
            return p
    return None


def make_sphere(center, radius=0.005, color=(1.0, 0.0, 0.0)):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh.translate(center)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh


def main():
    parser = argparse.ArgumentParser(
        description="读取 points.pt 和 colors.pt，并标记两个关键点坐标"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="/home/hkcrc/diffusion_edfs/diffusion_edf/demo/panda_mug_on_hanger/data/demo_0/step_0/grasp_pcd",
        help="包含 points.pt 和 colors.pt 的目录",
    )
    parser.add_argument("--points", type=str, default="", help="自定义 points.pt 路径")
    parser.add_argument("--colors", type=str, default="", help="自定义 colors.pt 路径")
    parser.add_argument(
        "--kp1",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 10.5],
        help="关键点1（单位：厘米）",
    )
    parser.add_argument(
        "--kp2",
        type=float,
        nargs=3,
        default=[-0.5, -0.5, 10.5],
        help="关键点2（单位：厘米）",
    )
    parser.add_argument(
        "--axis-size", type=float, default=0.2, help="坐标轴大小（米）"
    )
    parser.add_argument(
        "--kp-radius", type=float, default=0.01, help="关键点球体半径（米）"
    )
    args = parser.parse_args()

    dirpath = Path(args.dir)

    # 定位 points / colors 文件
    points_path = Path(args.points) if args.points else (
        auto_find_file(dirpath, ["points.pt", "point.pt", "pts.pt", "xyz.pt"])
    )
    colors_path = Path(args.colors) if args.colors else (
        auto_find_file(dirpath, ["colors.pt", "color.pt", "rgb.pt"])
    )

    if not points_path or not points_path.exists():
        raise FileNotFoundError(f"未找到 points 文件，请检查：{points_path or dirpath}")
    if not colors_path or not colors_path.exists():
        print(f"[警告] 未找到 colors 文件，将不使用颜色：{colors_path or dirpath}")
        colors_path = None

    # 读取
    points = load_pt_tensor(points_path)
    colors = load_pt_tensor(colors_path) if colors_path else None
    if colors is not None and colors.shape[0] != points.shape[0]:
        print(f"[警告] 颜色数量({colors.shape[0]})与点数量({points.shape[0]})不一致，忽略颜色")
        colors = None

    # 构建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        # 若颜色不是 [0,1]，尝试裁剪
        c = np.clip(colors.astype(np.float64), 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(c)

    # 关键点：厘米 → 米
    kp1_m = (np.asarray(args.kp1, dtype=np.float64) / 100.0).tolist()
    kp2_m = (np.asarray(args.kp2, dtype=np.float64) / 100.0).tolist()

    # 可视化几何体
    geoms = []
    geoms.append(pcd)

    # 坐标系（base/文件坐标原点）
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=args.axis_size, origin=[0, 0, 0]
    )
    #geoms.append(axis)

    # 两个关键点：球体标注
    s1 = make_sphere(kp1_m, radius=0.005, color=(1.0, 0.0, 0.0))  # 红
    s2 = make_sphere(kp2_m, radius=0.005, color=(0.0, 1.0, 0.0))  # 绿
    geoms.extend([s1, s2])

    print("可视化说明：")
    print(f"- 点云: {points.shape[0]} 点")
    print(f"- 关键点1(红，cm): {args.kp1}  -> (m): {kp1_m}")
    print(f"- 关键点2(绿，cm): {args.kp2}  -> (m): {kp2_m}")

    # Open3D 可视化；若 OpenGL 不可用则提示
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="点云与关键点", width=1280, height=720, left=100, top=100)
        for g in geoms:
            vis.add_geometry(g)
        vis.get_render_option().background_color = np.array([0.05, 0.05, 0.05], dtype=np.float32)  # 一行代码：深色背景
        vis.run()
        vis.destroy_window()
    except Exception as e:
        print("[警告] Open3D 可视化失败，可能是 OpenGL/GLX 环境缺失。")
        print(e)
        print("可安装显卡驱动或在本地有显示环境下运行；或改用 Matplotlib 备用方案。")


if __name__ == "__main__":
    main()