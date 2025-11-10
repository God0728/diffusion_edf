import argparse
import open3d as o3d

def main():
    parser = argparse.ArgumentParser(description="读取 PLY 点云并可视化其坐标系（X红, Y绿, Z蓝）")
    parser.add_argument("--input", type=str, required=True, help="PLY 点云文件路径")
    parser.add_argument("--axis-size", type=float, default=0.2, help="坐标系轴长度")
    args = parser.parse_args()

    # 读取点云
    pcd = o3d.io.read_point_cloud(args.input)
    if pcd.is_empty():
        raise RuntimeError(f"点云为空或无法读取: {args.input}")

    # 原点坐标系（X红, Y绿, Z蓝）
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.axis_size, origin=[0, 0, 0])

    print("可视化说明：")
    print("  - 直接按文件坐标显示（未做任何变换）")
    print("  - 坐标轴颜色：X=红，Y=绿，Z=蓝")
    print(f"  - 点数：{len(pcd.points)}")

    # 可视化
    o3d.visualization.draw_geometries(
        [pcd, axis],
        window_name="PLY 坐标系可视化 (X=红, Y=绿, Z=蓝)",
        width=1280,
        height=720,
        left=100,
        top=100
    )

if __name__ == "__main__":
    main()