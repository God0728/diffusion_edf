import argparse
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from numpy.linalg import inv

# === 1) 相机相对 EE 的外参（ee←cam） ===
# cam2ee = np.array([
#     [0.8031, -0.5953, -0.0242, -0.0937],
#     [0.5945,  0.8034, -0.0350, -0.0752],
#     [0.0402,  0.0137,  0.9991,  0.0400],
#     [0,       0,       0,       1.0000]
# ])

# 你提供的 base←cam 变换（用于点云变换）
T_cam2base = np.array([
    [ 0.786366,   0.615936,   0.0474553, -0.0901214],
    [-0.617143,   0.786694,   0.0157389, -0.0694668],
    [-0.0276386, -0.0416632,  0.998749,   0.00894665],
    [ 0.0,        0.0,        0.0,        1.0],
], dtype=float)

# === 2) EE 在基坐标系下的位姿（位置 + RPY，弧度）===
pos_ee_in_base = np.array([-0.3409013839894124, 0.08709154364428232, 0.676218455849546])
rpy_ee_in_base = np.array([2.702, 1.526, 0.168])  # 采用 XYZ 顺序的 RPY（roll,pitch,yaw）

def make_T(Rm, t):
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = t
    print(T)
    return T

def main():
    parser = argparse.ArgumentParser(description="用给定的 base←cam 矩阵将相机系点云变到 base，并可视化")
    parser.add_argument("--input", type=str, default="/home/hkcrc/DCIM/rs1105_3/cloud.ply")
    parser.add_argument("--save", type=str, default="", help="可选：保存变换后的PLY路径")
    parser.add_argument("--axis-size", type=float, default=0.2, help="坐标轴长度")
    args = parser.parse_args()

    # 读取点云（相机坐标系）
    pcd_cam = o3d.io.read_point_cloud(args.input)
    if pcd_cam.is_empty():
        raise RuntimeError(f"点云为空或无法读取: {args.input}")
    points_cam = np.asarray(pcd_cam.points)
    colors = np.asarray(pcd_cam.colors) if pcd_cam.has_colors() else None

    # 调试：相机原点在 base 中的位置（应等于 T_cam2base 的平移列）
    cam_origin_in_base = (T_cam2base @ np.array([0, 0, 0, 1.0]))[:3]
    print("✓ 使用 base←cam 矩阵直接变换点云")
    print("  T_cam2base =\n", T_cam2base)
    print("  相机原点在 base 中位置 =", cam_origin_in_base)

    # 将点云从相机系变换到 base 系：P_base = T_cam2base @ P_cam
    ones = np.ones((points_cam.shape[0], 1))
    points_cam_h = np.hstack([points_cam, ones])
    points_base = (T_cam2base @ points_cam_h.T).T[:, :3]

    # 构建点云
    pcd_base = o3d.geometry.PointCloud()
    pcd_base.points = o3d.utility.Vector3dVector(points_base)
    if colors is not None:
        pcd_base.colors = o3d.utility.Vector3dVector(colors)

    if args.save:
        o3d.io.write_point_cloud(args.save, pcd_base)
        print(f"✓ 已保存: {args.save}")

    # 可视化：base 原点坐标系 + 相机坐标系（用同一 T_cam2base 显示）
    axis_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.axis_size, origin=[0, 0, 0])
    axis_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.axis_size * 0.7)
    axis_cam.transform(T_cam2base)

    o3d.visualization.draw_geometries(
        [pcd_base, axis_base, axis_cam],
        window_name="点云 (base) 与 base/cam 坐标系",
        width=1280, height=720, left=100, top=100
    )

if __name__ == "__main__":
    main()