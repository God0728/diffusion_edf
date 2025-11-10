import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as R
import rtde_receive
from pathlib import Path
import argparse

# cam2ee 变换矩阵（从标定文件）
cam2ee = np.array([
    [0.8031, -0.5953, -0.0242, -0.0937],
    [0.5945,  0.8034, -0.0350, -0.0752],
    [0.0402,  0.0137,  0.9991,  0.0400],
    [0,       0,       0,       1.0000]
])


def pose_to_matrix_xyzrxryrz(xyzrxryrz):
    """将 UR 的 xyz rx ry rz 姿态转为 4x4 齐次矩阵。
    xyzrxryrz: [x, y, z, rx, ry, rz], 位置单位 m，旋转为轴角(弧度)
    """
    x, y, z, rx, ry, rz = xyzrxryrz
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec([rx, ry, rz]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def get_current_ee_to_base_transform(robot_ip="192.168.56.101", tcp_offset_pose=None):
    """
    从机器人获取当前的 TCP 位姿，并（可选）根据已配置的 TCP 偏置换算出法兰(EE)位姿。

    Args:
        robot_ip: 机器人 IP
        tcp_offset_pose: 法兰→TCP 的相对位姿 [x, y, z, rx, ry, rz]（单位 m, 弧度）。
                         若提供，将用 base→TCP 与其求逆相乘得到 base→法兰。

    Returns:
        T_base_tcp:  base←tcp 4x4
        T_base_ee:   base←ee(法兰) 4x4（若未提供偏置，则等于 T_base_tcp）
    """
    print(f"连接机器人: {robot_ip}")
    rtde = rtde_receive.RTDEReceiveInterface(robot_ip)
    
    # 获取当前TCP姿态
    tcp_pose = rtde.getActualTCPPose()
    if tcp_pose is None:
        raise ValueError("无法获取TCP姿态")
    
    position = tcp_pose[:3]
    rotvec = tcp_pose[3:]
    quaternion = R.from_rotvec(rotvec).as_quat()
    
    rtde.disconnect()
    
    print(f"✓ 获取当前TCP姿态:")
    print(f"  位置: {position}")
    print(f"  四元数: {quaternion}")

    # base←tcp
    T_base_tcp = np.eye(4)
    T_base_tcp[:3, :3] = R.from_rotvec(rotvec).as_matrix()
    T_base_tcp[:3, 3] = position

    if tcp_offset_pose is not None:
        # 法兰→TCP
        T_ee_tcp = pose_to_matrix_xyzrxryrz(tcp_offset_pose)
        # base←ee = base←tcp · tcp←ee = base←tcp · (ee←tcp)^-1 = base←tcp · (法兰→TCP)^-1
        T_base_ee = T_base_tcp @ np.linalg.inv(T_ee_tcp)
    else:
        print("⚠ 未提供 TCP 偏置 (法兰→TCP)，默认 EE=TCP。若标定使用法兰坐标，请提供 --tcp-offset 以修正 EE 位置。")
        T_base_ee = T_base_tcp.copy()

    return T_base_tcp, T_base_ee


def get_cam_to_base_transform(robot_ip="192.168.56.101", tcp_offset_pose=None):
    """
    计算相机在base_link坐标系下的变换矩阵
    返回:
        T_cam_base: 相机在 base 下的位姿 (4x4)
        T_ee_base:  末端在 base 下的位姿 (4x4)
    """
    # 1. 获取 base←tcp 与 base←ee(法兰) 变换
    T_base_tcp, T_base_ee = get_current_ee_to_base_transform(robot_ip, tcp_offset_pose)
    
    # 2. 计算 cam 在 base 下的位姿: base←cam = base←ee · ee←cam
    T_cam_base = T_base_ee @ cam2ee
    
    cam_position = T_cam_base[:3, 3]
    cam_rotation = R.from_matrix(T_cam_base[:3, :3])
    cam_quaternion = cam_rotation.as_quat()
    
    print(f"\n✓ 相机在base_link的变换:")
    print(f"  位置: {cam_position}")
    print(f"  四元数: {cam_quaternion}")
    # 同时打印 EE 在 base 下的位姿，便于对照
    ee_quat_print = R.from_matrix(T_base_ee[:3, :3]).as_quat()
    print(f"\n✓ 末端(EE)在base_link的变换:")
    print(f"  位置: {T_base_ee[:3, 3]}")
    print(f"  四元数: {ee_quat_print}")
    
    return T_cam_base, T_base_ee


def transform_pointcloud_cam2base(input_ply, T_cam_base):
    """
    将点云从相机坐标系变换到base_link坐标系
    
    Args:
        input_ply: 输入PLY文件路径
        T_cam_base: 相机在base_link中的位姿 (4x4)
    
    Returns:
        points_base: 变换后的点云 (N, 3)
        colors: 点云颜色 (N, 3)
    """
    print(f"\n加载点云: {input_ply}")
    pcd = o3d.io.read_point_cloud(str(input_ply))
    
    points_cam = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    
    print(f"✓ 原始点云: {len(points_cam)} 点")
    print(f"  范围: {points_cam.min(axis=0)} ~ {points_cam.max(axis=0)}")
    
    # ⭐ 关键：点的变换需要用逆矩阵！
    # T_cam_base: 相机坐标系在base中的位姿
    # T_base_cam: 用于变换点 (从cam到base)
    
    print(f"\n应用变换: 相机坐标系 → base_link")
    print(f"  T_cam_base (相机在base中的位姿):")
    print(T_cam_base)
    print(f"\n  T_base_cam (点变换矩阵, 逆矩阵):")
    
    # 转换为齐次坐标
    points_homo = np.hstack([points_cam, np.ones((points_cam.shape[0], 1))])
    
    # 应用变换
    points_base_homo = (T_cam_base @ points_homo.T).T
    points_base = points_base_homo[:, :3]
    
    print(f"\n✓ 变换后点云:")
    print(f"  范围: {points_base.min(axis=0)} ~ {points_base.max(axis=0)}")
    
    return points_base, colors


def save_as_pytorch(points, colors, output_pt):
    """
    保存为PyTorch .pt格式
    """
    output_pt = Path(output_pt)
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    
    points_tensor = torch.from_numpy(points.astype(np.float32))
    torch.save(points_tensor, output_pt)
    
    print(f"\n✓ 保存点坐标: {output_pt}")
    
    if colors is not None:
        colors_pt = output_pt.parent / 'colors.pt'
        colors_tensor = torch.from_numpy(colors.astype(np.float32))
        torch.save(colors_tensor, colors_pt)
        print(f"✓ 保存点颜色: {colors_pt}")


def create_coordinate_frame(size=0.1, origin=[0, 0, 0]):
    """
    创建坐标系可视化（XYZ轴）
    """
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=size, origin=origin)
    return mesh_frame


def visualize_result(points_base, colors, T_cam_base, T_ee_base):
    """
    可视化变换后的点云和坐标系
    """
    print(f"\n可视化结果...")
    
    # 1. 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_base)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 2. 创建base_link坐标系（原点）
    base_frame = create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    
    # 3. 创建相机坐标系（在base_link中的位置）
    cam_position = T_cam_base[:3, 3]
    cam_frame = create_coordinate_frame(size=0.1, origin=cam_position)
    # 应用旋转
    cam_frame.rotate(T_cam_base[:3, :3], center=cam_position)
    
    # 4. 创建末端(EE)坐标系（在base_link中的位置）
    ee_position = T_ee_base[:3, 3]
    ee_frame = create_coordinate_frame(size=0.12, origin=ee_position)
    ee_frame.rotate(T_ee_base[:3, :3], center=ee_position)
    
    # 4. 显示
    print(f"\n可视化说明:")
    print(f"  🔴 大坐标系 (0.2m): base_link 原点")
    print(f"  🔵 小坐标系 (0.1m): 相机位置 {cam_position}")
    print(f"  🟢 EE 坐标系 (0.12m): 末端位置 {ee_position}")
    print(f"  ⚪ 点云: 变换后的点云（base_link坐标系）")
    
    o3d.visualization.draw_geometries(
        [pcd, base_frame, cam_frame, ee_frame],
        window_name="点云变换结果 (base_link坐标系)",
        width=1920,
        height=1080,
        left=50,
        top=50
    )


def main():
    parser = argparse.ArgumentParser(description="将PLY点云从相机坐标系变换到base_link")
    parser.add_argument('--input', type=str, default='/home/hkcrc/DCIM/rs1105_3/cloud.ply')
    parser.add_argument('--output', type=str, default='points.pt')
    parser.add_argument('--robot_ip', type=str, default='192.168.56.101', help='机器人IP')
    parser.add_argument('--no-viz', action='store_true', help='不显示可视化')
    parser.add_argument('--tcp-offset', type=float, nargs=6, default=None,
                        metavar=('x','y','z','rx','ry','rz'),
                        help='法兰→TCP 的相对位姿 (米, 弧度)。用于从 TCP 位姿还原法兰(EE)位姿。')
    
    args = parser.parse_args()
    
    print("="*80)
    print("点云坐标变换: 相机坐标系 → base_link")
    print("="*80)
    
    # 1. 获取cam2base与ee2base变换（若提供 tcp 偏置则用于还原法兰位姿）
    T_cam_base, T_ee_base = get_cam_to_base_transform(args.robot_ip, args.tcp_offset)
    
    # 2. 变换点云
    points_base, colors = transform_pointcloud_cam2base(args.input, T_cam_base)
    
    # 3. 保存为.pt
    save_as_pytorch(points_base, colors, args.output)
    
    # 4. 可视化
    if not args.no_viz:
        visualize_result(points_base, colors, T_cam_base, T_ee_base)
    
    print("\n" + "="*80)
    print("✅ 完成！")
    print("="*80)


if __name__ == "__main__":
    main()