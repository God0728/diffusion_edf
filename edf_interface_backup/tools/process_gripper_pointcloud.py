#!/usr/bin/env python3
from pathlib import Path
import torch
from typing import Tuple
import open3d as o3d
import argparse

from edf_interface.data import PointCloud, SE3
from edf_interface.data.preprocess import downsample
from edf_interface.modules import RobotInterface as RI
from edf_interface.modules import TransformManager as TM

def concat(pcd1: PointCloud, pcd2: PointCloud) -> PointCloud:

    # 构造绕Z轴180度旋转的变换
    # 四元数 [0, 0, sin(90°), cos(90°)] = [0, 0, 1, 0] 表示绕Z轴180度
    T_flip = SE3(torch.tensor([0., 0., 1., 0., 0., 0., 0.], dtype=torch.float32))
    
    pcd2_aligned = pcd2.transformed(T_flip, squeeze=True)
    
    merged = PointCloud(
        points=torch.cat([pcd1.points, pcd2_aligned.points], dim=0),
        colors=torch.cat([pcd1.colors, pcd2_aligned.colors], dim=0) if pcd1.colors is not None else None
    )
    
    return downsample(merged, voxel_size=0.001) 


def process_gripper_pointcloud(
    ply1_path: str,
    ply2_path: str,
    gripper_bbox: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    T_cam_ee: SE3,
    output_dir: str,
) -> PointCloud:

    pcd1 = o3d.io.read_point_cloud(ply1_path)
    pcd1 = PointCloud.from_o3d(pcd1)
    pcd2 = o3d.io.read_point_cloud(ply2_path)
    pcd2 = PointCloud.from_o3d(pcd2)
    print(f"  PLY1: {pcd1.points.shape[0]} 点")
    print(f"  PLY2: {pcd2.points.shape[0]} 点")  
    
    pcd1_ee = pcd1.transformed(T_cam_ee, squeeze=True)
    pcd2_ee = pcd2.transformed(T_cam_ee, squeeze=True)

    gripper1 = PointCloud.crop_pointcloud_bbox(pcd1_ee, bbox=gripper_bbox)
    gripper2 = PointCloud.crop_pointcloud_bbox(pcd2_ee, bbox=gripper_bbox)
    print(f"  裁剪后 PLY1: {gripper1.points.shape[0]} 点")
    print(f"  裁剪后 PLY2: {gripper2.points.shape[0]} 点")
    
    gripper_ee = concat(gripper1, gripper2)
    print(f"  拼接后: {gripper_ee.points.shape[0]} 点")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    gripper_ee.save(output_path)
    print(f"  ✓ 已保存到: {output_path}")
        
    PointCloud.show(gripper_ee)
    
    return gripper_ee


def main():
    parser = argparse.ArgumentParser(description="grasp_pcd")
    parser.add_argument("--ply1",  default='/home/hkcrc/DCIM/rs1111_grasp1/cloud.ply')
    parser.add_argument("--ply2",  default='/home/hkcrc/DCIM/rs1111_grasp2/cloud.ply')
    parser.add_argument("--cam2base", default="/home/hkcrc/diffusion_edfs/diffusion_edf/tools/configs/scene_cam.json")
    parser.add_argument("--bbox", type=float, nargs=6, 
                        default=[-0.1, 0.1, -0.1, 0.1, -0.05, 0.15])
    parser.add_argument("--robot-ip", default="192.168.56.101")
    parser.add_argument("--output", default="gripper_pcd")
    
    args = parser.parse_args()
    
    bbox = (
        (args.bbox[0], args.bbox[1]),  # x
        (args.bbox[2], args.bbox[3]),  # y
        (args.bbox[4], args.bbox[5])   # z
    )

    robot = RI()
    pos, quat = robot.get_current_pose()
    T_base_ee = SE3(torch.tensor([*quat, *pos], dtype=torch.float32)).inv()
    pos_cam, quat_cam = TM.load_json(args.cam2base)
    T_cam_base = SE3(torch.tensor([*quat_cam, *pos_cam], dtype=torch.float32))
    T_cam_ee = SE3.multiply( T_cam_base,T_base_ee)

    process_gripper_pointcloud(
        ply1_path=args.ply1,
        ply2_path=args.ply2,
        gripper_bbox=bbox,
        T_cam_ee=T_cam_ee,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
