#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#重构 yzj1107
from pathlib import Path
import json
import numpy as np
import open3d as o3d
import argparse
import time
from edf_interface.modules.pointcloud import PointCloudHandler as P
from edf_interface.modules.transform import TransformManager as TM
from edf_interface.modules.robot import RobotInterface as RI



def main():
    # 1) 输入路径与变换定义
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=['ee', 'baselink'], required=True,)
    parser.add_argument("--input",  default="/home/hkcrc/diffusion_edfs/diffusion_edf/edf_interface/run_sessions/scene/20251112_154653/pcd/pose/A_1762933625131/cloud.ply")
    parser.add_argument("--output", default="../test/")
    parser.add_argument("--cam-to-ee", default="/home/hkcrc/diffusion_edfs/diffusion_edf/edf_interface/configs/cam_to_ee.json")
    args = parser.parse_args()


    args.output = args.output + f"{args.target}_raw"
        
    input_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 2) 加载点云
    points, colors = P.load(input_path)

    # 3) 构建变换矩阵
    position, quaternion = TM.load_json(args.cam_to_ee)
    T_cam_ee = TM.pose_to_matrix(position, quaternion)

    # 如需手写 4x4，可直接定义：
    # T = np.array([
    #     [1, 0, 0, 0.1],
    #     [0, 1, 0, 0.0],
    #     [0, 0, 1, 0.0],
    #     [0, 0, 0, 1.0],
    # ], dtype=float)

    # 4) 应用变换
    if args.target == "ee":
        T = T_cam_ee
    else:  # baselink
        robot = RI()
        pos_ee, quat_ee = robot.get_current_pose()
        T_base_ee = TM.pose_to_matrix(pos_ee, quat_ee)
        T = T_base_ee @ T_cam_ee
        
    points_tf = TM.apply_points(points.astype(np.float32), T)

    # 5) 可视化
    P.visualize((points_tf, colors), "Transformed Point Cloud")
    P.save(points_tf, out_dir, colors)
    
if __name__ == "__main__":
    main()