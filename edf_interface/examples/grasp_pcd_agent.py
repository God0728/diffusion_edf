import argparse
import torch
from edf_interface.data import SE3
from edf_interface.modules import RobotInterface as RI
from edf_interface.modules import TransformManager as TM
from tools.process_gripper_pointcloud import process_gripper_pointcloud
import socket, math
"""
TODO
"""
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