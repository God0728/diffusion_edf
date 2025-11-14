#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import torch
from pathlib import Path
from typing import List, Optional
from loguru import logger

CURRENT_DIR = Path(__file__).resolve().parent
EDF_INTERFACE_ROOT = CURRENT_DIR.parent
TOOLS_DIR = EDF_INTERFACE_ROOT / "tools"

# 添加到 sys.path
sys.path.insert(0, str(EDF_INTERFACE_ROOT))
sys.path.insert(0, str(TOOLS_DIR))


from edf_interface.data import SE3, PointCloud
from edf_interface.modules.robot import RobotInterface
from edf_interface.pyro import PyroServer, expose
from scene_sequence import SceneSequencePipeline
from grasp_sequence import GraspSequencePipeline


class EnvService():
    def __init__(
        self,
        robot_ip: str = "192.168.56.101",
        scene_device: str = "153122076446",  # 手眼相机D435
        grasp_device: str = "134322070890",  # 固定相机D435i
        handeye_camera_model: str = "/home/hkcrc/diffusion_edfs/diffusion_edf/edf_interface/configs/realsense_camera_model.json",
        scene_camera_model: str = "/home/hkcrc/diffusion_edfs/diffusion_edf/edf_interface/configs/realsense_camera_model.json",
        cam2ee_path: str = "/home/hkcrc/diffusion_edfs/diffusion_edf/edf_interface/configs/cam_to_ee.json",
        cam2base_path: str = "/home/hkcrc/diffusion_edfs/diffusion_edf/edf_interface/configs/scene_cam.json",
        session_root: str = "edf_interface/run_sessions_agent",
    ):
        
        self.robot = RobotInterface(robot_ip=robot_ip)
        logger.success(f"✓ 机器人连接成功: {robot_ip}")
        
        self.scene_config = {
            "camera_model_path": handeye_camera_model,
            "cam2base_path": cam2ee_path,
            "session_root": f"{session_root}/scene",
            "device": scene_device,
            "scale": 1,
        }
        
        self.grasp_config = {
            "camera_model_path": scene_camera_model,
            "cam2base_path": cam2base_path,
            "session_root": f"{session_root}/grasp",
            "device": grasp_device,
            "scale": 1,
        }
        
        self.default_scene_joints = [189.20, -21.22, -48.39, -157.33, 58.46, 241.34]
        self.default_grasp_joints = [324.46, -119.13, 91.84, -304.96, 76.45, 359.69]
    
        self._cached_scene_pcd: Optional[PointCloud] = None
        self._cached_grasp_pcd: Optional[PointCloud] = None
        
        logger.success("✓ Environment Service 初始化完成!")

    @expose
    def get_current_poses(self) -> SE3:
        
        pos, quat = self.robot.get_current_pose()  # quat: [x,y,z,w]
        
        quat_wxyz = [quat[3], quat[0], quat[1], quat[2]]
        pose_tensor = torch.tensor([*quat_wxyz, *pos], dtype=torch.float32)
        
        current_pose = SE3(pose_tensor.unsqueeze(0))  # shape: (1, 7)
        
        logger.success(f"✓ 当前位姿: pos={pos}, quat={quat}")
        return current_pose

    @expose
    def observe_scene(
        self,
        target_joint_deg: Optional[List[float]] = None,
        bbox: Optional[List[List[float]]] = None,
        return_cached: bool = False
    ) -> PointCloud:

        if return_cached and self._cached_scene_pcd is not None:
            logger.info("返回缓存的场景点云")
            return self._cached_scene_pcd
        
        logger.info("========== 开始采集场景点云 ==========")
        
        joints = target_joint_deg if target_joint_deg else self.default_scene_joints
        
        pipeline = SceneSequencePipeline(
            target_joint_deg=joints,
            **self.scene_config
        )
        
        if bbox is not None:
            pipeline.gripper_bbox = tuple(tuple(b) for b in bbox)
        
        try:
            pipeline.paths = pipeline._prepare_session_paths()
            pipeline.step1_move_to_target()
            pipeline.step2_capture_pose()
            pipeline.step3_generate_pointclouds()
            trans_path = pipeline.step4_trans_pointclouds()
            
            scene_pcd = PointCloud.load(str(trans_path))
            self._cached_scene_pcd = scene_pcd
            
            logger.success(f"✓ 场景点云采集完成! 点数: {scene_pcd.points.shape[0]}")
            return scene_pcd
            
        finally:
            pipeline._stop_camera()

    @expose
    def observe_grasp(
        self,
        target_joint_deg: Optional[List[float]] = None,
        rotate_deg: float = 180.0,
        bbox: Optional[List[List[float]]] = None,
        return_cached: bool = False
    ) -> PointCloud:

        joints = target_joint_deg if target_joint_deg else self.default_grasp_joints
        
        pipeline = GraspSequencePipeline(
            target_joint_deg=joints,
            rotate_deg=rotate_deg,
            **self.grasp_config
        )
        
        if bbox is not None:
            pipeline.gripper_bbox = tuple(tuple(b) for b in bbox)
        
        try:
            pipeline.paths = pipeline._prepare_session_paths()
            pipeline.step1_move_to_target()
            pipeline.step2_capture_pose0()
            pipeline.step3_rotate_180()
            pipeline.step4_capture_pose1()
            pipeline.step5_generate_pointclouds()
            fused_path = pipeline.step6_stitch_pointclouds()
            
            grasp_pcd = PointCloud.load(str(fused_path))
            
            self._cached_grasp_pcd = grasp_pcd
            
            logger.success(f"✓ 抓取点云采集完成! 点数: {grasp_pcd.points.shape[0]}")
            return grasp_pcd
            
        finally:
            pipeline._stop_camera()

    @expose
    def move_se3(
        self,
        target_poses: SE3,
        velocity: float = 0.2,
        acceleration: float = 1.0,
        wait: bool = True
    ) -> bool:
        
        pose_tensor = target_poses.poses[0]  # [w,x,y,z,tx,ty,tz]
        
        quat_wxyz = pose_tensor[:4].cpu().numpy()
        pos = pose_tensor[4:].cpu().numpy()
        
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        
        logger.info(f"目标位置: {pos.tolist()}")
        logger.info(f"目标四元数: {quat_xyzw}")
    
        success = self.robot.move_to_pose(
            position=pos.tolist(),
            quaternion=quat_xyzw,
            velocity=velocity,
            acceleration=acceleration,
            wait=wait
        )
        
        if success:
            logger.success("✓ 机器人移动成功!")
        else:
            logger.error("✗ 机器人移动失败!")
        
        return success


def main():
    service = EnvService()
    server = PyroServer(server_name='env', init_nameserver=True)
    server.register_service(service=service)
    server.run(nonblocking=False) # set nonblocking = True if you want to run server in another thread.

    server.close()


if __name__ == "__main__":
    main()