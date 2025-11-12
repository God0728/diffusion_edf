#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import math
import torch
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional
from loguru import logger

from edf_interface.modules.robot import RobotInterface
from edf_interface.modules.camera import RealSenseCalibrator
from process_gripper_pointcloud import *
from edf_interface.modules.transform import TransformManager
from edf_interface.data import SE3


# --- Stereo PCD  ---
stereo_tools_path = Path("/home/hkcrc/handeye")
sys.path.insert(0, str(stereo_tools_path))
from StereoPCDTools.stereo_pcd_generator.test import generate_pcd_dir


@dataclass
class SessionPaths:

    session_dir: Path
    raw_pose0_dir: Path
    raw_pose1_dir: Path
    pcd_pose0_dir: Path
    pcd_pose1_dir: Path
    fused_dir: Path


class GraspSequencePipeline:

    def __init__(
        self,
        target_joint_deg: List[float],
        rotate_deg: float = 180.0,
        camera_model_path: str = "/home/hkcrc/diffusion_edfs/diffusion_edf/edf_interface/configs/realsense_camera_model.json",
        cam2base_path: str = "/home/hkcrc/diffusion_edfs/diffusion_edf/edf_interface/configs/scene_cam.json",       
        session_root: str = "/home/hkcrc/diffusion_edfs/diffusion_edf/edf_interface/run_sessions",
        device: str = "134322070890",
        scale: int = 1,
        # gripper_bbox: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        #     (-0.3, 0.3),  # x
        #     (-0.3, 0.3),  # y
        #     (-0.5, 0.5) # z
        # ),
        gripper_bbox: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
            (-0.05, 0.05),  # x
            (-0.05, 0.05),  # y
            (-0.05, 0.2) # z
        ),
    ):
        logger.info("========== Init GraspSequencePipeline ==========")

        # Robot
        self.target_joint_rad = [math.radians(j) for j in target_joint_deg]
        self.rotate_rad = math.radians(rotate_deg)
        self.robot = RobotInterface()
        logger.info(f"[Robot] target joint angles (rad): {self.target_joint_rad}")
        logger.info(f"[Robot] rotation angle: {rotate_deg}°")

        # Camera
        self.device = device
        self.camera_model_path = Path(camera_model_path)
        self.cam2base_path = Path(cam2base_path)
        self.calibrator: Optional[RealSenseCalibrator] = None
        logger.info(f"[Camera] device: {device}")
        logger.info(f"[Camera] model: {camera_model_path}")

        # Point Cloud
        self.scale = scale
        self.gripper_bbox = gripper_bbox
        self.pose0 =None
        self.pose1 =None
        self.quat0 =None
        self.quat1 =None
        # Session
        self.session_root = Path(session_root)
        self.session_root.mkdir(parents=True, exist_ok=True)
        self.paths: Optional[SessionPaths] = None

    def _prepare_session_paths(self) -> SessionPaths:
        sid = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.session_root / sid
        
        paths = SessionPaths(
            session_dir=session_dir,
            raw_pose0_dir=session_dir / "raw" / "pose0",
            raw_pose1_dir=session_dir / "raw" / "pose1",
            pcd_pose0_dir=session_dir / "pcd" / "pose0",
            pcd_pose1_dir=session_dir / "pcd" / "pose1",
            fused_dir=session_dir / "fused",
        )

        for d in [
            paths.raw_pose0_dir,
            paths.raw_pose1_dir,
            paths.pcd_pose0_dir,
            paths.pcd_pose1_dir,
            paths.fused_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        logger.info(f"[Session] directory: {session_dir}")
        return paths

    def _init_camera(self) -> None:
        if self.calibrator is None:
            logger.info("[Camera] Starting camera...")
            self.calibrator = RealSenseCalibrator(self.device)
            time.sleep(1.0)  

    def _stop_camera(self) -> None:
        if self.calibrator is not None:
            self.calibrator.stop()
            self.calibrator = None
            logger.info("[Camera] Camera stopped")
    def step1_move_to_target(self) -> None:
        logger.info("[Step 1] Moving robot to target joint angles")
        self.robot.move_to_joint_rad(joint_rad=self.target_joint_rad)
        time.sleep(18.0)
        self.pose0, self.quat0 = self.robot.get_current_pose()
        logger.success("[Step 1] Move completed")

    def step2_capture_pose0(self) -> Tuple[Path, Path]:
        logger.info("[Step 2] Capturing Pose0 stereo images")
        self._init_camera()
        
        output_dir = self.paths.raw_pose0_dir
        self.calibrator.capture_images(output_dir=str(output_dir))
        
        left_imgs = sorted(output_dir.glob("A_*.jpg"))
        right_imgs = sorted(output_dir.glob("D_*.jpg"))
        
        if not left_imgs or not right_imgs:
            logger.warning("[Step 2] No A_/D_ images detected, please check capture_images output")
            return (output_dir / "A_dummy.jpg", output_dir / "D_dummy.jpg")
        
        logger.success(f"[Step 2] Saved: {left_imgs[0].name}, {right_imgs[0].name}")
        return (left_imgs[0], right_imgs[0])

    def step3_rotate_180(self) -> None:
        logger.info(f"[Step 3] wrist3 rotation {math.degrees(self.rotate_rad):.1f}°")
        
        current_joints = self.target_joint_rad
        new_joints = current_joints.copy()
        new_joints[5] -= self.rotate_rad  
        
        self.robot.move_to_joint_rad(joint_rad=new_joints)
        time.sleep(18.0)
        self.pose1, self.quat1 = self.robot.get_current_pose()
        logger.success("[Step 3] COMPLETED")

    def step4_capture_pose1(self) -> Tuple[Path, Path]:
        logger.info("[Step 4] Capturing Pose1 stereo images")
        
        output_dir = self.paths.raw_pose1_dir
        self.calibrator.capture_images(output_dir=str(output_dir))
        
        left_imgs = sorted(output_dir.glob("A_*.jpg"))
        right_imgs = sorted(output_dir.glob("D_*.jpg"))
        
        if not left_imgs or not right_imgs:
            logger.warning("[Step 4] No A_/D_ images detected, please check capture_images output")
            return (output_dir / "A_dummy.jpg", output_dir / "D_dummy.jpg")
        
        logger.success(f"[Step 4] Saved: {left_imgs[0].name}, {right_imgs[0].name}")
        return (left_imgs[0], right_imgs[0])

    def step5_generate_pointclouds(self) -> None:
        logger.info("[Step 5] Generating point clouds Pose0 + Pose1")
        
        # Pose0
        logger.info(f"[PCD] Processing Pose0: {self.paths.raw_pose0_dir}")
        generate_pcd_dir(
            raw_dir=str(self.paths.raw_pose0_dir),
            camera_model_path=str(self.camera_model_path),
            output_dir=str(self.paths.pcd_pose0_dir),
            scale=self.scale,
        )
        
        # Pose1
        logger.info(f"[PCD] Processing Pose1: {self.paths.raw_pose1_dir}")
        generate_pcd_dir(
            raw_dir=str(self.paths.raw_pose1_dir),
            camera_model_path=str(self.camera_model_path),
            output_dir=str(self.paths.pcd_pose1_dir),
            scale=self.scale,
        )
        
        logger.success(f"[Step 5] Point clouds generated:\n  - {self.paths.pcd_pose0_dir}\n  - {self.paths.pcd_pose1_dir}")

    def step6_stitch_pointclouds(self) -> Path:

        logger.info("[Step 6] Stitching point clouds")
        ply_files_pose0 = list(self.paths.pcd_pose0_dir.rglob("*.ply"))
        ply_files_pose1 = list(self.paths.pcd_pose1_dir.rglob("*.ply"))
        
        if not ply_files_pose0 or not ply_files_pose1:
            logger.warning(f"[Step 6] No point cloud files found:\n  Pose0: {len(ply_files_pose0)}\n  Pose1: {len(ply_files_pose1)}")
            fused_path = self.paths.fused_dir / "fused.ply"
            fused_path.touch()
            return fused_path
        
        ply1_path = ply_files_pose0[0]
        ply2_path = ply_files_pose1[0]
        logger.info(f"[Step 6] PLY1: {ply1_path}")
        logger.info(f"[Step 6] PLY2: {ply2_path}")
        
        pos_cam, quat_cam = TransformManager.load_json(str(self.cam2base_path))
        quat_cam = [quat_cam[3], quat_cam[0], quat_cam[1], quat_cam[2]]
        T_cam_base = SE3(torch.tensor([*quat_cam, *pos_cam], dtype=torch.float32))    

        pos, quat = self.pose0, self.quat0
        quat = [quat[3], quat[0], quat[1], quat[2]]
        T_base_ee1 = SE3(torch.tensor([*quat, *pos], dtype=torch.float32))
        T_cam_ee1 = SE3.multiply(T_base_ee1.inv(), T_cam_base)

        pos, quat = self.pose1, self.quat1
        quat = [quat[3], quat[0], quat[1], quat[2]]
        T_base_ee2 = SE3(torch.tensor([*quat, *pos], dtype=torch.float32))
        T_cam_ee2 = SE3.multiply( T_base_ee2.inv(), T_cam_base)
       
        process_gripper_pointcloud(
            ply1_path=str(ply1_path),
            ply2_path=str(ply2_path),
            gripper_bbox=self.gripper_bbox,
            T_cam_ee1=T_cam_ee1,
            T_cam_ee2=T_cam_ee2,
            output_dir=str(self.paths.fused_dir),
        )
        
        fused_files = list(self.paths.fused_dir.glob("*.ply"))
        if fused_files:
            fused_path = fused_files[0]
        else:
            fused_path = self.paths.fused_dir / "fused.ply"
        
        logger.success(f"[Step 6] Fused point cloud: {fused_path}")
        return fused_path

    def run(self) -> None:
        try:
            self.paths = self._prepare_session_paths()
            
            self.step1_move_to_target()
            self.step2_capture_pose0()
            self.step3_rotate_180()
            self.step4_capture_pose1()
            self.step5_generate_pointclouds()
            fused_path = self.step6_stitch_pointclouds()
            
            logger.success("========== Process completed ==========")
            logger.info(f"Session directory: {self.paths.session_dir}")
            logger.info(f"Fused point cloud: {fused_path}")
            
        except Exception as e:
            logger.error(f"Process failed: {e}")
            raise
        
        finally:
            self._stop_camera()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="grasp_pipeline")
    parser.add_argument(
        "--joints",
        type=str,
        default="190.03,-74.53,-87.62,-106.24,94.64,306.18",
    )
    parser.add_argument(
        "--rotate_deg",
        type=float,
        default=180.0,
    )
    parser.add_argument(
        "--camera_model",
        type=str,
        default="/home/hkcrc/diffusion_edfs/diffusion_edf/edf_interface/configs/realsense_camera_model.json",
    )    
    parser.add_argument(
        "--cam2base",
        type=str,
        default="/home/hkcrc/diffusion_edfs/diffusion_edf/edf_interface/configs/scene_cam.json",
    )
    parser.add_argument(
        "--session_root",
        type=str,
        default="/home/hkcrc/diffusion_edfs/diffusion_edf/edf_interface/run_sessions",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="134322070890",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    target_joint_deg = [float(x.strip()) for x in args.joints.split(",")]

    pipeline = GraspSequencePipeline(
        target_joint_deg=target_joint_deg,
        rotate_deg=args.rotate_deg,
        camera_model_path=args.camera_model,
        cam2base_path=args.cam2base,
        session_root=args.session_root,
        device=args.device,
        scale=args.scale,
    )
    pipeline.run()


if __name__ == "__main__":
    main()