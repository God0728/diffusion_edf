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
import open3d as o3d
from edf_interface.data import PointCloud
from edf_interface.modules.robot import RobotInterface
from edf_interface.modules.camera import RealSenseCalibrator
from edf_interface.modules.transform import TransformManager
from edf_interface.data import SE3


# --- Stereo PCD  ---
stereo_tools_path = Path("/home/hkcrc/handeye")
sys.path.insert(0, str(stereo_tools_path))
from StereoPCDTools.stereo_pcd_generator.test import generate_pcd_dir


@dataclass
class SessionPaths:

    session_dir: Path
    raw_pose_dir: Path
    pcd_pose_dir: Path



class SceneSequencePipeline:

    def __init__(
        self,
        target_joint_deg: List[float],
        camera_model_path: str = "/home/hkcrc/diffusion_edfs/diffusion_edf/edf_interface/configs/realsense_camera_model.json",
        cam2ee_path: str = "/home/hkcrc/diffusion_edfs/diffusion_edf/edf_interface/configs/cam_to_ee.json",       
        session_root: str = "/home/hkcrc/diffusion_edfs/diffusion_edf/edf_interface/run_sessions/scene",
        device: str = "153122076446",
        scale: int = 1,
        gripper_bbox: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
            (-0.6, 0.6),  # x
            (-0.6, 0.6),  # y
            (-0.2, 0.6) # z
        ),

    ):
        logger.info("========== Init SceneSequencePipeline ==========")

        # Robot
        self.target_joint_rad = [math.radians(j) for j in target_joint_deg]
        self.robot = RobotInterface()
        logger.info(f"[Robot] target joint angles (rad): {self.target_joint_rad}")

        # Camera
        self.device = device
        self.camera_model_path = Path(camera_model_path)
        self.cam2ee_path = Path(cam2ee_path)
        self.calibrator: Optional[RealSenseCalibrator] = None
        logger.info(f"[Camera] device: {device}")
        logger.info(f"[Camera] model: {camera_model_path}")

        # Point Cloud
        self.scale = scale
        self.gripper_bbox = gripper_bbox
        self.pose =None
        self.quat =None

        # Session
        self.session_root = Path(session_root)
        self.session_root.mkdir(parents=True, exist_ok=True)
        self.paths: Optional[SessionPaths] = None

    def _prepare_session_paths(self) -> SessionPaths:
        sid = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.session_root / sid
        
        paths = SessionPaths(
            session_dir=session_dir,
            raw_pose_dir=session_dir / "raw" / "pose",
            pcd_pose_dir=session_dir / "pcd" / "pose",
        )

        for d in [
            paths.raw_pose_dir,
            paths.pcd_pose_dir,
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
        time.sleep(10.0)
        self.pose, self.quat = self.robot.get_current_pose()
        logger.success("[Step 1] Move completed")

    def step2_capture_pose(self) -> Tuple[Path, Path]:
        logger.info("[Step 2] Capturing Pose stereo images")
        self._init_camera()
        
        output_dir = self.paths.raw_pose_dir
        self.calibrator.capture_images(output_dir=str(output_dir))
        
        left_imgs = sorted(output_dir.glob("A_*.jpg"))
        right_imgs = sorted(output_dir.glob("D_*.jpg"))
        
        if not left_imgs or not right_imgs:
            logger.warning("[Step 2] No A_/D_ images detected, please check capture_images output")
            return (output_dir / "A_dummy.jpg", output_dir / "D_dummy.jpg")
        
        logger.success(f"[Step 2] Saved: {left_imgs[0].name}, {right_imgs[0].name}")
        return (left_imgs[0], right_imgs[0])

    def step3_generate_pointclouds(self) -> None:
        logger.info("[Step 3] Generating point clouds")
        
        # Pose0
        logger.info(f"[PCD] Processing Pose0: {self.paths.raw_pose_dir}")
        generate_pcd_dir(
            raw_dir=str(self.paths.raw_pose_dir),
            camera_model_path=str(self.camera_model_path),
            output_dir=str(self.paths.pcd_pose_dir),
            scale=self.scale,
        )
        
        logger.success(f"[Step 3] Point clouds generated:\n  - {self.paths.pcd_pose_dir}")

    def step4_trans_pointclouds(self) -> Path:
        logger.info("[Step 4] Transforming point clouds to base frame")
        
        ply_files_pose = list(self.paths.pcd_pose_dir.rglob("*.ply"))
        
        if not ply_files_pose:
            logger.warning(f"[Step 4] No point cloud files found")
            empty_path = self.paths.pcd_pose_dir / "empty.ply"
            empty_path.touch()
            return empty_path
        
        ply_path = ply_files_pose[0]
        logger.info(f"[Step 4] Input point cloud: {ply_path}")

        pcd = o3d.io.read_point_cloud(str(ply_path))
        pcd = PointCloud.from_o3d(pcd)
        logger.info(f"[Step 4] Point cloud size: {pcd.points.shape[0]}")
        
        import numpy as np
        from edf_interface.modules.pointcloud import PointCloudHandler as P
        
        points, colors = P.load(ply_path)
        logger.info(f"[Step 4] Loaded with PointCloudHandler: {points.shape[0]} points")
        
        position, quaternion = TransformManager.load_json(str(self.cam2ee_path))
        T_cam_ee = TransformManager.pose_to_matrix(position, quaternion)
        logger.info(f"[Step 4] T_cam_ee loaded: pos={position}, quat={quaternion}")
        
        T_base_ee = TransformManager.pose_to_matrix(self.pose, self.quat)
        logger.info(f"[Step 4] T_base_ee computed: pos={self.pose}, quat={self.quat}")
        
        T = T_base_ee @ T_cam_ee
        logger.info(f"[Step 4] T_cam_base = T_base_ee @ T_cam_ee")
        
        points_tf = TransformManager.apply_points(points.astype(np.float32), T)
        logger.info(f"[Step 4] Point cloud transformed to base frame")
        
        output_dir = self.paths.pcd_pose_dir / "base_raw"
        output_dir.mkdir(parents=True, exist_ok=True)
        P.save(points_tf, output_dir, colors)
        logger.success(f"[Step 4] Saved to: {output_dir}")
        
        try:
            P.visualize((points_tf, colors), "Scene PointCloud (Base Frame)")
        except Exception as e:
            logger.warning(f"[Step 4] Visualization skipped (libGL issue): {e}")
        
        saved_files = list(output_dir.glob("*.pt"))
        if not saved_files:
            saved_files = list(output_dir.glob("*.ply"))
        
        if saved_files:
            output_path = saved_files[0]
        else:
            output_path = output_dir / "pcd.pt"
        
        return output_path

    def run(self) -> None:
        try:
            self.paths = self._prepare_session_paths()
            
            self.step1_move_to_target()
            self.step2_capture_pose()
            self.step3_generate_pointclouds()
            trans_path = self.step4_trans_pointclouds()
            
            logger.success("========== Process completed ==========")
            logger.info(f"Session directory: {self.paths.session_dir}")
            logger.info(f"Transformed point cloud: {trans_path}")
            
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
        default="181.20,-17.35,-50.57,-157.89,65.57, 63.07",
    )
    parser.add_argument(
        "--camera_model",
        type=str,
        default="/home/hkcrc/diffusion_edfs/diffusion_edf/edf_interface/configs/realsense_camera_model.json",
    )    
    parser.add_argument(
        "--cam2ee",
        type=str,
        default="/home/hkcrc/diffusion_edfs/diffusion_edf/edf_interface/configs/cam_to_ee.json",
    )
    parser.add_argument(
        "--session_root",
        type=str,
        default="/home/hkcrc/diffusion_edfs/diffusion_edf/edf_interface/run_sessions/scene",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="153122076446",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    target_joint_deg = [float(x.strip()) for x in args.joints.split(",")]

    pipeline = SceneSequencePipeline(
        target_joint_deg=target_joint_deg,
        camera_model_path=args.camera_model,
        cam2ee_path=args.cam2ee,
        session_root=args.session_root,
        device=args.device,
        scale=args.scale,
    )
    pipeline.run()


if __name__ == "__main__":
    main()