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
from edf_interface.data.base import *
from edf_interface.data.preprocess import downsample
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
        camera_model_path: str ,
        cam2ee_path: str ,       
        session_root: str ,
        device: str ,
        scale: int = 1,
        gripper_bbox: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
            (-0.5, 0.6),  # x
            (0.25, 0.8),  # y
            (-0.1, 0.6) # z
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
            pcd_pose_dir=session_dir / "scene_pcd",
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
        time.sleep(20.0)
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
            output_dir=str(self.paths.raw_pose_dir),
            scale=self.scale,
        )
        
        logger.success(f"[Step 3] Point clouds generated:\n  - {self.paths.raw_pose_dir}")

    def step4_trans_pointclouds(self) -> Path:
        
        ply_files_pose = list(self.paths.raw_pose_dir.rglob("*.ply"))
        
        
        ply_path = ply_files_pose[0]

        import numpy as np
        pcd_o3d = o3d.io.read_point_cloud(str(ply_path))
        pcd = PointCloud.from_o3d(pcd_o3d)
        logger.info(f"[Step 4] Loaded: {pcd.points.shape[0]} points")
        
        position, quaternion = TransformManager.load_json(str(self.cam2ee_path))
        T_cam_ee = TransformManager.pose_to_matrix(position, quaternion)
        
        T_base_ee = TransformManager.pose_to_matrix(self.pose, self.quat)
        
        T = T_base_ee @ T_cam_ee
        
        points_np = pcd.points.cpu().numpy()
        points_tf = TransformManager.apply_points(points_np.astype(np.float32), T)
        
        points_tensor = torch.from_numpy(points_tf).float()
        pcd_transformed = PointCloud(points=points_tensor, colors=pcd.colors)
        pcd_transformed = PointCloud.crop_pointcloud_bbox(pcd_transformed, bbox=self.gripper_bbox)
        pcd_transformed = PointCloud.remove_outliers(pcd_transformed, nb_neighbors=20, std_ratio=2.0)    
        output_dir = self.paths.pcd_pose_dir 
        output_dir.mkdir(parents=True, exist_ok=True)
        pcd_transformed.save(str(output_dir))
        logger.success(f"[Step 4] Saved to: {output_dir}")
        
        try:
            pcd_vis = downsample(pcd_transformed, voxel_size=0.005)
            fig = pcd_vis.show(point_size=2.0, name="Scene PointCloud (Base Frame)")
            logger.info(f"  - Figure type: {type(fig)}")
            logger.info(f"  - Figure data length: {len(fig.data) if hasattr(fig, 'data') else 'N/A'}")
            
            fig.show()
        except Exception as e:
            logger.warning(f"[Step 4] Visualization skipped: {e}")
        

        return output_dir

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
        default="189.20,-21.22,-48.39,-157.33,58.46, 241.34",
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