import pyrealsense2 as rs
import numpy as np
import json
from pathlib import Path
from loguru import logger
import cv2
import time
import string

class RealSenseCalibrator:

    def __init__(self, device:string, width=1280, height=720, fps=30):
        try:
            if device in ["scene", "scene_cam"]:
                device = "153122076446"  # D435
            elif device in ["grasp", "grasp_cam"]:
                device = "134322070890"  # D435i

            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_device(device)       

            self.config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
            self.config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)
            
            self.profile = self.pipeline.start(self.config)
            logger.success(f"RealSense camera initialized successfully: {width}x{height}@{fps}fps")
            
        except Exception as e:
            raise RuntimeError("RealSense camera initialization failed") from e
    
    def get_camera_intrinsics(self):

        if not self.profile:
            raise RuntimeError("camera init failed")
        
        ir_profile_left = self.profile.get_stream(rs.stream.infrared, 1)
        ir_profile_right = self.profile.get_stream(rs.stream.infrared, 2)
        
        ir_intrinsics_left = ir_profile_left.as_video_stream_profile().get_intrinsics()
        ir_intrinsics_right = ir_profile_right.as_video_stream_profile().get_intrinsics()
        
        
        return ir_intrinsics_left, ir_intrinsics_right
    
    def get_camera_extrinsics(self):

        if not self.profile:
            raise RuntimeError("camera init failed")
            
        ir_profile_left = self.profile.get_stream(rs.stream.infrared, 1)
        ir_profile_right = self.profile.get_stream(rs.stream.infrared, 2)

        extrinsics = ir_profile_left.as_video_stream_profile().get_extrinsics_to(
            ir_profile_right.as_video_stream_profile()
        )
        
        return extrinsics

    def capture_images(self, output_dir):

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        frames = self.pipeline.wait_for_frames()
        
        ir_left_frame = frames.get_infrared_frame(1)
        ir_right_frame = frames.get_infrared_frame(2)
        
        ir_left_image = np.asanyarray(ir_left_frame.get_data())
        ir_right_image = np.asanyarray(ir_right_frame.get_data())
                    
        ir_left_bgr = cv2.cvtColor(ir_left_image, cv2.COLOR_GRAY2BGR)
        ir_right_bgr = cv2.cvtColor(ir_right_image, cv2.COLOR_GRAY2BGR)
        
        timestamp = int(time.time() * 1000)
        left_path = output_path / f"A_{timestamp}.jpg"
        right_path = output_path / f"D_{timestamp}.jpg"
        
        cv2.imwrite(str(left_path), ir_left_bgr)
        cv2.imwrite(str(right_path), ir_right_bgr)
        
        logger.info(f"Saved image pair: {left_path.name}, {right_path.name}")
        

    def save_json(self, 
            save_path="/home/hkcrc/diffusion_edfs/diffusion_edf/edf_interface/configs/realsense_camera_model.json"):
        
        ir_left, ir_right = self.get_camera_intrinsics()
        extrinsics = self.get_camera_extrinsics()
        

        def intrinsics_to_matrix(intrinsics):

            return [
                [intrinsics.fx, 0.0, intrinsics.ppx],
                [0.0, intrinsics.fy, intrinsics.ppy],
                [0.0, 0.0, 1.0]
            ]
        
        def coeffs_to_list(intrinsics):
            # RealSense使用Brown-Conrady模型: [k1, k2, p1, p2, k3]
            return list(intrinsics.coeffs)
        

        R = np.array(extrinsics.rotation).reshape(3, 3)
        T = np.array(extrinsics.translation).reshape(3, 1)
        T_mm = T * 1000        

        camera_params = {
            "image_size": [ir_left.width, ir_left.height],
            "is_fisheye": False, 

            "cm1": intrinsics_to_matrix(ir_left),
            "cd1": [coeffs_to_list(ir_left)], 

            "cm2": intrinsics_to_matrix(ir_right),
            "cd2": [coeffs_to_list(ir_right)],

            "R": R.tolist(),
            "T": T_mm.tolist(),

            "baseline_mm": np.linalg.norm(T) * 1000, 
            "model": "Intel RealSense D435i",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        save_path = Path(save_path)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(camera_params, f, indent=4, ensure_ascii=False)
        
        logger.success(f"RealSense camera parameters saved to: {save_path}")
        
    
    
    def stop(self):

        if self.pipeline:
            self.pipeline.stop()
            logger.info("camera stopped")
