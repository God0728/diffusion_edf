import numpy as np
import torch
import open3d as o3d
import json
import os
from scipy.spatial.transform import Rotation as R

class TransformManager:
    """管理点云的几何变换"""
    
    @staticmethod
    def pose_to_matrix(position, quaternion):
        """位置+四元数 → 4x4变换矩阵"""
        R_mat = R.from_quat(quaternion).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = position
        return T

    @staticmethod
    def load_json(json_file):
        """从JSON文件加载变换参数"""
        if not os.path.isfile(json_file):
            raise ValueError("请输入JSON文件路径")
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        position = data['position']  # [x, y, z]
        quaternion = data['quaternion']  # [qx, qy, qz, qw]
        return position, quaternion

    @staticmethod
    def matrix_to_pose(T: np.ndarray):
        """
        outputs:
            position: List[float] 末端位置 [x, y, z]
            quaternion: List[float] 末端姿态 [qx, qy, qz, qw]
        """
        R = T[:3, :3]  
        t = T[:3, 3]   

        rotation = R.from_matrix(R)
        quaternion = rotation.as_quat()
        return t.tolist(), quaternion.tolist()
    
    @staticmethod
    def apply_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
        """将4x4变换矩阵应用到 Nx3 点云"""
        homo = np.hstack([points, np.ones((points.shape[0], 1), dtype=points.dtype)])
        return (T @ homo.T).T[:, :3]    
    
    @staticmethod
    def invert(T: np.ndarray) -> np.ndarray:
        """4x4 齐次矩阵求逆"""
        return np.linalg.inv(T)
    