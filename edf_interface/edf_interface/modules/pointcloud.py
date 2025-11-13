import torch
import open3d as o3d
import numpy as np
import os
from pathlib import Path
from typing import Union, Optional, Tuple, List

class PointCloudHandler:
    """统一的点云处理接口"""
            
    @staticmethod
    def load(filepath: Union[str, Path]) -> Union[torch.Tensor, o3d.geometry.PointCloud]:
        """
        inputs:
            filepath: (.pt, .ply)
        Returns:
            (points, colors) 
        """
        filepath = Path(filepath)
        if filepath.is_dir():
            points_pt = filepath / "points.pt"
            colors_pt = filepath / "colors.pt"
            if points_pt.exists():
                points = torch.load(points_pt).cpu().numpy()
                colors = torch.load(colors_pt).cpu().numpy() if colors_pt.exists() else None
                return points, colors
            ply_files = filepath / "cloud.ply"
            if ply_files.exists():
                pcd = o3d.io.read_point_cloud(str(ply_files))
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors) if pcd.has_colors() else None
                return points, colors
            
        suffix = filepath.suffix.lower()
        if filepath.suffix == '.pt':
            if filepath.name == 'points.pt':
                color_file = filepath.parent / 'colors.pt'
                points = torch.load(filepath).cpu().numpy()
                if color_file.exists():
                    colors = torch.load(color_file).cpu().numpy()
            return points, colors
        elif suffix in ['.ply']:
            pcd = o3d.io.read_point_cloud(filepath)
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) if pcd.has_colors() else None
            return points, colors
        else:
            raise ValueError(f"Unsupported format: {suffix}")
    
    @staticmethod
    def save(points: torch.Tensor, 
             filepath: Union[str, Path],
             colors: torch.Tensor) :
        """保存colors.pt + points.pt"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        points_path = os.path.join(filepath, "points.pt")
        colors_path = os.path.join(filepath, "colors.pt")

        torch.save(points, points_path)
        torch.save(colors, colors_path)
        print(f"保存到:{filepath}")

    
    @staticmethod
    def convert(input_path: Union[str, Path], 
                output_path: Union[str, Path]) -> bool:
        """        
        Args:
            ply2pt
            input_path: .ply
            output_path: 输出文件路径
        """
        try:
            points, colors = PointCloudHandler.load(input_path)
            if isinstance(points, np.ndarray):
                points = torch.from_numpy(points.astype(np.float32))
            if isinstance(colors, np.ndarray):
                colors = torch.from_numpy(colors.astype(np.float32))
            return PointCloudHandler.save(points, output_path, colors)
        except Exception as e:
            print(f"转换失败: {e}")
            return False
    
    
    @staticmethod
    def visualize(data: Union[torch.Tensor, o3d.geometry.PointCloud, np.ndarray, Tuple[np.ndarray, np.ndarray], str, Path],
                  window_name: str = "Point Cloud Viewer"):
        """
        Args:
            data: torch.Tensor, o3d.geometry.PointCloud, np.ndarray,
            (points, colors), tuple, str, Path
        """
        pcd = None
        if isinstance(data, (str, Path)):
            points, colors = PointCloudHandler.load(data)
        elif isinstance(data, tuple) and len(data) == 2:
            points, colors = data
        elif isinstance(data, o3d.geometry.PointCloud):
            pcd = data
        elif isinstance(data, (torch.Tensor, np.ndarray)):
            points = data
        else:
            raise TypeError(f"不支持的输入类型: {type(data)}")

        if pcd is None:
            pts_np = points.cpu().numpy() if isinstance(points, torch.Tensor) else points
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_np)
            if colors is not None:
                cols_np = colors.cpu().numpy() if isinstance(colors, torch.Tensor) else colors
                if cols_np.max() > 1.0:
                    cols_np = cols_np / 255.0
                if cols_np.shape == pts_np.shape:
                    pcd.colors = o3d.utility.Vector3dVector(cols_np)

        o3d.visualization.draw_geometries([pcd], window_name=window_name)



