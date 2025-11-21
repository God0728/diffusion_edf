#!/usr/bin/env python3
import multiprocessing as mp

AUTHKEY = b"diff_edf_secret"
mp.current_process().authkey = AUTHKEY
from multiprocessing.connection import Client
from typing import List, Tuple, Dict
import torch
from edf_interface import data


class ModelClient:
    
    def __init__(self, socket_path: str = "/tmp/diff_edf_model.sock", 
                 authkey: str = "diff_edf_secret"):
        self.socket_path = socket_path
        self.authkey = authkey.encode()
    
    def request_trajectories(
        self,
        scene_pcd: data.PointCloud,
        grasp_pcd: data.PointCloud,
        current_poses: data.SE3,
        task: str,
    ) -> Tuple[List[data.SE3], Dict[str, torch.Tensor]]:
        
        # 1. 构造请求
        request = {
            "cmd": "request_trajectories",
            "args": {
                "scene_pcd": scene_pcd.to("cpu"),
                "grasp_pcd": grasp_pcd.to("cpu"),
                "current_poses": current_poses.to("cpu"),
                "task": task
            }
        }
        
        # 2. 连接服务器并发送
        conn = Client(self.socket_path, family='AF_UNIX', authkey=self.authkey)
        try:
            conn.send(request)
            response = conn.recv()
        finally:
            conn.close()
        
        # 3. 检查响应
        if not response.get("ok"):
            raise RuntimeError(f"模型服务器返回错误: {response.get('error')}")
        
        # 4. 反序列化结果
        traj_tensors = response["trajectories"]
        trajectories = [data.SE3(poses=t) for t in traj_tensors]
        info = response["info"]
        
        return trajectories, info
    
    def shutdown(self):
        """关闭服务器"""
        conn = Client(self.socket_path, family='AF_UNIX', authkey=self.authkey)
        try:
            conn.send({"cmd": "shutdown"})
            conn.recv()
        finally:
            conn.close()