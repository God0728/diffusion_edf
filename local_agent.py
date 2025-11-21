#!/usr/bin/env python3
import os
os.environ["TORCH_JIT"] = "0"        
os.environ["PYTORCH_JIT_DISABLE"] = "1"

import multiprocessing as mp

AUTHKEY = b"diff_edf_secret"
mp.current_process().authkey = AUTHKEY
import argparse
import sys
import traceback
from multiprocessing.connection import Listener
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ur5e_agent import LocalAgentPipeline
from edf_interface import data


def serialize_se3_list(trajectories):
    return [t.poses.detach().cpu() for t in trajectories]


def serialize_info(info):
    return {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) 
            for k, v in info.items()}


def serve(configs_root: Path, socket_path: str, authkey: bytes):
    """启动模型服务器"""
    print(f"📦 正在加载模型...")
    try:
        pipeline = LocalAgentPipeline(
            configs_root=configs_root,
            compile_score_head=False  
        )
        print(f"✅ 模型加载完成！")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        traceback.print_exc()
        return

    listener = Listener(address=socket_path, family='AF_UNIX', authkey=authkey)
    print(f"🚀 模型服务器运行中: {socket_path}")

    try:
        while True:
            conn = listener.accept()
            print(f"📡 收到客户端连接")
            
            try:
                request = conn.recv()
                cmd = request.get("cmd")
                
                if cmd == "request_trajectories":
                    print(f"   → 处理轨迹请求 (task={request['args']['task']})")
                    
                    scene_pcd = request["args"]["scene_pcd"]
                    grasp_pcd = request["args"]["grasp_pcd"]
                    current_poses = request["args"]["current_poses"]
                    task = request["args"]["task"]
                    
                    trajectories, info = pipeline.request_trajectories(
                        scene_pcd=scene_pcd,
                        grasp_pcd=grasp_pcd,
                        current_poses=current_poses,
                        task=task
                    )
                    
                    response = {
                        "ok": True,
                        "trajectories": serialize_se3_list(trajectories),
                        "info": serialize_info(info)
                    }
                    conn.send(response)
                    print(f"   ✓ 返回 {len(trajectories)} 条轨迹")
                
                elif cmd == "shutdown":
                    print(f"   → 收到关闭命令")
                    conn.send({"ok": True})
                    conn.close()
                    break
                
                else:
                    conn.send({"ok": False, "error": f"未知命令: {cmd}"})
                    
            except Exception as e:
                print(f"   ❌ 处理请求失败: {e}")
                traceback.print_exc()
                conn.send({"ok": False, "error": str(e)})
            finally:
                conn.close()
                
    except KeyboardInterrupt:
        print(f"\n⚠️  服务器已停止")
    finally:
        listener.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型常驻服务器")
    parser.add_argument("--configs-root-dir", type=Path, required=True,
                       help="配置文件路径 (例如 configs/panda_bottle)")
    parser.add_argument("--socket", type=str, default="/tmp/diff_edf_model.sock",
                       help="Unix socket 路径")
    parser.add_argument("--authkey", type=str, default="diff_edf_secret",
                       help="连接密钥")
    args = parser.parse_args()
    serve(
        configs_root=args.configs_root_dir,
        socket_path=args.socket,
        authkey=args.authkey.encode()
    )