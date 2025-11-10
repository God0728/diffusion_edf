#!/usr/bin/env python3
#重构 yzj1110
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import yaml
from modules.robot import RobotInterface as RI

class PoseRecorder:
    
    def __init__(self, dataset_name, robot_ip="192.168.56.101"):
        # 路径
        base = Path(__file__).parent.parent / "demo" / dataset_name
        self.data_dir = base / "data"
        self.state_file = base / ".state.json"
        
        # 状态
        self.demo_idx = 0
        self.step_idx = 0  # 0=grasp, 1=place
        self.total = 0
        
        self.robot = RI(robot_ip)
        
        # 初始化
        self._load_state()
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_state(self):
        """加载状态"""
        if self.state_file.exists():
            s = json.loads(self.state_file.read_text())
            self.demo_idx = s.get('demo', 0)
            self.step_idx = s.get('step', 0)
            self.total = s.get('total', 0)
            print(f"✓ 恢复状态: demo_{self.demo_idx}/step_{self.step_idx}")
    
    def _save_state(self):
        """保存状态"""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps({
            'demo': self.demo_idx,
            'step': self.step_idx,
            'total': self.total,
            'time': datetime.now().isoformat()
        }, indent=2))
    
    def _make_metadata(self, path, type_name, extra=None):
        """创建metadata.yaml"""
        data = {'__type__': type_name, 'name': ''}
        if extra:
            data.update(extra)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def record(self):
        """记录当前pose"""
        demo, step = self.demo_idx, self.step_idx
        name = "grasp" if step == 0 else "place"
        
        print(f"\n{'='*50}")
        print(f"#{self.total + 1}: demo_{demo}/step_{step} ({name})")
        print(f"{'='*50}")
        
        # 获取pose
        pos, quat = self.robot.get_current_pose()
        pose = np.concatenate([quat, pos])  # [qx,qy,qz,qw, x,y,z]
        
        print(f"位置: {pos}")
        print(f"四元数: {quat}")
        
        if input("\n保存? (Enter=是, n=否): ").lower() == 'n':
            print("❌ 已取消")
            return False
        
        # 保存
        demo_dir = self.data_dir / f"demo_{demo}"
        step_dir = demo_dir / f"step_{step}"
        pose_dir = step_dir / "target_poses"
        pose_dir.mkdir(parents=True, exist_ok=True)
        
        # 写文件
        torch.save(torch.tensor(pose.reshape(1, 7), dtype=torch.float32), 
                   pose_dir / "pose.pt")
        self._make_metadata(demo_dir / "metadata.yaml", "DemoSequence")
        self._make_metadata(pose_dir / "metadata.yaml", "SE3", 
                           {'unit_length': '1 [m]'})
        
        print(f"✅ 已保存到: {pose_dir / 'pose.pt'}")
        
        # 更新状态
        self.total += 1
        if step == 0:
            self.step_idx = 1
            print(f"➡️  下一个: demo_{demo}/step_1 (place)")
        else:
            self.demo_idx += 1
            self.step_idx = 0
            print(f"➡️  下一个: demo_{self.demo_idx}/step_0 (grasp)")
        
        self._save_state()
        return True
    


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="rebar_grasping1110")
    parser.add_argument("--robot-ip", default="192.168.56.101")
    args = parser.parse_args()
    
    dataset = input(f"数据集名 [{args.dataset}]: ").strip() or args.dataset
    
    recorder = PoseRecorder(dataset, args.robot_ip)
    
    print("命令: [Enter]=记录 q=退出\n")
    
    try:
        while True:
            cmd = input(">>> ").strip().lower()
            if cmd == 'q':
                break
            else:
                recorder.record()
    except KeyboardInterrupt:
        print("\n\n 退出")


if __name__ == "__main__":
    main()