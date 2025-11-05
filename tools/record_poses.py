#!/usr/bin/env python3
"""
智能Pose记录工具 - 用于记录机械臂抓取和放置的6D poses

工作流程:
1. 第1次按键 -> demo_0/step_0 (抓取pose)
2. 第2次按键 -> demo_0/step_1 (放置pose)
3. 第3次按键 -> demo_1/step_0 (抓取pose)
4. 第4次按键 -> demo_1/step_1 (放置pose)
...以此类推

数据结构:
dataset/
  rebar_grasping/
    data.yaml
    data/
      demo_0/
        metadata.yaml
        step_0/
          target_poses/
            poses.pt
            metadata.yaml
        step_1/
          target_poses/
            poses.pt
            metadata.yaml
"""

import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation
import yaml
from datetime import datetime
import json

try:
    from rtde_receive import RTDEReceiveInterface
    RTDE_AVAILABLE = True
except ImportError:
    RTDE_AVAILABLE = False
    print("⚠ 警告: rtde_receive 未安装，将使用测试模式")


class PoseRecorder:
    def __init__(self, dataset_name="rebar_grasping", base_dir=None, robot_ip="192.168.56.101"):
        """
        初始化Pose记录器
        
        Args:
            dataset_name: 数据集名称（例如: "rebar_grasping"）
            base_dir: 数据集根目录，默认为 ../demo/
            robot_ip: 机械臂IP地址
        """
        if base_dir is None:
            # 默认放在demo文件夹下
            base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "demo")
        
        self.dataset_path = os.path.join(base_dir, dataset_name)
        self.data_dir = os.path.join(self.dataset_path, "data")
        self.robot_ip = robot_ip
        
        # 状态追踪
        self.current_demo = 0
        self.current_step = 0  # 0=抓取, 1=放置
        self.record_count = 0
        
        # 加载或创建状态文件
        self.state_file = os.path.join(self.dataset_path, ".recording_state.json")
        self._load_state()
        
        # 初始化机器人连接
        self.rtde_receive = None
        if RTDE_AVAILABLE:
            try:
                self.rtde_receive = RTDEReceiveInterface(robot_ip)
                print(f"✓ 已连接到机械臂: {robot_ip}")
            except Exception as e:
                print(f"⚠ 无法连接到机械臂: {e}")
                print("将使用测试模式")
        
        # 确保数据集结构存在
        self._ensure_dataset_structure()
    
    def _load_state(self):
        """加载记录状态"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.current_demo = state.get('current_demo', 0)
                    self.current_step = state.get('current_step', 0)
                    self.record_count = state.get('record_count', 0)
                print(f"✓ 加载上次状态: demo_{self.current_demo}/step_{self.current_step}")
            except:
                print("⚠ 状态文件损坏，使用默认状态")
    
    def _save_state(self):
        """保存记录状态"""
        state = {
            'current_demo': self.current_demo,
            'current_step': self.current_step,
            'record_count': self.record_count,
            'last_update': datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _ensure_dataset_structure(self):
        """确保数据集目录结构存在"""
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 创建或更新 data.yaml
        data_yaml_path = os.path.join(self.dataset_path, "data.yaml")
        if not os.path.exists(data_yaml_path):
            with open(data_yaml_path, 'w') as f:
                f.write("# Dataset entries will be added automatically\n")
                f.write("# Format:\n")
                f.write('# - path: "data/demo_X"\n')
                f.write('#   type: "DemoSequence"\n')
            print(f"✓ 创建 data.yaml")
    
    def _update_data_yaml(self, demo_num):
        """更新 data.yaml 文件，添加新的demo条目"""
        data_yaml_path = os.path.join(self.dataset_path, "data.yaml")
        
        # 读取现有条目
        existing_demos = []
        if os.path.exists(data_yaml_path):
            try:
                with open(data_yaml_path, 'r') as f:
                    content = yaml.safe_load(f)
                    if content:
                        existing_demos = [entry['path'] for entry in content if isinstance(entry, dict)]
            except:
                pass
        
        # 检查demo是否已存在
        demo_path = f"data/demo_{demo_num}"
        if demo_path not in existing_demos:
            # 添加新条目
            with open(data_yaml_path, 'a') as f:
                f.write(f'- path: "{demo_path}"\n')
                f.write(f'  type: "DemoSequence"\n')
            print(f"✓ 添加 demo_{demo_num} 到 data.yaml")
    
    def _create_demo_structure(self, demo_num, step_num):
        """创建demo和step的目录结构"""
        demo_path = os.path.join(self.data_dir, f"demo_{demo_num}")
        step_path = os.path.join(demo_path, f"step_{step_num}")
        target_poses_path = os.path.join(step_path, "target_poses")
        
        os.makedirs(target_poses_path, exist_ok=True)
        
        # 创建 demo metadata.yaml
        demo_metadata_path = os.path.join(demo_path, "metadata.yaml")
        if not os.path.exists(demo_metadata_path):
            with open(demo_metadata_path, 'w') as f:
                yaml.dump({
                    '__type__': 'DemoSequence',
                    'name': ''
                }, f, default_flow_style=False)
        
        # 创建 target_poses metadata.yaml
        poses_metadata_path = os.path.join(target_poses_path, "metadata.yaml")
        with open(poses_metadata_path, 'w') as f:
            yaml.dump({
                '__type__': 'SE3',
                'name': '',
                'unit_length': '1 [m]'
            }, f, default_flow_style=False)
        
        return target_poses_path
    
    def _get_robot_pose(self):
        """
        获取机械臂当前末端pose
        
        Returns:
            pose: [qx, qy, qz, qw, x, y, z] 形式的7D numpy数组
        """
        if self.rtde_receive is not None:
            try:
                # 获取末端执行器的6D pose [x, y, z, rx, ry, rz]
                tcp_pose = self.rtde_receive.getActualTCPPose()
                
                # 提取位置和旋转向量
                position = np.array(tcp_pose[:3])  # [x, y, z]
                rotvec = np.array(tcp_pose[3:])    # [rx, ry, rz] 轴角表示
                
                # 转换为四元数
                rotation = Rotation.from_rotvec(rotvec)
                quat = rotation.as_quat()  # [qx, qy, qz, qw]
                
                # 组合成 [qx, qy, qz, qw, x, y, z]
                pose = np.concatenate([quat, position])
                
                return pose
                
            except Exception as e:
                print(f"❌ 读取机械臂pose失败: {e}")
                return None
        else:
            # 测试模式：生成随机pose
            print("⚠ 测试模式：生成随机pose")
            quat = Rotation.random().as_quat()  # [qx, qy, qz, qw]
            position = np.random.uniform(-0.5, 0.5, 3)
            return np.concatenate([quat, position])
    
    def record_pose(self):
        """
        记录当前pose
        
        Returns:
            success: 是否成功记录
            info: 记录信息字典
        """
        # 确定当前要保存到哪个位置
        demo_num = self.current_demo
        step_num = self.current_step
        
        step_name = "抓取(grasp)" if step_num == 0 else "放置(place)"
        
        print(f"\n{'='*60}")
        print(f"准备记录 Pose #{self.record_count + 1}")
        print(f"位置: demo_{demo_num}/step_{step_num} ({step_name})")
        print(f"{'='*60}")
        
        # 获取机械臂pose
        print("正在读取机械臂pose...")
        pose = self._get_robot_pose()
        
        if pose is None:
            print("❌ 获取pose失败")
            return False, None
        
        # 显示pose信息
        quat = pose[:4]  # [qx, qy, qz, qw]
        position = pose[4:]  # [x, y, z]
        euler = Rotation.from_quat(quat).as_euler('xyz', degrees=True)
        
        print(f"\n✓ 成功读取Pose:")
        print(f"  四元数 [qx,qy,qz,qw]: {quat}")
        print(f"  位置 [x,y,z] (米):    {position}")
        print(f"  欧拉角 (度):          {euler}")
        
        # 确认保存
        confirm = input(f"\n确认保存到 demo_{demo_num}/step_{step_num}? (y/n, 默认y): ").strip().lower()
        if confirm == 'n':
            print("❌ 取消保存")
            return False, None
        
        # 创建目录结构
        target_poses_path = self._create_demo_structure(demo_num, step_num)
        
        # 保存poses.pt (格式: [1, 7] - 一个pose)
        poses_tensor = torch.from_numpy(pose.reshape(1, 7).astype(np.float32))
        pose_file = os.path.join(target_poses_path, "pose.pt")
        torch.save(poses_tensor, pose_file)
        
        # 更新data.yaml
        self._update_data_yaml(demo_num)
        
        print(f"\n✅ Pose已保存:")
        print(f"   文件: {pose_file}")
        print(f"   格式: {poses_tensor.shape} tensor")
        
        # 更新状态
        self.record_count += 1
        
        # 切换到下一个位置
        if step_num == 0:
            # 从step_0切换到step_1（同一个demo）
            self.current_step = 1
            print(f"\n➡️  下次将记录: demo_{demo_num}/step_1 (放置pose)")
        else:
            # 从step_1切换到下一个demo的step_0
            self.current_demo += 1
            self.current_step = 0
            print(f"\n➡️  下次将记录: demo_{self.current_demo}/step_0 (抓取pose)")
        
        self._save_state()
        
        info = {
            'demo_num': demo_num,
            'step_num': step_num,
            'step_name': step_name,
            'pose': pose,
            'file': pose_file,
            'total_count': self.record_count
        }
        
        return True, info
    
    def reset_state(self):
        """重置记录状态到demo_0/step_0"""
        self.current_demo = 0
        self.current_step = 0
        self.record_count = 0
        self._save_state()
        print("✓ 状态已重置到 demo_0/step_0")
    
    def show_status(self):
        """显示当前记录状态"""
        print(f"\n{'='*60}")
        print(f"📊 记录状态")
        print(f"{'='*60}")
        print(f"数据集路径: {self.dataset_path}")
        print(f"当前位置:   demo_{self.current_demo}/step_{self.current_step}")
        print(f"已记录数量: {self.record_count}")
        step_name = "抓取(grasp)" if self.current_step == 0 else "放置(place)"
        print(f"下次记录:   {step_name}")
        print(f"{'='*60}\n")
    
    def close(self):
        """关闭连接"""
        if self.rtde_receive is not None:
            try:
                # RTDE接口通常不需要显式关闭，但保留接口以防万一
                pass
            except:
                pass


def interactive_mode():
    """交互式记录模式"""
    print("="*60)
    print("🤖 机械臂Pose记录工具")
    print("="*60)
    print()
    
    # 配置
    dataset_name = input("数据集名称 (默认: rebar_grasping): ").strip()
    if not dataset_name:
        dataset_name = "rebar_grasping"
    
    robot_ip = input("机械臂IP地址 (默认: 192.168.56.101): ").strip()
    if not robot_ip:
        robot_ip = "192.168.56.101"
    
    # 创建记录器
    recorder = PoseRecorder(dataset_name=dataset_name, robot_ip=robot_ip)
    recorder.show_status()
    
    print("\n命令说明:")
    print("  [Enter]  - 记录当前pose")
    print("  r        - 重置状态到demo_0/step_0")
    print("  s        - 显示当前状态")
    print("  q        - 退出")
    print()
    
    try:
        while True:
            cmd = input("\n按Enter记录pose (或输入命令): ").strip().lower()
            
            if cmd == 'q':
                print("👋 退出记录")
                break
            elif cmd == 'r':
                confirm = input("确认重置状态? (y/n): ").strip().lower()
                if confirm == 'y':
                    recorder.reset_state()
            elif cmd == 's':
                recorder.show_status()
            else:
                # 记录pose
                success, info = recorder.record_pose()
                if success:
                    print(f"\n✅ 总共已记录 {info['total_count']} 个poses")
    
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，退出记录")
    
    finally:
        recorder.close()


def quick_record_mode():
    """快速记录模式（无交互）"""
    recorder = PoseRecorder()
    success, info = recorder.record_pose()
    recorder.close()
    return success


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="机械臂Pose记录工具")
    parser.add_argument("--quick", action="store_true", help="快速记录模式（直接记录当前pose）")
    parser.add_argument("--dataset", default="rebar_grasping", help="数据集名称")
    parser.add_argument("--robot-ip", default="192.168.56.101", help="机械臂IP地址")
    parser.add_argument("--reset", action="store_true", help="重置记录状态")
    
    args = parser.parse_args()
    
    if args.reset:
        recorder = PoseRecorder(dataset_name=args.dataset, robot_ip=args.robot_ip)
        recorder.reset_state()
        recorder.close()
    elif args.quick:
        quick_record_mode()
    else:
        interactive_mode()
