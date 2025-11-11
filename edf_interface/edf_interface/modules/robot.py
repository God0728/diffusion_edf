import rtde_receive
import numpy as np
from scipy.spatial.transform import Rotation as R
import socket
import math
from typing import List, Tuple, Optional
import time
class RobotInterface:
    def __init__(self, robot_ip="192.168.56.101"):  

        self.robot_ip = robot_ip
        self.port = 30002
        self.rtde = None

    def get_current_pose(self):
        """
        outputs:
            position: List[float] 末端位置 [x, y, z]
        """
        self.rtde = rtde_receive.RTDEReceiveInterface(self.robot_ip)
        print(f"成功连接机械臂: {self.robot_ip}")
        # 获取当前TCP姿态
        tcp_pose = self.rtde.getActualTCPPose()
        
        position = tcp_pose[:3]
        rotvec = tcp_pose[3:]
        quaternion = R.from_rotvec(rotvec).as_quat()
        
        self.rtde.disconnect()
        
        print(f"   ✓ 获取当前末端姿态: {position}")
        return list(position), list(quaternion)
    
    
    def get_joint_angles(self):
        """
        outputs:
            joint_angles: [q1, q2, q3, q4, q5, q6] 关节角度 (弧度)
        """
        self.rtde = rtde_receive.RTDEReceiveInterface(self.robot_ip)
        print(f"成功连接机械臂: {self.robot_ip}")
        
        # 获取关节角度
        joint_angles = self.rtde.getActualQ() 
        self.rtde.disconnect()
        
        print(f"关节角度 (弧度): {joint_angles}")
        return list(joint_angles)
    
    def move_to_joint_rad(
        self,
        joint_rad: List[float],
        velocity: float = 0.2,
        acceleration: float = 1.0,
    ) -> bool:
            ur_script = f"""def remote_move():
  target_joint = {joint_rad}
  movej(target_joint, a={acceleration}, v={velocity})
  sleep(1.0)
end
"""
            print("将发送的脚本：")
            print(ur_script)

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(8)
                s.connect((self.robot_ip, self.port))
                s.sendall(ur_script.encode("utf-8"))
                
                # UR 只会返回简单 “Program received” 字符串，或在出错时返回报错行
                try:
                    resp = s.recv(1024)
                    print("机器人响应:", resp.decode(errors="ignore"))
                except socket.timeout:
                    print("控制器未返回（通常也算正常）")
    
    def move_to_pose(
        self,
        position: List[float],
        quaternion: List[float],
        velocity: float = 0.2,
        acceleration: float = 1.0,
        wait: bool = True,
        timeout: float = 10.0
    ) -> bool:
        try:
            # 四元数转旋转向量
            rotvec = R.from_quat(quaternion).as_rotvec()
            target_pose = list(position) + list(rotvec)
            
            ur_script = f"""
            def remote_move():
    target = p{target_pose}
    movel(target, a={acceleration}, v={velocity})
end
remote_move()
            """
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(8)
                s.connect((self.robot_ip, self.port))
                s.sendall(ur_script.encode("utf-8"))
                
                # UR 只会返回简单 “Program received” 字符串，或在出错时返回报错行
                try:
                    resp = s.recv(1024)
                    print("机器人响应:", resp.decode(errors="ignore"))
                except socket.timeout:
                    print("控制器未返回（通常也算正常）")
            
        except Exception as e:
            print(f"❌ 运动失败: {e}")
            return False
