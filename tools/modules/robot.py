import rtde_receive

class RobotInterface:
    def __init__(self, robot_ip="192.168.56.101"):  

        self.robot_ip = robot_ip
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
        
        rtde.disconnect()
        
        print(f"   ✓ 获取当前末端姿态: {position}")
        return list(position), list(quaternion)
    
    
    def get_joint_angles(self):
        """
        outputs:
            joint_angles: [q1, q2, q3, q4, q5, q6] 关节角度 (弧度)
        """
        rtde = rtde_receive.RTDEReceiveInterface(self.robot_ip)
        print(f"成功连接机械臂: {self.robot_ip}")
        
        # 获取关节角度
        joint_angles = rtde.getActualQ() 
        rtde.disconnect()
        
        print(f"关节角度 (弧度): {joint_angles}")
        return list(joint_angles)
            
        