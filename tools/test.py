import numpy as np
from scipy.spatial.transform import Rotation as R
import rtde_receive

# cam2ee 变换矩阵
cam2ee = np.array([
    [0.8031, -0.5953, -0.0242, -0.0937],
    [0.5945,  0.8034, -0.0350, -0.0752],
    [0.0402,  0.0137,  0.9991,  0.0400],
    [0,       0,       0,       1.0000]
])

def get_cam_to_base_transform(robot_ip="192.168.56.101"):
    """
    计算相机在base_link坐标系下的位置
    cam2base = ee2base @ cam2ee
    """
    try:
        # 获取ee2base变换
        ee_position, ee_quaternion = get_current_ee_to_base_transform(robot_ip)
        
        # 构建ee2base变换矩阵
        ee2base = np.eye(4)
        ee2base[:3, :3] = R.from_quat(ee_quaternion).as_matrix()
        ee2base[:3, 3] = ee_position
        
        # 计算cam2base = ee2base @ cam2ee
        cam2base = ee2base @ cam2ee
        
        cam_position = cam2base[:3, 3]
        cam_rotation = R.from_matrix(cam2base[:3, :3])
        cam_quaternion = cam_rotation.as_quat()
        
        print(f"   ✓ 相机在base_link的位置: {cam_position}")
        print(f"   ✓ 相机在base_link的姿态(四元数): {cam_quaternion}")
        
        return cam_position, cam_quaternion
        
    except Exception as e:
        print(f"   ✗ 错误: {e}")
        return None, None


def get_current_ee_to_base_transform(robot_ip="192.168.56.101"):
    """
    从机器人获取当前的末端执行器到基座变换
    """
        
    print(f"   连接机器人: {robot_ip}")
    rtde = rtde_receive.RTDEReceiveInterface(robot_ip)
    
    # 获取当前TCP姿态
    tcp_pose = rtde.getActualTCPPose()
    if tcp_pose is None:
        raise ValueError("无法获取TCP姿态")
    
    position = tcp_pose[:3]
    rotvec = tcp_pose[3:]
    quaternion = R.from_rotvec(rotvec).as_quat()
    
    rtde.disconnect()
    
    print(f"   ✓ 获取当前末端姿态: {position}")
    return list(position), list(quaternion)
    
def main():
    robot_ip = "192.168.56.101"
    get_cam_to_base_transform(robot_ip)
    
if __name__ == "__main__":
    main()