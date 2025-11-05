def pose_to_homogeneous_matrix(position, quaternion):
    R_mat = R.from_quat(quaternion).as_matrix()  # 四元数转旋转矩阵
    T = np.eye(4)                                # 创建4x4单位矩阵
    T[:3, :3] = R_mat                           # 填入旋转部分
    T[:3, 3] = position                         # 填入平移部分
    return T

def transform_point_to_base(P_cam, T_cam_ee, T_ee_base):
    if len(P_cam) == 3:
        P_cam = np.append(P_cam, 1.0)           # 转为齐次坐标 [x,y,z,1]
    T_cam_base = T_ee_base @ T_cam_ee           # 链式变换：相机→基座
    P_base = T_cam_base @ P_cam                 # 应用变换
    return P_base[:3]                           # 返回3D坐标

def main():
    cam_pos = [0.0462466, 0.146513, 0.144871]      # 相机相对末端执行器的位置
    cam_quat = [-0.00473666, 0.0157987, 0.999864, 0.000675918]  # 相机朝向
    T_cam_ee = pose_to_homogeneous_matrix(cam_pos, cam_quat)
    ee_pose = group.get_current_pose().pose         # 获取当前机器人末端姿态
    ee_pos = [ee_pose.position.x, ee_pose.position.y, ee_pose.position.z]
    ee_quat = [ee_pose.orientation.x, ee_pose.orientation.y, ee_pose.orientation.z, ee_pose.orientation.w]
    T_ee_base = pose_to_homogeneous_matrix(ee_pos, ee_quat)
    P_cam = [center_msg.x, center_msg.y, center_msg.z]  # 相机坐标系中的点
    P_base = transform_point_to_base(P_cam, T_cam_ee, T_ee_base)  # 转换到基座坐标系