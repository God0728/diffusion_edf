import socket
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

def pose_to_homogeneous_matrix(position, quaternion):
    R_mat = R.from_quat(quaternion).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = position
    return T
def transform_point_to_base(P_cam, T_cam_base):
    if len(P_cam) == 3:
        P_cam = np.append(P_cam, 1.0)
    P_base = T_cam_base @ P_cam
    return P_base[:3]
def build_to_matrix(p_base, p_base_orientation, T_gripper2ee):
    T_gripper2base = np.eye(4)
    T_gripper2base[:3, 3] = p_base
    T_gripper2base[:3, :3] = p_base_orientation
    T_ee2base = T_gripper2base @ np.linalg.inv(T_gripper2ee)
    target_pos = T_ee2base[:3, 3]
    target_rot = T_ee2base[:3, :3]
    return target_pos, target_rot
def pixel_to_camera(u, v, Z, fx, fy, cx, cy):
   # 注意：行是 v，列是 u
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z])

if __name__ == "__main__":
    
    gripper_offset_position = [-0.05, 0.145, 0.179]
    gripper_offset_oritation = [0, 0, 0, 1]
    #r = R.from_rotvec(gripper_offset_oritation)
    #gripper_quat = r.as_quat()
    T_gripper2ee = pose_to_homogeneous_matrix(gripper_offset_position, gripper_offset_oritation)

    cam_pos = [0.1169, 0.3309, 0.5086] #result of calibration
    rotation_matrix = np.array([
    [-0.0166, 0.3851, 0.9227],
    [-0.0061, 0.9228, -0.3853],
    [-0.9998, -0.0120, -0.0129]
    ])
    Fxy = np.diag([-1, -1, 1])
    R_new = Fxy @ rotation_matrix

    # 创建旋转对象
    r = R.from_matrix(R_new)

    # 转换为四元数：[x, y, z, w]
    cam_quat = r.as_quat()
    print("四元数 (x, y, z, w):", cam_quat)
    #cam_quat = []
    #camera to base transition
    T_cam_base = pose_to_homogeneous_matrix(cam_pos, cam_quat)
    u = 808.5
    v = 350.5
    Z = 0.553
    fx = 2760.0421
    fy = 2758.4115
    cx = 2047.4786
    cy = 1497.3456
    u2 = 808.5
    v2 = 517
    z2 = 0.545
    u3 = 903.5
    v3 = 515.5
    z3 = 0.542
    u4 = 1097
    v4 = 686
    z4 = 0.540
    u5 = 1098
    v5 = 515.5
    z5 = 0.540

    scale_x = 3.44898
    scale_y = 3.450624
    u_orig = u * scale_x  
    v_orig = v * scale_y  
    
    u_orig2 = u2 * scale_x  
    v_orig2 = v2 * scale_y 
   
    u_orig3 = u3 * scale_x  
    v_orig3 = v3 * scale_y 

    u_orig4 = u4 * scale_x  
    v_orig4 = v4 * scale_y 

    u_orig5 = u5 * scale_x  
    v_orig5 = v5 * scale_y 



    p_cam1 = pixel_to_camera(u_orig, v_orig, Z, fx, fy, cx, cy)
    p_cam2 = pixel_to_camera(u_orig2, v_orig2, z2, fx, fy, cx, cy)
    p_cam3 = pixel_to_camera(u_orig3, v_orig3, z3, fx, fy, cx, cy)
    p_cam4 = pixel_to_camera(u_orig4, v_orig4, z4, fx, fy, cx, cy)
    p_cam5 = pixel_to_camera(u_orig5, v_orig5, z5, fx, fy, cx, cy)

    p_base = transform_point_to_base(p_cam1, T_cam_base)
    print("point's location:", p_base)
    p_base_orientation = [2.408, -1.064, -1.120]#rotation vector[rad] 
    r1 = R.from_rotvec(p_base_orientation)
    rotation_matrix = r1.as_matrix() 
    target_position, target_orientation = build_to_matrix(p_base, rotation_matrix, T_gripper2ee)
    r = R.from_matrix(target_orientation)
    rx, ry, rz = r.as_rotvec()
    x, y, z = target_position

    rx11, ry11, rz11 = (0.695, -2.157, -0.592)
    x11, y11, z11 =  (-0.28438, 0.34503, 0.47433)

    rx10, ry10, rz10 = (0.633, -2.149, -0.635)
    x10, y10, z10 = (-0.21565, 0.30820, 0.50366)

    p_base2 = transform_point_to_base(p_cam2, T_cam_base)
    print("point's location2:", p_base2)
    p_base_orientation2 = [0.652, -2.166, -0.485]#rotation vector[rad] 
    r2 = R.from_rotvec(p_base_orientation2)
    rotation_matrix2 = r2.as_matrix()
    target_position2, target_orientation2 = build_to_matrix(p_base2, rotation_matrix2, T_gripper2ee)
    r2 = R.from_matrix(target_orientation2)
    rx2, ry2, rz2 = r2.as_rotvec()
    x2, y2, z2 = target_position2

    rx21, ry21, rz21 = (0.652, -2.140, -0.610)
    x21, y21, z21 = (-0.24971, 0.42812, 0.38848)

    rx20, ry20, rz20 = (0.604, -2.125, -0.690)
    x20, y20, z20 = (-0.18518, 0.38567, 0.41126)

    p_base3 = transform_point_to_base(p_cam3, T_cam_base)
    print("point's location3:", p_base3)
    p_base_orientation3 = [0.643, -2.138, -0.502]#rotation vector[rad] 
    r3 = R.from_rotvec(p_base_orientation3)
    rotation_matrix3 = r3.as_matrix()
    target_position3, target_orientation3 = build_to_matrix(p_base3, rotation_matrix3, T_gripper2ee)
    r3 = R.from_matrix(target_orientation3)
    rx3, ry3, rz3 = r3.as_rotvec()
    x3, y3, z3 = target_position3

    rx31, ry31, rz31 = (0.806, -2.221, -0.401)
    x31, y31, z31 = (-0.25945, 0.35958, 0.21835)

    rx30, ry30, rz30 = (0.771, -2.153, -0.553)
    x30, y30, z30 = (-0.19951, 0.32539, 0.24125)

    p_base4 = transform_point_to_base(p_cam4, T_cam_base)
    print("point's location4:", p_base3)
    p_base_orientation4 = [0.973,-2.19,-0.218]#rotation vector[rad] 
    r4 = R.from_rotvec(p_base_orientation4)
    rotation_matrix4 = r4.as_matrix()
    target_position4, target_orientation4 = build_to_matrix(p_base4, rotation_matrix4, T_gripper2ee)
    r4 = R.from_matrix(target_orientation4)
    rx4, ry4, rz4 = r4.as_rotvec()
    x4, y4, z4 = target_position4

    rx41, ry41, rz41 = (0.458, -2.170, -0.735)
    x41, y41, z41 = (-0.31499, 0.31164, 0.37086)

    rx40, ry40, rz40 = (0.439, -2.122, -0.781)
    x40, y40, z40 = (-0.25209, 0.28456, 0.39585)

    p_base5 = transform_point_to_base(p_cam5, T_cam_base)
    print("point's location5:", p_base5)
    p_base_orientation5 = [0.75,-2.465,-0.406]#rotation vector[rad] 
    r5 = R.from_rotvec(p_base_orientation5)
    rotation_matrix5 = r5.as_matrix()
    target_position5, target_orientation5 = build_to_matrix(p_base5, rotation_matrix5, T_gripper2ee)
    r5 = R.from_matrix(target_orientation5)
    rx5, ry5, rz5 = r5.as_rotvec()
    x5, y5, z5 = target_position5

    rx51, ry51, rz51 = (0.724, -2.178, -0.666)
    x51, y51, z51 = (-0.28081, 0.34543, 0.30180)

    rx50, ry50, rz50 = (0.683, -2.128, -0.753)
    x50, y50, z50 = (-0.20387, 0.30237, 0.33217)
    
    #ur_script = f"""
    #pose_target = p[{x}, {y}, {z}, {rx}, {ry}, {rz}]
    #joint_target = get_inverse_kin(pose_target)
    #movej(joint_target, a=1.0, v=0.5)
    #"""

    target_joint_deg = [119.61, -82.24, -149.62, -41.04, 90.00, 8.69]
    target_joint_rad = [math.radians(j) for j in target_joint_deg]
    target_joint_deg2 = [109.82, -63.97, -121.62, -84.33, 90.00, 8.69]
    target_joint_rad2 = [math.radians(j) for j in target_joint_deg2]
    target_joint_deg3 = [160.01, -63.97, -121.62, -84.33, 90.00, 8.69]
    target_joint_rad3 = [math.radians(j) for j in target_joint_deg3]
    target_joint_deg4 = [160.19, -28.82, -121.27, -46.29, 90.00, -63.69]
    target_joint_rad4 = [math.radians(j) for j in target_joint_deg4]
    # target_joint_deg6 = [137.36, -91.53, -106.43, -21.37, 108.45, 196.93]
    target_joint_deg5 = [152.89, -93.72, -67.27, -24.41, 96.88, -63.02]
    target_joint_rad5 = [math.radians(j) for j in target_joint_deg5]
    # target_joint_deg7 = [141.74, -99.61, -116.31, 33.60, 108.33, 203.17]
    # target_joint_rad7 = [math.radians(j) for j in target_joint_deg7]
    

    #DO_ID = 0      
    #DO_PULSE = 1.0 

    ur_script = f"""


def multi_move():
    # 起始回位


    target_joint4 = [{target_joint_rad4[0]}, {target_joint_rad4[1]}, {target_joint_rad4[2]}, {target_joint_rad4[3]}, {target_joint_rad4[4]}, {target_joint_rad4[5]}]
    movej(target_joint4, a=1.0, v=0.5)
    sleep(1.0)






end

multi_move()
"""
    HOST = "192.168.56.101"
    PORT = 30002          # 默认端口 30002 
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.settimeout(8) # 设置5秒超时，避免卡死
        s.connect((HOST, PORT))
        print("连接成功！")
        
        #target_degrees = [322.62,-68.04, -131.59, 13.08, 98.17, 344]
        #target_radians = [math.radians(angle) for angle in target_degrees]
        #command = f"movej({target_radians}, a=1.0, v=0.5)\n".encode()
        print("发送指令:", ur_script.strip())
        #s.send(ur_script.encode())
        #s.send((ur_script + "\n").encode())
        ur_script = ur_script.strip() + "\n\n"
        s.send(ur_script.encode("utf-8"))
        
        #print("发送指令:", command.decode().strip())
        #s.send(command)
        
        response = s.recv(1024)  # 尝试接收UR的响应
        print("机器人响应:", response)

    except socket.timeout:
        print("错误：连接超时，检查IP/端口或网络！")
    except ConnectionRefusedError:
        print("错误：连接被拒绝，检查UR是否开放端口！")
    except Exception as e:
        print("未知错误:", e)
    finally:
        s.close()
        print("连接关闭")