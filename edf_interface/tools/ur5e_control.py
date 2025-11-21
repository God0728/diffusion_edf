import math
from edf_interface.modules.robot import RobotInterface as RI
def main():
    # target_joint_deg = [189.20, -61.79, -86.89, -128.12, 91.40, 137.29]
    target_joint_deg = [189.20, -61.79, -86.89, -128.12, 91.40, 317.29]
    #target_joint_deg= [251.26, -73.91, 39.62, -239.84, 87.47, 191.20]
    target_joint_rad = [math.radians(j) for j in target_joint_deg]
    robot = RI()
    
    robot.move_to_joint_rad(joint_rad=target_joint_rad)

if __name__ == "__main__":
    main()