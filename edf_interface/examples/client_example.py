from edf_interface.data import SE3, PointCloud
from edf_interface.pyro import PyroClientBase
from typing import Optional, Dict, Any, Tuple, List

class RobotClient(PyroClientBase):
    def __init__(self, env_server_name: str = 'env',
                 agent_sever_name: str = 'agent'):
        super().__init__(service_names=[env_server_name, agent_sever_name])
    def get_current_poses(self, **kwargs) -> SE3: ... 
    
    def observe_scene(self, **kwargs) -> PointCloud: ...
    
    def observe_grasp(self, **kwargs) -> PointCloud: ...

    def move_se3(self, target_poses: SE3, **kwargs) -> bool: ...

    def request_trajectories(self, **kwargs
                                ) -> Tuple[List[data.SE3], Dict[str, Any]]:...
        
        
def main():
    
    client = RobotClient(env_server_name='env', agent_sever_name='agent')
    
    scene_pcd = client.observe_scene()
    grasp_pcd = client.observe_grasp()
    
    current_poses = client.get_current_poses()
    print(f"当前机械臂位姿: {current_poses}")
    trajectories, info = client.request_trajectories(
        scene_pcd=scene_pcd,
        grasp_pcd=grasp_pcd,
        current_poses=current_poses,
        task_name="pick"
    )
    
    best_pose = trajectories[0][0]  
    success = client.move_se3(best_pose)

    
    print(f"✓ 抓取任务{'成功' if success else '失败'}!")

    scene_pcd = client.observe_scene()
    grasp_pcd = client.observe_grasp()
    
    current_poses = client.get_current_poses()
    
    trajectories, info = client.request_trajectories(
        scene_pcd=scene_pcd,
        grasp_pcd=grasp_pcd,
        current_poses=current_poses,
        task_name="place"
    )
    
    best_pose = trajectories[0][0]  
    success = client.move_se3(best_pose)

    print(f"✓ 放置任务{'成功' if success else '失败'}!")


if __name__ == "__main__":
    main()