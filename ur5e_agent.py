from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EDF_INTERFACE_DIR = PROJECT_ROOT / "edf_interface"
EXAMPLES_DIR = EDF_INTERFACE_DIR / "examples"
for candidate in (EDF_INTERFACE_DIR, EXAMPLES_DIR):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch
import yaml

from edf_interface import data
try:
    from edf_interface.examples.env_server import EnvService
except ModuleNotFoundError:
    from env_server import EnvService
from edf_interface.utils.manipulation_utils import (
    compute_pre_pick_trajectories,
    compute_pre_place_trajectories,
)
from diffusion_edf.agent import DiffusionEdfAgent


class LocalAgentPipeline:
    """Wraps pick/place Diffusion-EDF agents for direct invocation."""

    def __init__(self, configs_root: Path, compile_score_head: bool = False) -> None:
        self.configs_root = configs_root
        self.compile_score_head = compile_score_head
        self._load_configs()
        self._build_agents()

    def _load_configs(self) -> None:
        with open(self.configs_root / "agent.yaml") as fp:
            agent_cfg = yaml.safe_load(fp)
        with open(self.configs_root / "preprocess.yaml") as fp:
            preprocess_cfg = yaml.safe_load(fp)
        with open(self.configs_root / "server.yaml") as fp:
            server_cfg = yaml.safe_load(fp)

        self.device: str = agent_cfg["device"]
        self.pick_model_cfgs: List[Dict] = agent_cfg["model_kwargs"]["pick_models_kwargs"]
        self.place_model_cfgs: List[Dict] = agent_cfg["model_kwargs"]["place_models_kwargs"]
        self.pick_critic_cfg: Dict | None = agent_cfg["model_kwargs"].get("pick_critic_kwargs")
        self.place_critic_cfg: Dict | None = agent_cfg["model_kwargs"].get("place_critic_kwargs")

        self.preprocess_sequence: List[Dict] = preprocess_cfg["preprocess_config"]
        self.unprocess_sequence: List[Dict] = preprocess_cfg["unprocess_config"]

        self.pick_diffusion_cfg = server_cfg["pick_diffusion_configs"]
        self.place_diffusion_cfg = server_cfg["place_diffusion_configs"]
        self.pick_traj_cfg = server_cfg["pick_trajectory_configs"]
        self.place_traj_cfg = server_cfg["place_trajectory_configs"]

    def _build_agents(self) -> None:
        common_kwargs = dict(
            preprocess_config=self.preprocess_sequence,
            unprocess_config=self.unprocess_sequence,
            device=self.device,
            compile_score_head=self.compile_score_head,
        )
        self.pick_agent = DiffusionEdfAgent(
            model_kwargs_list=self.pick_model_cfgs,
            critic_kwargs=self.pick_critic_cfg,
            **common_kwargs,
        )
        self.place_agent = DiffusionEdfAgent(
            model_kwargs_list=self.place_model_cfgs,
            critic_kwargs=self.place_critic_cfg,
            **common_kwargs,
        )

    def _select_agent(self, task: str) -> Tuple[DiffusionEdfAgent, Dict, Dict]:
        if task == "pick":
            return self.pick_agent, self.pick_diffusion_cfg, self.pick_traj_cfg
        if task == "place":
            return self.place_agent, self.place_diffusion_cfg, self.place_traj_cfg
        raise ValueError(f"Unsupported task '{task}'. Use 'pick' or 'place'.")

    def denoise_sequences(
        self,
        scene_pcd: data.PointCloud,
        grasp_pcd: data.PointCloud,
        current_poses: data.SE3,
        task: str,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        agent, diffusion_cfg, _ = self._select_agent(task)
        result, _, _, info = agent.sample(
            scene_pcd=scene_pcd.to(self.device),
            grasp_pcd=grasp_pcd.to(self.device),
            Ts_init=current_poses.to(self.device),
            N_steps_list=diffusion_cfg["N_steps_list"],
            timesteps_list=diffusion_cfg["timesteps_list"],
            temperatures_list=diffusion_cfg["temperatures_list"],
            diffusion_schedules_list=diffusion_cfg["diffusion_schedules_list"],
            log_t_schedule=diffusion_cfg["log_t_schedule"],
            time_exponent_temp=diffusion_cfg["time_exponent_temp"],
            time_exponent_alpha=diffusion_cfg["time_exponent_alpha"],
            return_info=True,
        )
        return result.detach().cpu(), {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in info.items()}

    def final_poses(
        self,
        denoise_seq: torch.Tensor,
        task: str,
    ) -> data.SE3:
        agent, _, _ = self._select_agent(task)
        final = data.SE3(poses=denoise_seq[-1])
        return agent.unprocess_fn(final)

    def request_trajectories(
        self,
        scene_pcd: data.PointCloud,
        grasp_pcd: data.PointCloud,
        current_poses: data.SE3,
        task: str,
    ) -> Tuple[List[data.SE3], Dict[str, torch.Tensor]]:
        denoise_seq, info = self.denoise_sequences(scene_pcd, grasp_pcd, current_poses, task)
        final_Ts = self.final_poses(denoise_seq, task)
        if task == "pick":
            trajectories = compute_pre_pick_trajectories(pick_poses=final_Ts, **self.pick_traj_cfg)
        else:
            trajectories = compute_pre_place_trajectories(
                place_poses=final_Ts,
                scene_pcd=scene_pcd,
                grasp_pcd=grasp_pcd,
                **self.place_traj_cfg,
            )
        return trajectories, info


def execute_trajectory(
    env: EnvService,
    trajectory: data.SE3,
    velocity: float,
    acceleration: float,
    wait: bool,
) -> None:
    """Stream each pose in ``trajectory`` to the real robot."""

    pose = trajectory.poses[-1:]
    single_step = data.SE3(poses=pose) 
    env.move_se3(single_step, velocity=velocity, acceleration=acceleration, wait=wait)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pick/place locally without Pyro")
    parser.add_argument("--configs-root-dir", type=Path, required=True, help="Path to task config folder (e.g. configs/panda_bottle)")
    parser.add_argument("--compile-score-model-head", action="store_true", help="JIT compile score head for faster inference")
    parser.add_argument("--move-velocity", type=float, default=0.2, help="Joint move velocity")
    parser.add_argument("--move-acceleration", type=float, default=1.0, help="Joint move acceleration")
    parser.add_argument("--no-wait", action="store_true", help="Do not block between intermediate moves")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env_service = EnvService()


    def acquire_pointclouds() -> Tuple[data.PointCloud, data.PointCloud]:
        scene = env_service.observe_scene()
        grasp = env_service.observe_grasp()
        return scene, grasp

    scene_pcd, grasp_pcd = acquire_pointclouds()
    agent_pipeline = LocalAgentPipeline(
        configs_root=args.configs_root_dir,
        compile_score_head=args.compile_score_model_head,
    )
    for task in ("pick",):
        current_pose = env_service.get_current_poses()
        # if task == "place" :
        #     scene_pcd, grasp_pcd = acquire_pointclouds()

        trajectories, info = agent_pipeline.request_trajectories(
            scene_pcd=scene_pcd,
            grasp_pcd=grasp_pcd,
            current_poses=current_pose,
            task=task,
        )
        print(f"[{task}] planned {len(trajectories)} trajectories.")
        print(trajectories)
        if not trajectories:
            raise RuntimeError(f"Agent returned zero trajectories for task '{task}'.")

        primary_traj = trajectories[0]
        execute_trajectory(
            env=env_service,
            trajectory=primary_traj,
            velocity=args.move_velocity,
            acceleration=args.move_acceleration,
            wait=not args.no_wait,
        )

        print(f"[{task}] executed {primary_traj.poses.shape[0]} waypoints | info keys: {list(info.keys())}")


if __name__ == "__main__":
    main()