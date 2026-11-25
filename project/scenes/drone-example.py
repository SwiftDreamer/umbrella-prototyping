"""Crazyflie climbing from 0.5 m to a 1 m hover under a scripted altitude controller.

    ./simproto render project/scenes/drone-example.py --seconds 4

The env is the physics and the action interface (collective thrust in newtons); the
controller is `policy`, outside it, which is the slot a trained RL policy takes later.
Like Isaac Lab's quadcopter tasks, flight is a force on the `body` link, not simulated
propeller aerodynamics; the rotor joints only spin for looks.
"""

from collections.abc import Sequence

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.assets import ArticulationCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils import replace
from isaaclab.visualizers import VisualizerCfg
from isaaclab_assets.robots.quadcopter import CRAZYFLIE_CFG

HOVER_HEIGHT = 1.0  # m
GRAVITY = 9.81


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0)
    )
    # {ENV_REGEX_NS} puts one drone in each cloned env
    robot: ArticulationCfg = replace(CRAZYFLIE_CFG, prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class EnvCfg(DirectRLEnvCfg):
    decimation = 2  # physics steps per env step: actions at 60 Hz, physics at 120 Hz
    episode_length_s = 10.0
    action_space = 1  # collective thrust [N]
    observation_space = 2  # height [m], vertical velocity [m/s]
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    scene: InteractiveSceneCfg = SceneCfg(num_envs=1, env_spacing=2.0)

    def __post_init__(self):
        # the default view of any visualizer, including the one render records with;
        # a 24 mm lens frames the 9 cm drone and its 0.5 m climb
        self.sim.default_visualizer_cfg = VisualizerCfg(
            eye=(1.0, -1.0, 1.1), lookat=(0.0, 0.0, 0.75), focal_length=24.0
        )


class Env(DirectRLEnv):
    cfg: EnvCfg

    def __init__(self, cfg: EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot: Articulation = self.scene["robot"]
        body_ids, _ = self.robot.find_bodies("body")
        self.body_ids = torch.tensor(body_ids, device=self.device)
        # every link's mass, since the force on the body lifts the rotors too
        self.mass = self.robot.data.body_mass.torch.sum(dim=-1).to(self.device)
        self.force = torch.zeros(self.num_envs, 1, 3, device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # once per env step; thrust can't pull downward
        self.force[:, 0, 2] = actions[:, 0].clamp(min=0.0)

    def _apply_action(self) -> None:
        # before every physics step; body frame, which is world-up while level
        self.robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self.force, body_ids=self.body_ids
        )

    def _get_observations(self) -> dict:
        height = self.robot.data.root_pos_w.torch[:, 2:3]
        climb_rate = self.robot.data.root_lin_vel_w.torch[:, 2:3]
        return {"policy": torch.cat((height, climb_rate), dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        # a render scene, not a task yet; training needs real reward terms here
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length
        return torch.zeros_like(time_out), time_out

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        # back to the configured start: 0.5 m up, at rest, rotors spinning
        pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=self.robot.data.default_root_vel.torch[env_ids],
            env_ids=env_ids,
        )
        self.robot.write_joint_position_to_sim_index(
            position=self.robot.data.default_joint_pos.torch[env_ids], env_ids=env_ids
        )
        self.robot.write_joint_velocity_to_sim_index(
            velocity=self.robot.data.default_joint_vel.torch[env_ids], env_ids=env_ids
        )


def policy(env: Env, obs: dict[str, torch.Tensor]) -> torch.Tensor:
    """PD on height around the hover thrust m*g."""
    height, climb_rate = obs["policy"].unbind(dim=-1)
    # altitude controller equation so that the drone can float
    acceleration = GRAVITY + 4 * (HOVER_HEIGHT - height) - 2.5 * climb_rate
    return (env.mass * acceleration).unsqueeze(-1)
