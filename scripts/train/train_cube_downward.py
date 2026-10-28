"""Train the fixed palm-down cube Teacher: first hold, then reorient."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.lab_api.math import quat_inv, quat_mul
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends
from wuji_mjlab.rl.runner import WujiOnPolicyRunner
from wuji_mjlab.tasks.reorient import mdp
from wuji_mjlab.assets.objects.inhand_object.object_cfg import get_inhand_object_cfg
from wuji_mjlab.tasks.reorient.config.wuji_hand.env_cfgs import (
  wuji_hand_cube_downward_env_cfg,
)
from wuji_mjlab.tasks.reorient.config.wuji_hand.rsl_rl.ppo import (
  wuji_hand_reorient_ppo_runner_cfg,
)


def center_reward(
  env, target_pos: tuple[float, float, float] = (0.105, 0.009, 0.470)
) -> torch.Tensor:
  """Keep the cube near its initial palm-relative center in world coordinates."""
  cube_pos = env.scene["object"].data.root_link_pos_w - env.scene.env_origins
  target = cube_pos.new_tensor(target_pos)
  distance_sq = ((cube_pos - target) ** 2).sum(dim=-1)
  return torch.exp(-distance_sq / (2.0 * 0.035**2))


def alive_reward(env) -> torch.Tensor:
  return torch.ones(env.num_envs, device=env.device)


def goal_directed_rotation_rate(env) -> torch.Tensor:
  """Reward world angular velocity that reduces the current SO(3) error.

  The quaternion residual ``goal * inverse(object)`` gives the shortest-path
  rotation axis in world coordinates. Projecting object angular velocity onto
  that axis rewards motion toward the sampled goal and penalizes motion away
  from it. It is a progress aid; orientation_alignment remains the actual
  pose objective and hold_escalation rewards staying inside its success band.
  """
  cube = env.scene["object"]
  goal = env.command_manager.get_term("reorient_command").goal_quat
  error_quat = quat_mul(goal, quat_inv(cube.data.root_link_quat_w))
  error_quat = torch.where(error_quat[:, :1] < 0.0, -error_quat, error_quat)
  axis = error_quat[:, 1:4]
  axis_norm = axis.norm(dim=-1, keepdim=True)
  axis = axis / axis_norm.clamp_min(1.0e-6)
  directed_rate = (cube.data.root_link_ang_vel_w * axis).sum(dim=-1)
  return torch.where(
    axis_norm.squeeze(-1) > 0.1, directed_rate.clamp(-2.0, 2.0), 0.0
  )


def spin_rate(env) -> torch.Tensor:
  target_rate = 1.5  # rad/s about the palm normal
  rate = env.scene["object"].data.root_link_ang_vel_w[:, 2]
  contact_count = mdp.fingertip_contact_count(
    env, SceneEntityCfg("tip_object_contact")
  )
  contact_gate = (contact_count / 3.0).clamp(0.0, 1.0)
  return (rate / target_rate).clamp(-1.0, 1.0) * contact_gate


def spin_tilt_rate(env) -> torch.Tensor:
  angular_velocity = env.scene["object"].data.root_link_ang_vel_w
  return angular_velocity[:, :2].square().sum(dim=-1).clamp_max(4.0)


def make_cfg(
  phase: str,
  num_envs: int,
  seed: int,
  goal_angle: float | None = None,
  official_actor: bool = False,
  gravity_scale: float = 1.0,
  cube_edge_m: float = 0.054,
  cube_mass_kg: float | None = None,
  wrist_assembly: bool = False,
):
  cube_palm_offset = cube_edge_m / 2.0 + 0.003
  cfg = wuji_hand_cube_downward_env_cfg(
    num_envs=num_envs,
    official_actor_compatible=official_actor,
    cube_edge_m=cube_edge_m,
    wrist_assembly=wrist_assembly,
    object_palm_pos=(cube_palm_offset, 0.009, 0.105),
  )
  cfg.seed = seed
  if cube_mass_kg is not None:
    obj = cfg.scene.entities["object"]
    obj.spec_fn = get_inhand_object_cfg(cube_edge_m, cube_mass_kg).spec_fn
  cfg.sim.mujoco.gravity = (0.0, 0.0, -9.81 * gravity_scale)
  if goal_angle is not None:
    cfg.commands["reorient_command"].max_goal_angle_rad = goal_angle
  cfg.events = {name: term for name, term in cfg.events.items() if term.mode == "reset"}
  cfg.curriculum.clear()
  cfg.observations["policy"].enable_corruption = False
  cfg.terminations["cage_drop"].params["max_outside_steps"] = 14
  cfg.rewards["cage_escape"].weight = -750.0
  cfg.rewards["palm_detach"].weight = 0.0
  cfg.rewards["action_rate"].weight = -0.2
  spin_explore = phase == "spin" and gravity_scale < 1.0
  if phase == "rotate":
    # Keep the three task objectives explicit: reach the full-pose goal,
    # move along its shortest rotation direction, then hold it before advance.
    cfg.rewards["orientation_alignment"].weight = 40.0
    cfg.rewards["hold_escalation"].weight = 25.0
    cfg.rewards["directed_rotation"] = RewardTermCfg(
      func=goal_directed_rotation_rate, weight=5.0
    )
  elif phase == "spin":
    cfg.rewards.pop("orientation_alignment")
    cfg.rewards.pop("hold_escalation")
    cfg.rewards["spin_rate"] = RewardTermCfg(
      func=spin_rate, weight=100.0 if spin_explore else 35.0
    )
    cfg.rewards["spin_tilt_rate"] = RewardTermCfg(
      func=spin_tilt_rate, weight=-1.0 if spin_explore else -2.0
    )
    if spin_explore:
      cfg.episode_length_s = 10.0
      cfg.terminations["cage_drop"].params["max_outside_steps"] = 30
      cfg.rewards["cage_escape"].weight = -150.0
      cfg.rewards["action_rate"].weight = 0.0
      cfg.rewards["torque"].weight = -5.0
  cfg.rewards["center"] = RewardTermCfg(
    func=center_reward,
    weight=15.0 if phase == "hold" else 1.0 if spin_explore else 5.0 if phase == "spin" else 0.5,
    params={"target_pos": (0.105, 0.009, 0.5 - cube_palm_offset)},
  )
  cfg.rewards["alive"] = RewardTermCfg(
    func=alive_reward, weight=0.5 if spin_explore else 2.0
  )
  cfg.rewards["fingertip_contact"] = RewardTermCfg(
    func=mdp.fingertip_contact_count,
    weight=0.5 if phase == "hold" else 0.0 if spin_explore else 0.5,
    params={"sensor_cfg": SceneEntityCfg("tip_object_contact")},
  )
  if phase == "hold":
    cfg.rewards.pop("orientation_alignment")
    cfg.rewards.pop("hold_escalation")
  return cfg


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--phase", choices=("hold", "rotate", "spin"), required=True)
  parser.add_argument("--resume", type=Path)
  parser.add_argument("--num-envs", type=int, default=2048)
  parser.add_argument("--iterations", type=int, default=1000)
  parser.add_argument("--seed", type=int, default=84)
  parser.add_argument(
    "--goal-angle", type=float, help="Maximum relative goal angle in radians; default samples full SO(3)"
  )
  parser.add_argument("--official-actor", action="store_true")
  parser.add_argument("--gravity-scale", type=float, default=1.0)
  parser.add_argument("--cube-edge-m", type=float, default=0.054)
  parser.add_argument("--cube-mass-kg", type=float)
  parser.add_argument("--wrist-assembly", action="store_true", help="Include extracted flange, docking and D405 camera")
  parser.add_argument("--output", type=Path)
  parser.add_argument("--smoke", action="store_true")
  args = parser.parse_args()
  if args.smoke:
    args.num_envs, args.iterations = 32, 2
  if args.num_envs <= 0 or args.iterations <= 0:
    parser.error("num-envs and iterations must be positive")
  if args.phase in ("rotate", "spin") and args.resume is None:
    parser.error("rotation phase requires a hold checkpoint via --resume")
  if args.goal_angle is not None and not (0.4 < args.goal_angle <= 3.141592653589793):
    parser.error("goal-angle must be greater than 0.4 and at most pi")
  if not (0.0 < args.gravity_scale <= 1.0):
    parser.error("gravity-scale must be in (0, 1]")
  if not (0.03 <= args.cube_edge_m <= 0.054):
    parser.error("cube-edge-m must be between 0.030 and 0.054 meters")
  if args.resume is not None and not args.resume.is_file():
    parser.error(f"Checkpoint not found: {args.resume}")
  if not torch.cuda.is_available():
    parser.error("Training requires CUDA")

  configure_torch_backends()
  output = args.output or (
    Path("logs/rsl_rl/wuji_cube_downward")
    / f"{args.phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
  )
  output.mkdir(parents=True, exist_ok=False)
  (output / "launch.json").write_text(json.dumps(vars(args), default=str, indent=2))
  cfg = make_cfg(
    args.phase, args.num_envs, args.seed, args.goal_angle,
    args.official_actor, args.gravity_scale, args.cube_edge_m, args.cube_mass_kg, args.wrist_assembly,
  )
  agent = wuji_hand_reorient_ppo_runner_cfg(
    run_name=f"CubeDownward{args.phase.title()}", max_iterations=args.iterations
  )
  agent.experiment_name = "wuji_cube_downward"
  agent.logger = "tensorboard"
  agent.upload_model = False
  agent.seed = args.seed
  agent.algorithm.num_mini_batches = min(32, args.num_envs)
  if args.phase == "rotate":
    agent.algorithm.learning_rate = 1.0e-4
    agent.algorithm.entropy_coef = 1.0e-3
  elif args.phase == "spin":
    agent.algorithm.learning_rate = 1.0e-5
    agent.algorithm.entropy_coef = 1.0e-3
  if args.smoke:
    agent.num_steps_per_env = 8
    agent.algorithm.num_learning_epochs = 1
  dump_yaml(output / "params/env.yaml", asdict(cfg))
  dump_yaml(output / "params/agent.yaml", asdict(agent))

  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    obs, _ = env.reset()
    expected_policy = 207 if args.official_actor else 413
    if obs["policy"].shape != (args.num_envs, expected_policy) or obs["critic"].shape != (args.num_envs, 413):
      raise RuntimeError(f"Unexpected observation shapes: {obs['policy'].shape}, {obs['critic'].shape}")
    if not all(torch.isfinite(value).all() for value in obs.values()):
      raise RuntimeError("Nonfinite initial observation")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = WujiOnPolicyRunner(wrapped, asdict(agent), str(output), "cuda:0")
    if args.resume is not None:
      runner.load(
        str(args.resume),
        load_cfg={
          "actor": True,
          "critic": True,
          "optimizer": False,
          "iteration": False,
          "rnd": False,
        },
        map_location="cuda:0",
      )
    runner.learn(args.iterations, init_at_random_ep_len=False)
    runner.save(str(output / "final.pt"))
  finally:
    env.close()
  (output / "result.json").write_text(
    json.dumps({"phase": args.phase, "checkpoint": str((output / "final.pt").resolve()), "source": str(args.resume.resolve()) if args.resume else None}, indent=2)
  )


if __name__ == "__main__":
  main()
