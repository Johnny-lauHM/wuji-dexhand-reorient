"""Evaluate a privileged-state dodecahedron policy at a fixed wrist pitch."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.models import MLPModel
from tensordict import TensorDict
from wuji_mjlab.tasks.dodecahedron.env_cfg import (
  wuji_hand_dodecahedron_env_cfg,
  wuji_hand_dodecahedron_ppo_cfg,
)
from wuji_mjlab.tasks.reorient import mdp


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--wrist-pitch", type=float, default=90.0)
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--seconds", type=float, default=30.0)
  parser.add_argument("--seed", type=int, default=321)
  parser.add_argument("--grip-offset", type=float, default=0.0)
  parser.add_argument("--downward-grasp", action="store_true")
  parser.add_argument("--object-palm-x", type=float, default=0.060)
  parser.add_argument("--object-palm-y", type=float, default=0.010)
  parser.add_argument("--object-palm-z", type=float, default=0.140)
  args = parser.parse_args()
  if not args.checkpoint.is_file():
    parser.error(f"Checkpoint not found: {args.checkpoint}")
  if args.num_envs <= 0 or args.seconds <= 0:
    parser.error("num-envs and seconds must be positive")
  args.output.mkdir(parents=True, exist_ok=False)
  configure_torch_backends()

  cfg = wuji_hand_dodecahedron_env_cfg(
    num_envs=args.num_envs,
    sim2real_domain_randomization=False,
    wrist_pitch_deg=args.wrist_pitch,
    grip_offset_rad=args.grip_offset,
    downward_grasp=args.downward_grasp,
    object_palm_x_m=args.object_palm_x,
    object_palm_y_m=args.object_palm_y,
    object_palm_z_m=args.object_palm_z,
  )
  cfg.seed = args.seed
  cfg.events["reset_robot_joints"].params["position_range"] = {".*": (0.0, 0.0)}
  cfg.events["reset_object_pose"].params["pos_noise"] = 0.0015
  cfg.terminations["cage_drop"].params["max_outside_steps"] = 14
  cfg.rewards["cage_escape"].weight = -750.0
  cfg.rewards["palm_detach"].weight = 2.0
  cfg.rewards["fingertip_contact"] = RewardTermCfg(
    func=mdp.fingertip_contact_count,
    weight=2.0,
    params={"sensor_cfg": SceneEntityCfg("tip_object_contact")},
  )
  agent = wuji_hand_dodecahedron_ppo_cfg()
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    obs, _ = wrapped.reset()
    actor_cfg = asdict(agent.actor)
    for key in ("class_name", "cnn_cfg", "rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
      actor_cfg.pop(key, None)
    template = TensorDict(
      {"policy": torch.zeros_like(obs["policy"])}, batch_size=[args.num_envs]
    )
    policy = MLPModel(
      template, {"policy": ["policy"]}, "policy", wrapped.num_actions, **actor_cfg
    ).to("cuda:0")
    checkpoint = torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)
    policy.load_state_dict(checkpoint["actor_state_dict"])
    policy.eval()

    command = env.command_manager.get_term("reorient_command")
    active = torch.ones(args.num_envs, dtype=torch.bool, device="cuda:0")
    successes = torch.zeros_like(active)
    goal_counts = torch.zeros(args.num_envs, dtype=torch.int32, device="cuda:0")
    first_success = torch.full_like(goal_counts, -1)
    termination_step = torch.full_like(goal_counts, -1)
    steps = int(args.seconds / env.step_dt)
    with torch.inference_mode():
      for step in range(steps):
        action = policy(TensorDict({"policy": obs["policy"]}, batch_size=[args.num_envs]))
        obs, _, dones, _ = wrapped.step(action)
        goal_counts += (active & command.success_achieved).int()
        new_success = active & (command.goal_reach_count > 0) & ~successes
        first_success[new_success] = step + 1
        successes |= new_success
        ended = active & dones.bool()
        termination_step[ended] = step + 1
        active &= ~dones.bool()

    success_times = first_success[first_success > 0].float() * env.step_dt
    report = {
      "checkpoint": str(args.checkpoint.resolve()),
      "observation_source": "privileged object state + proprioception",
      "wrist_pitch_deg": args.wrist_pitch,
      "grip_offset_rad": args.grip_offset,
      "downward_grasp": args.downward_grasp,
      "object_palm_x_m": args.object_palm_x,
      "object_palm_y_m": args.object_palm_y,
      "object_palm_z_m": args.object_palm_z,
      "num_envs": args.num_envs,
      "horizon_s": args.seconds,
      "first_goal_successes": int(successes.sum()),
      "first_goal_success_rate": float(successes.float().mean()),
      "mean_first_success_s": float(success_times.mean()) if success_times.numel() else None,
      "survived_horizon": int(active.sum()),
      "survival_rate": float(active.float().mean()),
      "total_goals_before_termination": int(goal_counts.sum()),
      "mean_goals_per_trial": float(goal_counts.float().mean()),
      "termination_times_s": [
        float(value * env.step_dt) if value > 0 else None
        for value in termination_step.cpu().tolist()
      ],
    }
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
