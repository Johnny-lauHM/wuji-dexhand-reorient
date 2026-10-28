"""Evaluate a fixed palm-down cube Teacher on survival and rotation goals."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.lab_api.math import quat_error_magnitude
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.models import MLPModel
from tensordict import TensorDict
from train.train_cube_downward import make_cfg
from wuji_mjlab.tasks.reorient.mdp.metrics import fingertip_contact_count
from wuji_mjlab.tasks.reorient.config.wuji_hand.rsl_rl.ppo import (
  wuji_hand_reorient_ppo_runner_cfg,
)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--phase", choices=("hold", "rotate", "spin"), required=True)
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--seconds", type=float, default=20.0)
  parser.add_argument("--goal-deadline-s", type=float, default=8.0)
  parser.add_argument("--seed", type=int, default=321)
  parser.add_argument("--goal-angle", type=float)
  parser.add_argument("--official-actor", action="store_true")
  parser.add_argument("--gravity-scale", type=float, default=1.0)
  parser.add_argument("--cube-edge-m", type=float, default=0.054)
  parser.add_argument("--cube-mass-kg", type=float)
  parser.add_argument("--wrist-assembly", action="store_true")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if not args.checkpoint.is_file():
    parser.error(f"Checkpoint not found: {args.checkpoint}")
  if args.num_envs <= 0 or args.seconds <= 0 or args.goal_deadline_s <= 0:
    parser.error("num-envs, seconds, and goal-deadline-s must be positive")
  if not (0.0 < args.gravity_scale <= 1.0):
    parser.error("gravity-scale must be in (0, 1]")
  args.output.mkdir(parents=True, exist_ok=False)
  configure_torch_backends()

  cfg = make_cfg(
    args.phase, args.num_envs, args.seed, args.goal_angle,
    args.official_actor, args.gravity_scale, args.cube_edge_m, args.cube_mass_kg, args.wrist_assembly,
  )
  agent = wuji_hand_reorient_ppo_runner_cfg()
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
    state = torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)
    policy.load_state_dict(state["actor_state_dict"])
    policy.eval()

    command = env.command_manager.get_term("reorient_command")
    active = torch.ones(args.num_envs, dtype=torch.bool, device="cuda:0")
    success = torch.zeros_like(active)
    first_success = torch.full((args.num_envs,), -1, dtype=torch.int32, device="cuda:0")
    termination_step = torch.full_like(first_success, -1)
    goal_counts = torch.zeros_like(first_success)
    goal_attempts = torch.ones_like(first_success)
    deadline_successes = torch.zeros_like(first_success)
    angle_bin_edges = torch.tensor(
      [torch.pi / 3.0, 2.0 * torch.pi / 3.0], device="cuda:0"
    )
    bin_attempts = torch.zeros(3, dtype=torch.int64, device="cuda:0")
    bin_deadline_successes = torch.zeros_like(bin_attempts)
    active_steps = torch.zeros(args.num_envs, dtype=torch.int32, device="cuda:0")
    fingertip_contact_steps = torch.zeros_like(active_steps)
    cube = env.scene["object"]
    start_pos = cube.data.root_link_pos_w.clone()
    start_quat = cube.data.root_link_quat_w.clone()
    initial_goal_error = quat_error_magnitude(start_quat, command.goal_quat)
    goal_bin_at_start = torch.bucketize(initial_goal_error, angle_bin_edges)
    for bin_idx in range(3):
      bin_attempts[bin_idx] = (goal_bin_at_start == bin_idx).int().sum()
    min_goal_error = initial_goal_error.clone()
    max_cube_rotation = torch.zeros(args.num_envs, device="cuda:0")
    max_drift = torch.zeros(args.num_envs, device="cuda:0")
    net_spin = torch.zeros(args.num_envs, device="cuda:0")
    steps = int(args.seconds / env.step_dt)
    with torch.inference_mode():
      for step in range(steps):
        was_active = active.clone()
        action = policy(TensorDict({"policy": obs["policy"]}, batch_size=[args.num_envs]))
        obs, _, dones, _ = wrapped.step(action)
        active_steps += was_active.int()
        in_contact = fingertip_contact_count(env) > 0
        fingertip_contact_steps += (was_active & in_contact).int()
        drift = (cube.data.root_link_pos_w - start_pos).norm(dim=-1)
        max_drift = torch.where(active, torch.maximum(max_drift, drift), max_drift)
        goal_error = quat_error_magnitude(cube.data.root_link_quat_w, command.goal_quat)
        min_goal_error = torch.where(
          active, torch.minimum(min_goal_error, goal_error), min_goal_error
        )
        cube_rotation = quat_error_magnitude(cube.data.root_link_quat_w, start_quat)
        max_cube_rotation = torch.where(
          active, torch.maximum(max_cube_rotation, cube_rotation), max_cube_rotation
        )
        net_spin += torch.where(
          active, cube.data.root_link_ang_vel_w[:, 2] * env.step_dt, 0.0
        )
        reached_now = was_active & command.success_achieved
        goal_counts += reached_now.int()
        within_deadline = command.goal_timer.float() * env.step_dt <= args.goal_deadline_s
        deadline_now = reached_now & within_deadline
        deadline_successes += deadline_now.int()
        for bin_idx in range(3):
          bin_deadline_successes[bin_idx] += (deadline_now & (goal_bin_at_start == bin_idx)).int().sum()
        switched = was_active & command.goal_switched
        goal_attempts += switched.int()
        if switched.any():
          new_error = quat_error_magnitude(
            cube.data.root_link_quat_w, command.goal_quat
          )
          new_bin = torch.bucketize(new_error, angle_bin_edges)
          goal_bin_at_start = torch.where(switched, new_bin, goal_bin_at_start)
          for bin_idx in range(3):
            bin_attempts[bin_idx] += (switched & (new_bin == bin_idx)).int().sum()
        reached = active & (command.goal_reach_count > 0) & ~success
        first_success[reached] = step + 1
        success |= reached
        ended = active & dones.bool()
        termination_step[ended] = step + 1
        active &= ~dones.bool()

    success_times = first_success[first_success > 0].float() * env.step_dt
    completed_horizon = active | (termination_step >= steps)
    report = {
      "checkpoint": str(args.checkpoint.resolve()),
      "phase": args.phase,
      "cube_edge_m": args.cube_edge_m,
      "cube_mass_kg": args.cube_mass_kg,
      "wrist_assembly": args.wrist_assembly,
      "observation_source": "privileged simulator state",
      "official_actor_compatible": args.official_actor,
      "wrist_pitch_deg": 90.0,
      "gravity_scale": args.gravity_scale,
      "goal_angle_rad": args.goal_angle,
      "goal_deadline_s": args.goal_deadline_s,
      "num_envs": args.num_envs,
      "horizon_s": args.seconds,
      "survived_horizon": int(completed_horizon.sum()),
      "survival_rate": float(completed_horizon.float().mean()),
      "first_goal_successes": int(success.sum()),
      "first_goal_success_rate": float(success.float().mean()),
      "mean_first_success_s": float(success_times.mean()) if success_times.numel() else None,
      "total_goals_before_termination": int(goal_counts.sum()),
      "goal_attempts_started": int(goal_attempts.sum()),
      "goal_deadline_successes": int(deadline_successes.sum()),
      "goal_deadline_success_rate": float(deadline_successes.sum().float() / goal_attempts.sum().clamp_min(1)),
      "mean_goals_reached_per_survived_minute": float(
        goal_counts.sum().float() / (active_steps.sum().float() * env.step_dt / 60.0).clamp_min(1e-6)
      ),
      "goal_deadline_success_by_initial_error_rad": {
        "0_to_60_deg": {
          "attempts": int(bin_attempts[0]),
          "successes": int(bin_deadline_successes[0]),
          "rate": float(bin_deadline_successes[0].float() / bin_attempts[0].clamp_min(1)),
        },
        "60_to_120_deg": {
          "attempts": int(bin_attempts[1]),
          "successes": int(bin_deadline_successes[1]),
          "rate": float(bin_deadline_successes[1].float() / bin_attempts[1].clamp_min(1)),
        },
        "120_to_180_deg": {
          "attempts": int(bin_attempts[2]),
          "successes": int(bin_deadline_successes[2]),
          "rate": float(bin_deadline_successes[2].float() / bin_attempts[2].clamp_min(1)),
        },
      },
      "fingertip_contact_fraction": float(
        fingertip_contact_steps.sum().float() / active_steps.sum().clamp_min(1)
      ),
      "mean_max_position_drift_m": float(max_drift.mean()),
      "mean_initial_goal_error_rad": float(initial_goal_error.mean()),
      "mean_min_goal_error_rad": float(min_goal_error.mean()),
      "mean_max_cube_rotation_rad": float(max_cube_rotation.mean()),
      "mean_net_spin_rad": float(net_spin.mean()),
      "mean_abs_net_spin_rad": float(net_spin.abs().mean()),
      "net_spin_rad_by_trial": [float(value) for value in net_spin.cpu().tolist()],
      "positive_spin_trials": int((net_spin > 0.0).sum()),
      "completed_full_turns": int((net_spin >= 2.0 * torch.pi).sum()),
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
