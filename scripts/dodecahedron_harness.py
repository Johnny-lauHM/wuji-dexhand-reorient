"""Local acceptance harness: real contacts, SO(3), reset, PPO and checkpoint replay.

Run from the repository root. Outputs stay local; no dashboard or service needed.
This is a smoke/acceptance check, not a claim that the policy has converged.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v3 as iio
import torch
import wuji_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

TASK = "WujiHand_Dodecahedron"


def require(condition: bool, message: str) -> None:
  if not condition:
    raise RuntimeError(message)


def check_finite(obs, reward=None) -> None:
  for name, value in obs.items():
    require(bool(torch.isfinite(value).all()), f"Non-finite observation: {name}")
  if reward is not None:
    require(bool(torch.isfinite(reward).all()), "Non-finite reward")


def run(args, report: dict) -> None:
  configure_torch_backends()
  cfg = load_env_cfg(TASK, play=bool(args.checkpoint))
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  cfg.viewer.width, cfg.viewer.height = 960, 720
  env = ManagerBasedRlEnv(
    cfg=cfg, device=args.device, render_mode=None if args.no_render else "rgb_array"
  )
  try:
    obs, _ = env.reset()
    check_finite(obs)
    report["observation_shapes"] = {k: list(v.shape) for k, v in obs.items()}
    action = torch.zeros(
      (env.num_envs, env.action_manager.total_action_dim), device=env.device
    )
    require(action.shape[1] == 20, "Expected 20 Wuji Hand actions")
    if not args.no_render:
      iio.imwrite(args.output / "initial_scene.png", env.render())

    # Exercise the real command with a stationary simulator state: non-aligned
    # goals must fail; aligned goals must hold before success and then resample.
    cmd = env.command_manager.get_term("reorient_command")
    q = env.scene["object"].data.root_link_quat_w
    # Left-compose a 180-degree X rotation, giving an error of pi for every q.
    cmd.goal_quat_w.copy_(torch.stack((-q[:, 1], q[:, 0], -q[:, 3], q[:, 2]), dim=-1))
    cmd._update_command()
    require(not bool(cmd.within_threshold.any()), "Opposite orientation accepted")
    cmd.goal_quat_w.copy_(env.scene["object"].data.root_link_quat_w)
    original_goal = cmd.goal_quat_w.clone()
    for _ in range(cmd.cfg.success_hold_steps - 1):
      cmd._update_command()
      require(
        not bool(cmd.success_achieved.any()), "Goal succeeded before hold completed"
      )
    cmd._update_command()
    require(bool(cmd.success_achieved.all()), "Aligned hold did not succeed")
    for _ in range(cmd.cfg.goal_switch_delay - 1):
      cmd._update_command()
    require(bool(cmd.goal_switched.all()), "Goal did not advance")
    require(
      not torch.allclose(original_goal, cmd.goal_quat_w), "Goal was not resampled"
    )
    report["goal_hold_and_advance"] = "passed"

    env.reset()
    contacts = 0
    drops = 0
    for _ in range(args.steps):
      obs, reward, terminated, _, _ = env.step(action)
      check_finite(obs, reward)
      require(
        bool(torch.isfinite(env.scene["object"].data.root_link_pose_w).all()),
        "Non-finite object pose",
      )
      contacts += int(env.scene["tip_object_contact"].data.found.sum().item())
      drops += int(terminated.sum().item())
    require(contacts > 0, "No fingertip/dodecahedron contact observed")
    report["zero_action_rollout"] = {
      "steps": args.steps,
      "fingertip_contacts": contacts,
      "drop_resets": drops,
    }

    # Force all objects well outside the cage and require actual termination.
    env.reset()
    obj = env.scene["object"]
    pose = obj.data.root_link_pose_w.clone()
    pose[:, 0] += 1.0
    obj.write_root_link_pose_to_sim(pose)
    dropped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for _ in range(16):
      obs, reward, terminated, _, _ = env.step(action)
      check_finite(obs, reward)
      dropped |= terminated
    require(
      bool(dropped.all()), "Forced cage escape failed to terminate every environment"
    )
    report["forced_drop_reset"] = "passed"

    agent = load_rl_cfg(TASK)
    agent.seed = args.seed
    # Preserve network, optimizer and rollout defaults; smaller batches for smoke runs.
    agent.algorithm.num_mini_batches = min(32, args.num_envs)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner_cls = load_runner_cls(TASK)
    runner = runner_cls(
      wrapped, asdict(agent), str(args.output / "ppo"), device=args.device
    )
    if args.checkpoint:
      runner.load(str(args.checkpoint), map_location=args.device)
    else:
      before = [p.detach().clone() for p in runner.alg.actor.parameters()]
      runner.learn(args.iterations)
      after = list(runner.alg.actor.parameters())
      require(
        all(bool(torch.isfinite(p).all()) for p in after), "Non-finite actor parameters"
      )
      require(
        any(not torch.equal(a, b) for a, b in zip(before, after, strict=True)),
        "PPO did not update actor parameters",
      )
      checkpoint = args.output / "smoke_policy.pt"
      runner.save(str(checkpoint))
      # Confirm strict load plus deterministic inference round-trip.
      observation = wrapped.get_observations()
      policy = runner.get_inference_policy(device=args.device)
      with torch.inference_mode():
        expected = policy(observation).clone()
      # A fresh runner also checks that replay does not depend on in-memory
      # training state (including inference-mode normalization buffers).
      runner = runner_cls(wrapped, asdict(agent), device=args.device)
      runner.load(str(checkpoint), map_location=args.device)
      with torch.inference_mode():
        actual = runner.get_inference_policy(device=args.device)(observation)
      require(
        torch.allclose(expected, actual), "Checkpoint inference changed after reload"
      )
      report["ppo"] = {
        "iterations": args.iterations,
        "parameters_updated": True,
        "checkpoint": str(checkpoint),
        "fresh_runner_replay": "passed",
      }

    # First-target outcomes only: subsequent targets/resets cannot inflate success.
    # PPO rollouts create inference-mode sensor buffers; reset in the same mode.
    with torch.inference_mode():
      obs, _ = wrapped.reset()
    policy = runner.get_inference_policy(device=args.device)
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    successes = failures = 0
    with torch.inference_mode():
      for _ in range(args.steps):
        obs, reward, done, _ = wrapped.step(policy(obs))
        check_finite(obs, reward)
        reached = active & cmd.success_achieved & ~done.bool()
        failed = active & done.bool()
        successes += int(reached.sum().item())
        failures += int(failed.sum().item())
        active &= ~(reached | failed)
    report["first_target_evaluation"] = {
      "trials": env.num_envs,
      "horizon_s": args.steps * env.step_dt,
      "successes": successes,
      "terminated_or_timed_out": failures,
      "unresolved_at_horizon": int(active.sum().item()),
      "success_rate": successes / env.num_envs,
      "checkpoint": str(args.checkpoint)
      if args.checkpoint
      else "smoke_policy.pt (not converged)",
    }
  finally:
    env.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--steps", type=int, default=200)
  parser.add_argument("--iterations", type=int, default=2)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--checkpoint", type=Path)
  parser.add_argument("--no-render", action="store_true")
  parser.add_argument(
    "--output", type=Path, default=Path("outputs/dodecahedron_harness")
  )
  args = parser.parse_args()
  if min(args.num_envs, args.steps, args.iterations) < 1:
    parser.error("num-envs, steps and iterations must be positive")
  args.output.mkdir(parents=True, exist_ok=True)
  report = {
    "task": TASK,
    "status": "running",
    "seed": args.seed,
    "versions": {
      name: version(name)
      for name in ("mjlab", "mujoco", "mujoco-warp", "torch", "rsl-rl-lib")
    },
  }
  started = time.monotonic()
  try:
    run(args, report)
    report["status"] = "passed"
  except Exception as exc:
    report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    raise
  finally:
    report["elapsed_s"] = time.monotonic() - started
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
  main()
