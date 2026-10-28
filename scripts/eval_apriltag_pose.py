"""Evaluate a tag-pose policy and film the first four trials without cherry-picking."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import imageio.v2 as iio
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.lab_api.math import quat_error_magnitude
from mjlab.utils.torch import configure_torch_backends
from wuji_mjlab.rl.runner import WujiOnPolicyRunner
from wuji_mjlab.tasks.dodecahedron.tag_pose import make_tag_pose_configs


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--seconds", type=float, default=30)
  parser.add_argument("--seed", type=int, default=123)
  args = parser.parse_args()
  if args.num_envs < 4 or args.seconds <= 0:
    parser.error("Need at least four worlds and positive duration")
  args.output.mkdir(parents=True, exist_ok=False)
  cv2.setNumThreads(1)
  configure_torch_backends()
  cfg, agent = make_tag_pose_configs(args.num_envs, seed=args.seed, detector_workers=8)
  board = args.checkpoint.parent / "apriltag_board.json"
  if board.exists():
    cfg.observations["policy"].terms["apriltag_pose"].params["board_path"] = str(
      board.resolve()
    )
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  tracker = None
  writer = None
  try:
    tracker = env.observation_manager.get_term_cfg("policy", "apriltag_pose").func
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = WujiOnPolicyRunner(wrapped, asdict(agent), device="cuda:0")
    runner.load(str(args.checkpoint), map_location="cuda:0")
    policy = runner.get_inference_policy()
    cmd = env.command_manager.get_term("reorient_command")
    active = np.ones(args.num_envs, dtype=bool)
    goals = np.zeros(args.num_envs, dtype=int)
    first_success = [None] * args.num_envs
    drops = [None] * args.num_envs
    tracked = np.zeros(args.num_envs, dtype=int)
    frames = np.zeros(args.num_envs, dtype=int)
    tiles = [np.zeros((360, 640, 3), dtype=np.uint8) for _ in range(4)]
    last_errors = np.zeros(args.num_envs)
    writer = iio.get_writer(
      str(args.output / "demo.mp4"), fps=10, codec="libx264", quality=8
    )
    with torch.inference_mode():
      obs, _ = wrapped.reset()
      for step in range(int(args.seconds / env.step_dt)):
        t = step * env.step_dt
        error = (
          quat_error_magnitude(env.scene["object"].data.root_link_quat_w, cmd.goal_quat)
          .cpu()
          .numpy()
        )
        last_errors[active] = error[active]
        fresh = tracker.output[:, 9].cpu().numpy() > 0.5
        frames += active
        tracked += active & fresh
        if step % 2 == 0:
          rgb = env.scene["l515"].data.rgb[:4].cpu().numpy()
          for i in range(4):
            if active[i]:
              tiles[i] = cv2.resize(rgb[i], (640, 360))
            tile = tiles[i].copy()
            cv2.rectangle(tile, (0, 0), (640, 68), (15, 15, 15), -1)
            status = "ACTIVE" if active[i] else "TRIAL ENDED (drop/termination)"
            cv2.putText(
              tile,
              f"Trial {i + 1} | {t:.1f}s | {status}",
              (10, 22),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.52,
              (255, 255, 255),
              1,
            )
            cv2.putText(
              tile,
              f"Goals: {goals[i]} | error: {np.degrees(last_errors[i]):.1f} deg | tag: {'OK' if fresh[i] and active[i] else 'LOST/ENDED'}",
              (10, 49),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.5,
              (255, 255, 255),
              1,
            )
            if i == 0:
              montage = np.zeros((720, 1280, 3), dtype=np.uint8)
            montage[
              i // 2 * 360 : (i // 2 + 1) * 360, i % 2 * 640 : (i % 2 + 1) * 640
            ] = tile
          writer.append_data(montage)
          if step == 100:
            iio.imwrite(args.output / "preview.png", montage)
        action = policy(obs)
        if not torch.isfinite(action).all():
          raise RuntimeError("Nonfinite policy action")
        obs, reward, done, _ = wrapped.step(action)
        ended = done.cpu().numpy().astype(bool) & active
        success = cmd.success_achieved.cpu().numpy() & active & ~ended
        goals += success
        for i in np.flatnonzero(success):
          if first_success[i] is None:
            first_success[i] = (step + 1) * env.step_dt
        for i in np.flatnonzero(ended):
          drops[i] = (step + 1) * env.step_dt
        active &= ~ended
        if step % 100 == 0:
          print(
            f"t={t:.1f}s active={active.sum()} first_success={sum(x is not None for x in first_success)} goals={goals.sum()}",
            flush=True,
          )
        if not active.any():
          break
    result = {
      "checkpoint": str(args.checkpoint.resolve()),
      "seed": args.seed,
      "trials": args.num_envs,
      "horizon_s": args.seconds,
      "first_goal_successes": sum(x is not None for x in first_success),
      "first_goal_success_rate": sum(x is not None for x in first_success)
      / args.num_envs,
      "total_goals_before_first_termination": int(goals.sum()),
      "survived_horizon": int(active.sum()),
      "trials_detail": [
        {
          "trial": i + 1,
          "first_success_s": first_success[i],
          "termination_s": drops[i],
          "goals": int(goals[i]),
          "accepted_tracking_fraction": float(tracked[i] / max(frames[i], 1)),
          "last_orientation_error_deg": float(np.degrees(last_errors[i])),
        }
        for i in range(args.num_envs)
      ],
      "video_trials": [1, 2, 3, 4],
      "policy": "deterministic mean actions",
      "note": "Original training configuration; first episode only per trial. Video freezes terminated trials; no reset successes counted.",
    }
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2))
    iio.imwrite(args.output / "final_frame.png", montage)
    print(
      json.dumps({k: v for k, v in result.items() if k != "trials_detail"}, indent=2),
      flush=True,
    )
  finally:
    if writer is not None:
      writer.close()
    if tracker is not None:
      tracker.close()
    env.close()


if __name__ == "__main__":
  main()
