"""Run the converged state Teacher from explicit AprilTag board poses."""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import imageio.v2 as iio
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.models import MLPModel
from tensordict import TensorDict
from wuji_mjlab.tasks.dodecahedron.env_cfg import wuji_hand_dodecahedron_env_cfg
from wuji_mjlab.tasks.dodecahedron.tag_pose import (
  make_tag_pose_configs,
  teacher_observation_from_apriltag_policy,
)

DEFAULT_TEACHER = Path(
  "logs/rsl_rl/wuji_dodecahedron/2026-09-12_15-55-31_Dodecahedron/model_4999.pt"
)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--seconds", type=float, default=30.0)
  parser.add_argument("--seed", type=int, default=123)
  parser.add_argument("--width", type=int, default=1280)
  parser.add_argument("--height", type=int, default=720)
  parser.add_argument("--detector-workers", type=int, default=8)
  parser.add_argument(
    "--domain-randomization",
    action="store_true",
    help="Enable the moderate sim-to-real dynamics profile",
  )
  parser.add_argument(
    "--video", action="store_true", help="Record the first four camera views"
  )
  parser.add_argument(
    "--video-style",
    choices=("official", "camera-grid"),
    default="official",
    help="Official external viewer or four diagnostic sensor views",
  )
  args = parser.parse_args()
  if args.num_envs <= 0 or args.seconds <= 0:
    parser.error("num-envs and seconds must be positive")
  if not args.teacher.is_file():
    parser.error(f"Teacher checkpoint not found: {args.teacher}")
  args.output.mkdir(parents=True, exist_ok=False)
  cv2.setNumThreads(1)
  configure_torch_backends()
  cfg, agent = make_tag_pose_configs(
    args.num_envs,
    args.width,
    args.height,
    args.seed,
    args.detector_workers,
    sim2real_domain_randomization=args.domain_randomization,
  )
  if args.video and args.video_style == "official":
    cfg.viewer.width = 640
    cfg.viewer.height = 480
    cfg.commands["reorient_command"].debug_vis = True
  truth_cfg = wuji_hand_dodecahedron_env_cfg(num_envs=args.num_envs)
  cfg.observations["teacher_audit"] = deepcopy(truth_cfg.observations["policy"])
  render_mode = "rgb_array" if args.video and args.video_style == "official" else None
  env = ManagerBasedRlEnv(cfg, device="cuda:0", render_mode=render_mode)
  tracker = None
  writer = None
  try:
    tracker = env.observation_manager.get_term_cfg("policy", "apriltag_pose").func
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    obs, _ = wrapped.reset()
    reconstructed = teacher_observation_from_apriltag_policy(obs["policy"])
    actor_cfg = asdict(agent.actor)
    for key in (
      "class_name",
      "cnn_cfg",
      "rnn_type",
      "rnn_hidden_dim",
      "rnn_num_layers",
    ):
      actor_cfg.pop(key, None)
    template = TensorDict(
      {"teacher": torch.zeros_like(reconstructed)}, batch_size=[args.num_envs]
    )
    policy = MLPModel(
      template, {"teacher": ["teacher"]}, "teacher", wrapped.num_actions, **actor_cfg
    ).to("cuda:0")
    checkpoint = torch.load(args.teacher, map_location="cuda:0", weights_only=False)
    policy.load_state_dict(checkpoint["actor_state_dict"])
    policy.eval()
    cmd = env.command_manager.get_term("reorient_command")
    if args.video and args.video_style == "official":
      cmd._visualization.show_frames = False
      cmd._visualization.show_status = True
      for name, term in env.reward_manager.get_visualizable_terms():
        if name == "cage_escape":
          term.debug_vis = lambda _visualizer: None
    successes = torch.zeros(args.num_envs, dtype=torch.bool, device="cuda:0")
    active = torch.ones(args.num_envs, dtype=torch.bool, device="cuda:0")
    goal_counts = torch.zeros(args.num_envs, dtype=torch.int32, device="cuda:0")
    termination_step = torch.full(
      (args.num_envs,), -1, dtype=torch.int32, device="cuda:0"
    )
    first_success = torch.full((args.num_envs,), -1, dtype=torch.int32, device="cuda:0")
    tracked = torch.zeros(args.num_envs, device="cuda:0")
    audit_abs_sum = 0.0
    object_audit_abs_sum = 0.0
    audit_samples = 0
    object_audit_samples = 0
    steps = int(args.seconds / env.step_dt)
    montage = None
    if args.video:
      if args.video_style == "camera-grid" and args.num_envs < 4:
        parser.error("camera-grid video requires at least four environments")
      writer = iio.get_writer(
        str(args.output / "demo.mp4"), fps=10, codec="libx264", quality=8
      )
    with torch.inference_mode():
      for step in range(steps):
        teacher_obs = teacher_observation_from_apriltag_policy(obs["policy"])
        audit_error = (teacher_obs[active] - obs["teacher_audit"][active]).abs()
        audit_abs_sum += audit_error.sum().item()
        audit_samples += audit_error.numel()
        object_error = audit_error[:, 120:147]
        object_audit_abs_sum += object_error.sum().item()
        object_audit_samples += object_error.numel()
        tracked += tracker.output[:, 9]
        if writer is not None and step % 2 == 0:
          if args.video_style == "official":
            montage = env.render()
            cv2.putText(
              montage,
              "1X Speed",
              (470, 42),
              cv2.FONT_HERSHEY_SIMPLEX,
              1.0,
              (255, 255, 255),
              2,
            )
            cv2.putText(
              montage,
              "APRILTAG VISION",
              (18, 34),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.55,
              (255, 255, 255),
              1,
            )
            cv2.putText(
              montage,
              "SIM",
              (265, 452),
              cv2.FONT_HERSHEY_SIMPLEX,
              1.45,
              (255, 255, 255),
              2,
            )
          else:
            rgb = env.scene["l515"].data.rgb[:4].cpu().numpy()
            montage = np.zeros((720, 1280, 3), dtype=np.uint8)
            for i in range(4):
              tile = cv2.resize(rgb[i], (640, 360))
              cv2.rectangle(tile, (0, 0), (640, 62), (15, 15, 15), -1)
              fresh = tracker.output[i, 9].item() > 0.5
              tag_status = "FRESH" if fresh else "CACHED"
              goal_status = "YES" if successes[i] else "NO"
              cv2.putText(
                tile,
                f"Trial {i + 1} | {step * env.step_dt:.1f}s | Tag: {tag_status}",
                (10, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.53,
                (255, 255, 255),
                1,
              )
              cv2.putText(
                tile,
                f"Frozen Teacher | Goal reached: {goal_status}",
                (10, 49),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (70, 255, 70) if successes[i] else (255, 255, 255),
                1,
              )
              row, column = divmod(i, 2)
              montage[
                row * 360 : (row + 1) * 360, column * 640 : (column + 1) * 640
              ] = tile
          writer.append_data(montage)
        action = policy(
          TensorDict({"teacher": teacher_obs}, batch_size=[args.num_envs])
        )
        obs, _, dones, _ = wrapped.step(action)
        goal_counts += (active & cmd.success_achieved).int()
        new_success = active & (cmd.goal_reach_count > 0) & ~successes
        first_success[new_success] = step + 1
        successes |= new_success
        ended = active & dones.bool()
        termination_step[ended] = step + 1
        active &= ~dones.bool()
    times = first_success[first_success > 0].float() * env.step_dt
    report = {
      "teacher_checkpoint": str(args.teacher.resolve()),
      "observation_source": "RGB -> AprilTag 36h11 -> board PnP -> frozen Teacher",
      "seed": args.seed,
      "num_envs": args.num_envs,
      "horizon_s": args.seconds,
      "domain_randomization": args.domain_randomization,
      "first_goal_successes": int(successes.sum()),
      "first_goal_success_rate": float(successes.float().mean()),
      "mean_first_success_s": float(times.mean()) if times.numel() else None,
      "fresh_detection_fraction": float((tracked / steps).mean()),
      "teacher_observation_mae": audit_abs_sum / audit_samples,
      "teacher_object_state_mae": object_audit_abs_sum / object_audit_samples,
      "first_episodes_ended": int((~active).sum()),
      "survived_horizon": int(active.sum()),
      "total_goals_before_termination": int(goal_counts.sum()),
      "mean_goals_per_trial": float(goal_counts.float().mean()),
      "termination_times_s": [
        float(step * env.step_dt) if step > 0 else None
        for step in termination_step.cpu().tolist()
      ],
      "detector_frames_per_wall_second": tracker.frames
      / max(tracker.detector_seconds, 1.0e-9),
      "rejected_pose_jumps": tracker.cache.rejected_jumps,
      "video": str((args.output / "demo.mp4").resolve()) if args.video else None,
      "video_style": args.video_style if args.video else None,
    }
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2))
    if montage is not None:
      iio.imwrite(args.output / "preview.png", montage)
    print(json.dumps(report, indent=2), flush=True)
  finally:
    if tracker is not None:
      tracker.close()
    if writer is not None:
      writer.close()
    env.close()


if __name__ == "__main__":
  main()
