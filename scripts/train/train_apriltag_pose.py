"""Train from actual rendered AprilTag detections, not simulated truth poses."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import imageio.v3 as iio
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends
from wuji_mjlab.rl.runner import WujiOnPolicyRunner
from wuji_mjlab.tasks.dodecahedron.tag_pose import make_tag_pose_configs, quat_matrix


def positive_int(value):
  number = int(value)
  if number <= 0:
    raise argparse.ArgumentTypeError("must be positive")
  return number


def summarize(values):
  if not values:
    return None
  return {
    "median": float(np.median(values)),
    "p95": float(np.percentile(values, 95)),
    "max": float(np.max(values)),
  }


def run(args, output, report):
  cv2.setNumThreads(1)
  cv2.setRNGSeed(args.seed)
  configure_torch_backends()
  cfg, agent = make_tag_pose_configs(
    args.num_envs, args.width, args.height, args.seed, args.detector_workers
  )
  pose_params = cfg.observations["policy"].terms["apriltag_pose"].params
  board_snapshot = output / "apriltag_board.json"
  shutil.copyfile(pose_params["board_path"], board_snapshot)
  pose_params["board_path"] = str(board_snapshot.resolve())
  agent.max_iterations = args.iterations
  if args.smoke:
    agent.num_steps_per_env = 8
    agent.algorithm.num_learning_epochs = 1
  dump_yaml(output / "params/env.yaml", asdict(cfg))
  dump_yaml(output / "params/agent.yaml", asdict(agent))
  env = ManagerBasedRlEnv(cfg, device=args.device)
  tracker = None
  try:
    obs, _ = env.reset()
    tracker = env.observation_manager.get_term_cfg("policy", "apriltag_pose").func
    calibration = {
      "K": tracker.K.tolist(),
      "distortion": [0.0] * 5,
      "camera_optical_to_tag_rotation": tracker.camera_R_tag.tolist(),
      "camera_origin_in_tag_m": tracker.camera_position_tag.tolist(),
      "opencv_version": cv2.__version__,
      "step_dt_s": env.step_dt,
      "extrinsics_source": "fixed simulated camera/wrist mounting configuration; not real calibration",
    }
    (output / "calibration.json").write_text(json.dumps(calibration, indent=2))
    if set(obs) != {"policy", "critic"} or obs["policy"].shape != (args.num_envs, 213):
      raise RuntimeError(
        f"Unexpected observation interface: {[(k, v.shape) for k, v in obs.items()]}"
      )
    report["observation_shapes"] = {k: list(v.shape) for k, v in obs.items()}
    for i, rgb in enumerate(env.scene["l515"].data.rgb.cpu().numpy()):
      if i < 4:
        iio.imwrite(output / f"camera_initial_{i}.png", rgb)
        pose = tracker.last_poses[i]
        if pose is not None:
          overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
          cv2.drawFrameAxes(
            overlay,
            tracker.K,
            np.zeros(5),
            cv2.Rodrigues(pose.rotation)[0],
            pose.translation,
            0.025,
          )
          iio.imwrite(
            output / f"pose_overlay_{i}.png", cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
          )

    # Ground truth is read only HERE for audit; never passed back to the tracker
    # or policy. Probe uses zero actions to characterize initial hand occlusion.
    position_errors, rotation_errors = [], []
    detected_per_env = np.zeros(args.num_envs, dtype=int)
    latency = []
    action = torch.zeros((args.num_envs, 20), device=args.device)
    outliers = []
    for probe_step in range(args.probe_steps):
      tick = time.perf_counter()
      obs, reward, *_ = env.step(action)
      torch.cuda.synchronize(args.device)
      latency.append(time.perf_counter() - tick)
      if not all(torch.isfinite(v).all() for v in (*obs.values(), reward)):
        raise RuntimeError("Nonfinite probe observations or rewards")
      positions = env.scene["object"].data.root_link_pos_w.cpu().numpy()
      rotations = quat_matrix(env.scene["object"].data.root_link_quat_w.cpu().numpy())
      # Use configured camera transform as independent audit reference.
      camera = cfg.scene.sensors[-1]
      world_to_camera = np.diag([1.0, -1.0, -1.0]) @ quat_matrix(camera.quat).T
      for i, pose in enumerate(tracker.last_poses):
        if pose is None:
          continue
        detected_per_env[i] += 1
        position_errors.append(
          float(
            np.linalg.norm(
              pose.translation - world_to_camera @ (positions[i] - camera.pos)
            )
          )
        )
        relative = pose.rotation @ (world_to_camera @ rotations[i]).T
        rotation_errors.append(
          float(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
        )
        if position_errors[-1] > 0.03 or rotation_errors[-1] > 0.3:
          outliers.append(
            {
              "step": probe_step,
              "env": i,
              "ids": pose.ids,
              "position_error_m": position_errors[-1],
              "rotation_error_rad": rotation_errors[-1],
              "reprojection_error_px": pose.reprojection_error_px,
            }
          )
          if len(outliers) <= 3:
            iio.imwrite(
              output / f"outlier_{probe_step}_{i}.png",
              env.scene["l515"].data.rgb[i].cpu().numpy(),
            )
    report["probe"] = {
      "steps": args.probe_steps,
      "actions": "zero",
      "detected_per_env": detected_per_env.tolist(),
      "valid_fraction": float(
        detected_per_env.sum() / (args.probe_steps * args.num_envs)
      ),
      "position_error_m": summarize(position_errors),
      "rotation_error_rad": summarize(rotation_errors),
      "vector_step_seconds": summarize(latency),
      "environment_steps_per_second": args.num_envs * args.probe_steps / sum(latency),
    }
    print(json.dumps(report["probe"], indent=2), flush=True)
    report["probe"]["outliers"] = outliers
    if not position_errors:
      raise RuntimeError(
        "No AprilTag pose detections during probe; inspect camera images before training"
      )
    if np.median(position_errors) > 0.03 or np.median(rotation_errors) > 0.3:
      raise RuntimeError("Detected poses fail the coarse coordinate/geometry audit")

    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = WujiOnPolicyRunner(wrapped, asdict(agent), str(output), args.device)
    if args.resume:
      runner.load(str(args.resume.resolve()), map_location=args.device)
    storage = runner.alg.storage.observations
    if any(v.ndim != 3 for v in storage.values()):
      raise RuntimeError("Image or non-vector observation leaked into PPO storage")
    report["rollout_observation_bytes"] = sum(
      v.numel() * v.element_size() for v in storage.values()
    )
    before = (
      [p.detach().clone() for p in runner.alg.actor.parameters()]
      if args.smoke
      else None
    )
    torch.cuda.reset_peak_memory_stats(args.device)
    training_frames_before, training_detected_before = tracker.frames, tracker.detected
    tick = time.perf_counter()
    runner.learn(args.iterations, init_at_random_ep_len=True)
    torch.cuda.synchronize(args.device)
    report["train_seconds"] = time.perf_counter() - tick
    report["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
    report["peak_memory_note"] = "PyTorch only; excludes some Warp/CUDA allocations"
    report["tracking"] = {
      "frames": tracker.frames,
      "detected": tracker.detected,
      "detector_and_transfer_seconds": tracker.detector_seconds,
      "transfer_seconds": tracker.transfer_seconds,
      "detection_seconds": tracker.detection_seconds,
      "rejected_jumps": tracker.cache.rejected_jumps,
    }
    report["training_tracking"] = {
      "frames": tracker.frames - training_frames_before,
      "detected": tracker.detected - training_detected_before,
    }
    checkpoint = output / "final.pt"
    runner.save(str(checkpoint))
    report["checkpoint"] = str(checkpoint)
    if args.smoke:
      after = list(runner.alg.actor.parameters())
      if not all(torch.isfinite(p).all() for p in after):
        raise RuntimeError("Nonfinite actor parameters")
      if not any(not torch.equal(a, b) for a, b in zip(before, after, strict=True)):
        raise RuntimeError("PPO did not update actor")
      observation = wrapped.get_observations()
      with torch.inference_mode():
        expected = runner.get_inference_policy()(observation).clone()
      replay = WujiOnPolicyRunner(wrapped, asdict(agent), device=args.device)
      replay.load(str(checkpoint), map_location=args.device)
      with torch.inference_mode():
        actual = replay.get_inference_policy()(observation)
      torch.testing.assert_close(actual, expected)
      report.update(actor_updated=True, checkpoint_replay="passed")
    report.update(status="passed", convergence_validated=False)
  finally:
    if tracker is not None:
      tracker.close()
    env.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--num-envs", type=positive_int, default=4)
  parser.add_argument("--detector-workers", type=positive_int, default=8)
  parser.add_argument("--iterations", type=positive_int, default=5000)
  parser.add_argument("--probe-steps", type=positive_int, default=32)
  parser.add_argument("--width", type=positive_int, default=1280)
  parser.add_argument("--height", type=positive_int, default=720)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument(
    "--resume", type=Path, help="Only a matching 213-D tag-pose checkpoint"
  )
  parser.add_argument("--output", type=Path)
  parser.add_argument(
    "--smoke",
    action="store_true",
    help="4 worlds, 32 probe steps, 2 PPO updates and reload",
  )
  args = parser.parse_args()
  if args.smoke:
    args.num_envs, args.iterations = 4, 2
  if not args.device.startswith("cuda") or not torch.cuda.is_available():
    parser.error("CUDA is required for simulation camera rendering")
  output = args.output or Path(
    "logs/rsl_rl/wuji_dodecahedron_apriltag_pose"
  ) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
  output.mkdir(parents=True, exist_ok=False)
  (output / "launch.json").write_text(json.dumps(vars(args), default=str, indent=2))
  report = {
    "status": "running",
    "observation_source": "rendered RGB -> tag36h11 -> board PnP",
  }
  try:
    run(args, output, report)
  except Exception as error:
    report.update(status="failed", error=f"{type(error).__name__}: {error}")
    raise
  finally:
    (output / "result.json").write_text(json.dumps(report, indent=2))
    print(f"Output: {output.resolve()}", flush=True)


if __name__ == "__main__":
  main()
