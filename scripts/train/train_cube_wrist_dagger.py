"""Aggregate camera-only student trajectories with privileged teacher labels."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from scipy.spatial.transform import Rotation
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_cube_wrist_tags import build_cube
from train.train_cube_downward import make_cfg
from train.train_cube_wrist_student import ROOT, TEACHER, model
from wuji_mjlab.perception.apriltag import AprilTagBoardEstimator, PoseCache
from wuji_mjlab.perception.cube_wrist import SensorHistory, calibration
from wuji_mjlab.tasks.reorient.config.wuji_hand.rsl_rl.ppo import (
  wuji_hand_reorient_ppo_runner_cfg,
)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument(
    "--student",
    type=Path,
    default=ROOT / "logs/rsl_rl/wuji_cube_wrist_student/initial_bc_20260930/student.pt",
  )
  p.add_argument("--output", type=Path, required=True)
  p.add_argument("--rounds", type=int, default=3)
  p.add_argument("--episodes", type=int, default=20)
  p.add_argument("--seconds", type=float, default=16)
  p.add_argument("--updates", type=int, default=800)
  p.add_argument("--seed", type=int, default=9511)
  a = p.parse_args()
  a.output = a.output.resolve()
  a.output.mkdir(parents=True, exist_ok=False)
  configure_torch_backends()
  torch.manual_seed(a.seed)
  rng = np.random.default_rng(a.seed)
  student = model(207, 1, "cuda:0")
  student.load_state_dict(torch.load(a.student, weights_only=False)["actor_state_dict"])
  teacher = model(413, 1, "cuda:0")
  teacher.load_state_dict(torch.load(TEACHER, weights_only=False)["actor_state_dict"])
  teacher.eval()
  student.eval()
  xml, board = build_cube(a.output, "tag36h11")
  detector = AprilTagBoardEstimator(board)
  cfg = make_cfg(
    "rotate", 1, a.seed, cube_edge_m=0.045, cube_mass_kg=0.028192, wrist_assembly=True
  )
  cfg.scene.entities["object"].spec_fn = partial(mujoco.MjSpec.from_file, str(xml))
  cfg.viewer.width, cfg.viewer.height = 640, 480
  cfg.commands["reorient_command"].debug_vis = False
  env = ManagerBasedRlEnv(cfg, device="cuda:0", render_mode="rgb_array")
  clip = wuji_hand_reorient_ppo_runner_cfg().clip_actions
  wrapped = RslRlVecEnvWrapper(env, clip_actions=clip)
  renderer = mujoco.Renderer(env.sim.mj_model, height=480, width=640)
  cid = mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "robot/d405_b")
  opt = mujoco.MjvOption()
  opt.geomgroup[3:] = 0
  Rcam, pcam, Ttag, K = calibration()
  world_tag = Rotation.from_euler("y", 90, degrees=True).as_matrix() @ Ttag
  robot = env.scene["robot"]
  command = env.command_manager.get_term("reorient_command")
  optimizer = torch.optim.Adam(student.parameters(), lr=3e-5)
  xs, ys, reports = [], [], []
  started = time.perf_counter()
  try:
    for round_id in range(a.rounds):
      beta = [0.5, 0.2, 0.0][min(round_id, 2)]
      fresh_n = usable_n = total_n = dropped = 0
      for episode in range(a.episodes):
        obs, _ = wrapped.reset()
        cache = PoseCache(
          1, 0.15, max_position_speed_m_s=0.5, max_rotation_speed_rad_s=12
        )
        history = SensorHistory()
        last = np.zeros(20)
        previous = np.zeros(20)
        blackout = 0
        for step in range(round(a.seconds / env.step_dt)):
          env._offline_renderer.update(env.sim.data)
          renderer.update_scene(
            env._offline_renderer._data, camera=cid, scene_option=opt
          )
          rgb = renderer.render().copy()
          # Contiguous short missing-frame bursts, in addition to actual occlusion.
          if blackout == 0 and rng.random() < 0.015:
            blackout = int(rng.integers(1, 4))
          pose = None if blackout else detector.estimate(rgb, K)
          blackout = max(blackout - 1, 0)
          fresh, usable, _ = cache.update([pose], step * env.step_dt, Rcam, pcam)
          goal_q = command.goal_quat[0].cpu().numpy()
          goal = world_tag.T @ Rotation.from_quat(goal_q[[1, 2, 3, 0]]).as_matrix()
          x = history.update(
            robot.data.joint_pos[0].cpu().numpy(),
            robot.data.soft_joint_pos_limits[0].cpu().numpy(),
            env.action_manager.get_term("joint_pos").processed_action[0].cpu().numpy(),
            previous,
            cache.position[0],
            cache.rotation[0],
            goal,
          )
          with torch.inference_mode():
            # Privileged observations are restricted to the supervising teacher.
            label = teacher(TensorDict({"policy": obs["policy"]}, batch_size=[1]))
            sensor = torch.tensor(x, device=env.device)[None]
            predicted = student(TensorDict({"policy": sensor}, batch_size=[1]))
          if usable[0]:
            xs.append(sensor.detach().clone())
            ys.append(label.detach().clone())
            action = label if rng.random() < beta else predicted
          else:
            action = torch.tensor(last, dtype=torch.float32, device=env.device)[None]
          if clip is not None:
            action = action.clamp(-clip, clip)
          previous, last = last.copy(), action[0].cpu().numpy().copy()
          obs, _, done, _ = wrapped.step(action)
          fresh_n += int(fresh[0])
          usable_n += int(usable[0])
          total_n += 1
          if done[0]:
            dropped += 1
            break
        print(
          f"round={round_id} episode={episode + 1}/{a.episodes} samples={len(xs)} drops={dropped}",
          flush=True,
        )
      if not xs:
        raise RuntimeError("No usable camera samples collected")
      xdata, ydata = torch.cat(xs).detach().clone(), torch.cat(ys).detach().clone()
      # Keep normalizer fixed, matching deployment. All accumulated sensor samples
      # remain available to prevent forgetting previous student state distributions.
      for _update in range(a.updates):
        ids = torch.randint(len(xdata), (min(512, len(xdata)),), device=env.device)
        prediction = student(TensorDict({"policy": xdata[ids]}, batch_size=[len(ids)]))
        loss = torch.nn.functional.smooth_l1_loss(prediction, ydata[ids])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
      checkpoint = {
        "actor_state_dict": student.state_dict(),
        "input_dim": 207,
        "teacher": str(TEACHER),
        "training": "Camera DAgger with privileged teacher supervision only",
        "round": round_id,
        "visual_timeout_s": 0.15,
      }
      torch.save(checkpoint, a.output / f"student_round{round_id}.pt")
      torch.save(checkpoint, a.output / "student.pt")
      reports.append(
        {
          "round": round_id,
          "teacher_action_probability": beta,
          "aggregated_samples": len(xdata),
          "dropped_episodes": dropped,
          "fresh_fraction": fresh_n / total_n,
          "usable_fraction": usable_n / total_n,
          "last_training_loss": float(loss.detach()),
          "elapsed_s": time.perf_counter() - started,
        }
      )
      report = {
        "seed": a.seed,
        "rounds": reports,
        "student_input": "207D camera pose + encoders + known commands/actions",
        "teacher_privilege": "Training labels and teacher-mixed collection actions only; evaluation is student only",
        "camera_mount": json.loads(
          (
            ROOT
            / "src/wuji_mjlab/assets/robots/wuji_hand/meshes/wrist_assembly/source.json"
          ).read_text()
        ),
      }
      (a.output / "training.json").write_text(json.dumps(report, indent=2))
      print(json.dumps(reports[-1]), flush=True)
  finally:
    renderer.close()
    env.close()


if __name__ == "__main__":
  main()
