"""Closed-loop evaluation: object observations come only from camera marker pixels."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import csv
import json
import time
from functools import partial
from pathlib import Path

import mediapy as media
import mujoco
import numpy as np
import torch
from check_cube_wrist_tags import build_cube
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
from tensordict import TensorDict
from train.train_cube_downward import make_cfg
from train.train_cube_wrist_student import model
from wuji_mjlab.perception.apriltag import AprilTagBoardEstimator, PoseCache
from wuji_mjlab.perception.cube_wrist import SensorHistory, calibration
from wuji_mjlab.tasks.reorient.config.wuji_hand.rsl_rl.ppo import (
  wuji_hand_reorient_ppo_runner_cfg,
)
from wuji_mjlab.tasks.reorient.mdp.command_visualization import (
  ReorientCommandVisualization,
)
from wuji_mjlab.tasks.reorient.reorient_constants import TAG_IN_PALM_POS

ROOT = Path(__file__).resolve().parents[1]


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument(
    "--student",
    type=Path,
    default=ROOT / "logs/rsl_rl/wuji_cube_wrist_student/initial_bc_20260930/student.pt",
  )
  p.add_argument("--trials", type=int, default=16)
  p.add_argument("--seconds", type=float, default=32)
  p.add_argument("--seed", type=int, default=7400)
  p.add_argument("--output", type=Path, required=True)
  p.add_argument(
    "--camera-profile",
    choices=("assembly", "factory-vga", "nominal-wide"),
    default="assembly",
  )
  p.add_argument(
    "--factory-intrinsics",
    type=Path,
    default=ROOT
    / "src/wuji_mjlab/assets/robots/wuji_hand/meshes/wrist_assembly/camera_calibration.json",
  )
  p.add_argument(
    "--official-demo",
    action="store_true",
    help="Synchronized external side view with visual state machine plus wrist pixels",
  )
  a = p.parse_args()
  out = a.output.resolve()
  out.mkdir(parents=True, exist_ok=False)
  xml, board = build_cube(out, "tag36h11")
  est = AprilTagBoardEstimator(board)
  configure_torch_backends()
  student = model(207, 1, "cuda:0")
  student.load_state_dict(
    torch.load(a.student, map_location="cuda:0", weights_only=False)["actor_state_dict"]
  )
  student.eval()
  cfg = make_cfg(
    "rotate", 1, a.seed, cube_edge_m=0.045, cube_mass_kg=0.028192, wrist_assembly=True
  )
  cfg.scene.entities["object"].spec_fn = partial(mujoco.MjSpec.from_file, str(xml))
  cfg.viewer.width = 640
  cfg.viewer.height = 480
  if a.camera_profile == "nominal-wide":
    cfg.viewer.width = 848
  if a.official_demo:
    cfg.viewer.width = 960
    cfg.viewer.height = 720
    cfg.viewer.azimuth = 90
    cfg.viewer.elevation = -12
    cfg.viewer.distance = 0.48
  cfg.commands["reorient_command"].debug_vis = False
  # Truth is used for physics/rewards/termination/scoring, never to build actor input.
  env = ManagerBasedRlEnv(cfg, device="cuda:0", render_mode="rgb_array")
  renderer = None
  trials = []
  rows = []
  videos = []
  combined_videos = []
  start = time.perf_counter()
  Rcam, pcam, Ttag, K = calibration()
  image_width, image_height = 640, 480
  projection_source = "Historical assembly pinhole intrinsics"
  if a.camera_profile == "factory-vga":
    profile = json.loads(a.factory_intrinsics.read_text())
    K = np.array(
      [[profile["fx"], 0, profile["cx"]], [0, profile["fy"], profile["cy"]], [0, 0, 1]]
    )
    projection_source = "Factory RGB 640x480 intrinsics; pinhole simulation without raw image distortion"
  elif a.camera_profile == "nominal-wide":
    image_width = 848
    K = np.array(
      [
        [image_width / (2 * np.tan(np.radians(87 / 2))), 0, (image_width - 1) / 2],
        [0, image_height / (2 * np.tan(np.radians(58 / 2))), (image_height - 1) / 2],
        [0, 0, 1],
      ]
    )
    projection_source = (
      "Theoretical 87x58 degree pinhole FOV at 848x480; not measured device calibration"
    )
  palmworld = Rotation.from_euler("y", 90, degrees=True).as_matrix()
  Rtagworld = palmworld @ Ttag
  rng = np.random.default_rng(a.seed)
  clip = wuji_hand_reorient_ppo_runner_cfg().clip_actions
  try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=clip)
    m = env.sim.mj_model
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "robot/d405_b")
    if a.camera_profile != "assembly":
      # Configure rendering and PnP from the same K, using pixel-center coordinates.
      ss = m.cam_sensorsize[cid]
      m.cam_resolution[cid] = [image_width, image_height]
      m.cam_intrinsic[cid] = [
        K[0, 0] * ss[0] / image_width,
        K[1, 1] * ss[1] / image_height,
        ((image_width - 1) / 2 - K[0, 2]) * ss[0] / image_width,
        ((image_height - 1) / 2 - K[1, 2]) * ss[1] / image_height,
      ]
      m.cam_fovy[cid] = np.degrees(2 * np.arctan(image_height / (2 * K[1, 1])))
    renderer = mujoco.Renderer(m, height=image_height, width=image_width)
    opt = mujoco.MjvOption()
    opt.geomgroup[3:] = 0
    robot = env.scene["robot"]
    palm_ids, _ = robot.find_bodies(".*_palm_link")
    palm_id = int(palm_ids[0])
    goal_visualization = ReorientCommandVisualization(
      entity_name="object",
      show_frames=False,
      show_status=True,
      goal_z_offset=0.09,
      ghost_mesh_size=np.full(3, 0.025),
    )
    command = env.command_manager.get_term("reorient_command")
    command._update_command = lambda: None
    totalsteps = round(a.seconds / env.step_dt)
    blocksteps = round(8 / env.step_dt)
    nwindows = totalsteps // blocksteps
    for trial in range(a.trials):
      wrapped.reset()
      command.time_left.fill_(1e9)
      goals = Rotation.random(nwindows, random_state=rng).as_matrix()
      cache = PoseCache(
        1, 0.15, max_position_speed_m_s=0.5, max_rotation_speed_rad_s=12
      )
      history = SensorHistory()
      lastaction = np.zeros(20)
      previous_action = np.zeros(20)
      visualerror = None
      success = np.zeros(nwindows, bool)
      truehold = 0
      window = -1
      freshcount = 0
      usablecount = 0
      holdcount = 0
      frames = 0
      doneflag = False
      missingrun = 0
      longest = 0
      visualhold = 0
      visually_reached = False
      for step in range(nwindows * blocksteps):
        block = step // blocksteps
        if block != window:
          window = block
          truehold = 0
          visualhold = 0
          visually_reached = False
          q = np.roll(Rotation.from_matrix(goals[block]).as_quat(), 1)
          command.goal_quat_w[:] = torch.tensor(q, device=env.device)
          command.time_left.fill_(1e9)
        # Rendering needs physics state; the observation builder only receives pixels.
        env._offline_renderer.update(env.sim.data)
        renderer.update_scene(env._offline_renderer._data, camera=cid, scene_option=opt)
        rgb = renderer.render().copy()
        ts = time.perf_counter()
        pose = est.estimate(rgb, K)
        detms = (time.perf_counter() - ts) * 1000
        fresh, usable, age = cache.update([pose], step * env.step_dt, Rcam, pcam)
        freshcount += int(fresh[0])
        usablecount += int(usable[0])
        missingrun = 0 if fresh[0] else missingrun + 1
        longest = max(longest, missingrun)
        # Encoders and the previously commanded target are allowed proprioception.
        q = robot.data.joint_pos[0].cpu().numpy()
        limits = robot.data.soft_joint_pos_limits[0].cpu().numpy()
        target = (
          env.action_manager.get_term("joint_pos").processed_action[0].cpu().numpy()
        )
        goal_tag = Rtagworld.T @ goals[block]
        position = cache.position[0].copy()
        rotation = cache.rotation[0].copy()
        x = history.update(
          q, limits, target, previous_action, position, rotation, goal_tag
        )
        if usable[0]:
          with torch.inference_mode():
            action = student(
              TensorDict(
                {"policy": torch.tensor(x, device=env.device)[None]}, batch_size=[1]
              )
            )
          mode = "VISUAL CONTROL"
          visualerror = float(Rotation.from_matrix(rotation @ goal_tag.T).magnitude())
        else:
          # Hold the last command on missing/stale vision; no truth-based fallback.
          action = torch.tensor(lastaction, dtype=torch.float32, device=env.device)[
            None
          ]
          mode = "HOLD / NO VISION"
          holdcount += 1
          visualerror = None
        if usable[0] and visualerror is not None and visualerror < 0.2:
          visualhold += 1
        else:
          visualhold = 0
        visually_reached = visually_reached or visualhold >= 5
        machine_state = (
          "NO VISION"
          if not usable[0]
          else "GOAL REACHED"
          if visually_reached
          else "HOLD"
          if visualhold > 0
          else "APPROACH"
        )
        if a.official_demo and trial == 0:
          # Both cameras show the same pre-action physics frame. Status and goal
          # marker use the camera estimate, never the ground-truth scoring error.
          palm_pose = robot.data.body_link_pose_w[:, palm_id, :]
          tag_origin = palm_pose[0, :3].cpu().numpy() + palmworld @ np.array(
            TAG_IN_PALM_POS
          )
          visual_world_position = tag_origin + Rtagworld @ position
          visual_world_rotation = Rtagworld @ rotation

          def draw_goal(
            visualizer,
            palm_pose=palm_pose,
            visual_world_position=visual_world_position,
            visualerror=visualerror,
            usable_now=bool(usable[0]),
            visual_world_rotation=visual_world_rotation,
            goal_rotation=goals[block].copy(),
          ):
            goal_visualization.draw_debug_visuals(
              visualizer=visualizer,
              num_envs=1,
              palm_pose_w=palm_pose,
              object_pos_w=torch.tensor(visual_world_position, device=env.device)[None],
              goal_quat_w=command.goal_quat_w,
              policy_status_for_env=lambda _i: (
                visualerror if visualerror is not None else np.pi,
                bool(usable_now and visualerror is not None and visualerror < 0.2),
              ),
            )
            # Use the same RGB frame convention as the official visualizer:
            # rotation columns are the local X/red, Y/green and Z/blue axes.
            target_position = visual_world_position + np.array(
              [0.0, 0.0, goal_visualization.goal_z_offset]
            )
            visualizer.add_frame(
              position=target_position,
              rotation_matrix=goal_rotation,
              scale=0.075,
              axis_radius=0.002,
              label="TARGET ORIENTATION",
            )
            if usable_now:
              visualizer.add_frame(
                position=visual_world_position,
                rotation_matrix=visual_world_rotation,
                scale=0.06,
                axis_radius=0.0015,
                label="OBSERVED CUBE",
              )

          env._offline_renderer.update(env.sim.data, debug_vis_callback=draw_goal)
          exterior = env._offline_renderer.render()
          if exterior.ndim == 4:
            exterior = exterior[0]
          external_image = Image.fromarray(np.asarray(exterior, dtype=np.uint8))
          ext_draw = ImageDraw.Draw(external_image)
          ext_draw.rectangle((0, 0, 960, 120), fill=(12, 20, 32))
          ext_draw.text(
            (20, 10),
            "EXTERNAL SIDE VIEW | PALM DOWN | CAMERA STUDENT + ENCODERS",
            fill="white",
          )
          states = ["APPROACH", "HOLD", "GOAL REACHED", "NO VISION"]
          for i, state in enumerate(states):
            box = (20 + i * 225, 36, 225 + i * 225, 70)
            color = (30, 145, 80) if state == "GOAL REACHED" else (35, 110, 170)
            ext_draw.rounded_rectangle(
              box, radius=6, fill=color if state == machine_state else (35, 45, 58)
            )
            ext_draw.text((box[0] + 12, 47), state, fill="white")
          angle_text = (
            f"Visual error {np.degrees(visualerror):.1f} deg"
            if visualerror is not None
            else "Pose unavailable: hold command"
          )
          ext_draw.text(
            (20, 84),
            f"Goal {block + 1}/{nwindows} | {angle_text} | stable {min(visualhold, 5)}/5 steps",
            fill="white",
          )
          ext_draw.text(
            (20, 104),
            "Upper cube = TARGET | lower cube = OBSERVED | RGB axes: X red, Y green, Z blue",
            fill="white",
          )
        if clip is not None:
          action = action.clamp(-clip, clip)
        previous_action = lastaction.copy()
        lastaction = action[0].detach().cpu().numpy().copy()
        _, _, done, _ = wrapped.step(action)
        frames += 1
        doneflag = bool(done[0])
        trueerror = None
        if not doneflag:
          truth = env.scene["object"].data.root_link_quat_w[0].cpu().numpy()
          trueR = Rotation.from_quat(truth[[1, 2, 3, 0]]).as_matrix()
          trueerror = float(Rotation.from_matrix(trueR @ goals[block].T).magnitude())
          truehold = truehold + 1 if trueerror < 0.2 else 0
          if truehold >= 5:
            success[block] = True
        rows.append(
          {
            "trial": trial,
            "time_s": (step + 1) * env.step_dt,
            "goal_window": block,
            "fresh": bool(fresh[0]),
            "usable": bool(usable[0]),
            "mode": mode,
            "detector_ms": detms,
            "visual_error_rad": visualerror,
            "scoring_true_error_rad": trueerror,
            "done": doneflag,
            "visual_state_machine": machine_state,
          }
        )
        if trial == 0:
          img = Image.fromarray(rgb)
          draw = ImageDraw.Draw(img)
          draw.rectangle((0, 0, image_width, 65), fill="black")
          draw.text((10, 5), f"STUDENT / CAMERA + ENCODERS | {mode}", fill="white")
          draw.text(
            (10, 24),
            f"Goal {block + 1}/{nwindows} | score {success.sum()} | tags {len(pose.ids) if pose else 0}",
            fill="white",
          )
          draw.text(
            (10, 43),
            f"vision angle {np.degrees(visualerror):.1f} deg"
            if visualerror is not None
            else "No fresh pose: hold last command",
            fill="white",
          )
          videos.append(np.asarray(img))
          if a.official_demo:
            wrist_panel = img.resize((960, 720))
            combined_videos.append(
              np.concatenate(
                (np.asarray(external_image), np.asarray(wrist_panel)), axis=1
              )
            )
        if doneflag:
          break
      t = {
        "trial": trial,
        "duration_s": frames * env.step_dt,
        "survived": not doneflag,
        "scheduled_goal_windows": nwindows,
        "scored_goal_successes": int(success.sum()),
        "goal_window_successes": success.tolist(),
        "fresh_fraction": freshcount / max(frames, 1),
        "usable_fraction": usablecount / max(frames, 1),
        "hold_fraction": holdcount / max(frames, 1),
        "longest_no_fresh_s": longest * env.step_dt,
        "rejected_visual_jumps": cache.rejected_jumps,
      }
      trials.append(t)
      print(json.dumps(t), flush=True)
    report = {
      "student": str(a.student.resolve()),
      "control_source": "207D student: camera AprilTag object pose, encoder joint angles, command target and action history; no object truth or privileged physical parameters in actor input.",
      "hand": "Official Wuji Hand 1 right_mjlab.xml",
      "camera_mount": json.loads(
        (
          ROOT
          / "src/wuji_mjlab/assets/robots/wuji_hand/meshes/wrist_assembly/source.json"
        ).read_text()
      ),
      "camera_calibration": {
        "profile": a.camera_profile,
        "projection_source": projection_source,
        "resolution": [image_width, image_height],
        "centered_fov_deg": np.degrees(
          2
          * np.arctan(
            np.array([image_width, image_height]) / (2 * np.array([K[0, 0], K[1, 1]]))
          )
        ).tolist(),
        "K": K.tolist(),
        "camera_to_tag_rotation": Rcam.tolist(),
        "camera_origin_tag_m": pcam.tolist(),
      },
      "goal_protocol": "Externally sampled full-SO(3) absolute targets; fixed 8s windows; true angular error<0.2rad for 5 consecutive control steps. Remaining scheduled windows after a drop count as failures.",
      "scoring_only": "Object ground truth and termination are used only for scoring/physics. Automatic truth-based goal advancement disabled.",
      "trials": trials,
      "trial_count": len(trials),
      "horizon_s": nwindows * 8,
      "survived_trials": sum(t["survived"] for t in trials),
      "survival_rate": sum(t["survived"] for t in trials) / len(trials),
      "goal_successes": sum(t["scored_goal_successes"] for t in trials),
      "goal_attempts": sum(t["scheduled_goal_windows"] for t in trials),
      "goal_success_rate": sum(t["scored_goal_successes"] for t in trials)
      / sum(t["scheduled_goal_windows"] for t in trials),
      "elapsed_wall_s": time.perf_counter() - start,
      "seed": a.seed,
      "real_hardware_test": False,
    }
    (out / "evaluation.json").write_text(json.dumps(report, indent=2))
    media.write_video(str(out / "camera_closed_loop.mp4"), videos, fps=20)
    if combined_videos:
      media.write_video(str(out / "official_dual_view.mp4"), combined_videos, fps=20)
      Image.fromarray(combined_videos[len(combined_videos) // 3]).save(
        out / "official_dual_view_preview.jpg"
      )
    with (out / "trajectory.csv").open("w") as f:
      w = csv.DictWriter(f, fieldnames=rows[0])
      w.writeheader()
      w.writerows(rows)
    print(json.dumps({k: v for k, v in report.items() if k != "trials"}, indent=2))
  finally:
    if renderer:
      renderer.close()
    env.close()


if __name__ == "__main__":
  main()
