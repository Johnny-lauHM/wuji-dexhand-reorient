"""Record an annotated palm-down SO(3) reorientation demonstration."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import mediapy as media
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.lab_api.math import quat_error_magnitude
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from train.train_cube_downward import make_cfg
from wuji_mjlab.tasks.reorient.config.wuji_hand.rsl_rl.ppo import (
  wuji_hand_reorient_ppo_runner_cfg,
)
from wuji_mjlab.tasks.reorient.mdp.command_visualization import (
  ReorientCommandVisualization,
)
from wuji_mjlab.tasks.reorient.mdp.metrics import fingertip_contact_count


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--checkpoint",
    type=Path,
    default=Path(
      "logs/rsl_rl/wuji_cube_downward/anypose_rewardfix_20260923_v1/"
      "rotate_full_so3_8000/final.pt"
    ),
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=Path("logs/rsl_rl/wuji_cube_downward/demo_annotated_20260923/palm_down_anypose_state_machine.mp4"),
  )
  parser.add_argument("--phase", choices=("rotate", "spin"), default="rotate")
  parser.add_argument("--seconds", type=float, default=30.0)
  parser.add_argument("--seed", type=int, default=444)
  parser.add_argument("--width", type=int, default=1280)
  parser.add_argument("--height", type=int, default=720)
  parser.add_argument("--azimuth", type=float, default=90.0)
  parser.add_argument("--elevation", type=float, default=-28.0)
  parser.add_argument("--distance", type=float, default=0.38)
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--env-index", type=int, default=0)
  parser.add_argument("--cube-edge-m", type=float, default=0.045)
  parser.add_argument("--wrist-assembly", action="store_true")
  parser.add_argument("--cube-mass-kg", type=float)
  parser.add_argument("--wrist-video", type=Path, help="Record D405 view alongside external demo; requires --wrist-assembly")
  args = parser.parse_args()
  if args.wrist_video and not args.wrist_assembly:
    parser.error("--wrist-video requires --wrist-assembly")
  if not args.checkpoint.is_file():
    parser.error(f"Checkpoint not found: {args.checkpoint}")
  if args.seconds <= 0 or args.width <= 0 or args.height <= 0 or args.num_envs <= 0:
    parser.error("seconds, width, height, and num-envs must be positive")
  if not (0 <= args.env_index < args.num_envs):
    parser.error("env-index must be in [0, num-envs)")

  configure_torch_backends()
  cfg = make_cfg(
    args.phase,
    num_envs=args.num_envs,
    seed=args.seed,
    gravity_scale=1.0,
    cube_edge_m=args.cube_edge_m,
    cube_mass_kg=args.cube_mass_kg,
    wrist_assembly=args.wrist_assembly,
  )
  cfg.viewer.width = args.width
  cfg.viewer.height = args.height
  cfg.viewer.distance = args.distance
  cfg.viewer.azimuth = args.azimuth
  cfg.viewer.elevation = args.elevation
  cfg.viewer.max_extra_envs = 0
  cfg.viewer.env_idx = args.env_index
  cfg.commands["reorient_command"].debug_vis = False

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg, device=device, render_mode="rgb_array")
  wrist_renderer = None
  try:
    agent = wuji_hand_reorient_ppo_runner_cfg()
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
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    policy.load_state_dict(checkpoint["actor_state_dict"])
    policy.eval()

    frames: list[np.ndarray] = []
    wrist_frames: list[np.ndarray] = []
    wrist_renderer = None
    if args.wrist_video:
      wrist_renderer = mujoco.Renderer(env.sim.mj_model, height=480, width=640)
      wrist_camera_id = mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "robot/d405_b")
      if wrist_camera_id < 0:
        raise RuntimeError("Extracted D405 camera not found")
    trajectory: list[dict[str, float | int]] = []
    accumulated_spin = 0.0
    switch_flash_steps = max(1, round(0.5 / env.step_dt))
    switch_flash_remaining = 0
    command = env.command_manager.get_term("reorient_command")
    robot = env.scene["robot"]
    palm_ids, _ = robot.find_bodies(".*_palm_link")
    if not palm_ids:
      raise RuntimeError("Could not find the hand palm body for the target-pose overlay")
    palm_body_id = int(palm_ids[0])
    goal_visualization = ReorientCommandVisualization(
      entity_name="object",
      show_frames=False,
      show_status=True,
      goal_z_offset=0.10,
      ghost_mesh_size=np.full(3, 0.030 * args.cube_edge_m / 0.054),
    )
    steps = int(args.seconds / env.step_dt)
    cube = env.scene["object"]
    try:
      font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
      small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
      bold_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except OSError:
      font = small_font = bold_font = ImageFont.load_default()
    with torch.inference_mode():
      for step in range(steps):
        action = policy(
          TensorDict({"policy": obs["policy"]}, batch_size=[args.num_envs])
        )
        obs, _, dones, _ = wrapped.step(action)
        axial_rate = float(cube.data.root_link_ang_vel_w[args.env_index, 2])
        accumulated_spin += axial_rate * env.step_dt
        contact_count = int(fingertip_contact_count(env)[args.env_index].item())
        goal_error_rad = float(
          quat_error_magnitude(
            cube.data.root_link_quat_w[args.env_index : args.env_index + 1],
            command.goal_quat[args.env_index : args.env_index + 1],
          )[0].item()
        )
        goal_count = int(command.goal_reach_count[args.env_index].item())
        in_success_window = bool(command.in_success_window[args.env_index].item())
        hold_counter = int(command.hold_counter[args.env_index].item())
        hold_progress = int(command.reward_hold_counter_snapshot[args.env_index].item())
        window_timer = int(command.window_timer[args.env_index].item())
        goal_switched = bool(command.goal_switched[args.env_index].item())
        if goal_switched:
          switch_flash_remaining = switch_flash_steps
        else:
          switch_flash_remaining = max(0, switch_flash_remaining - 1)
        goal_quat = command.goal_quat[args.env_index].detach().cpu().numpy()
        if goal_switched:
          machine_state = "NEXT GOAL"
        elif in_success_window:
          machine_state = "HOLD"
        else:
          machine_state = "APPROACH"
        trajectory.append(
          {
            "time_s": (step + 1) * env.step_dt,
            "axial_rate_rad_s": axial_rate,
            "accumulated_spin_rad": accumulated_spin,
            "fingertip_contacts": contact_count,
            "goal_error_rad": goal_error_rad,
            "goal_reach_count": goal_count,
            "state_machine_state": machine_state,
            "approach_hold_steps": hold_counter,
            "success_window_steps": window_timer,
            "goal_quat_w": float(goal_quat[0]),
            "goal_quat_x": float(goal_quat[1]),
            "goal_quat_y": float(goal_quat[2]),
            "goal_quat_z": float(goal_quat[3]),
          }
        )

        def draw_goal_pose(visualizer) -> None:
          goal_visualization.draw_debug_visuals(
            visualizer=visualizer,
            num_envs=args.num_envs,
            palm_pose_w=robot.data.body_link_pose_w[:, palm_body_id, :],
            object_pos_w=cube.data.root_link_pos_w,
            goal_quat_w=command.goal_quat,
            policy_status_for_env=lambda env_idx: (
              float(
                quat_error_magnitude(
                  cube.data.root_link_quat_w[env_idx : env_idx + 1],
                  command.goal_quat[env_idx : env_idx + 1],
                )[0].item()
              ),
              bool(
                quat_error_magnitude(
                  cube.data.root_link_quat_w[env_idx : env_idx + 1],
                  command.goal_quat[env_idx : env_idx + 1],
                )[0].item()
                < command.cfg.success_threshold
              ),
            ),
          )

        # Add only the translucent target cube and its green/red status marker;
        # do not enable the default cage and contact debug overlays.
        env._offline_renderer.update(env.sim.data, debug_vis_callback=draw_goal_pose)
        rendered = env._offline_renderer.render()
        if wrist_renderer is not None:
          wrist_renderer.update_scene(env._offline_renderer._data, camera=wrist_camera_id)
          wrist_frames.append(wrist_renderer.render().copy())
        frame = rendered[args.env_index] if rendered.ndim == 4 else rendered
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
        draw = ImageDraw.Draw(image, "RGBA")
        panel_right = min(args.width - 20, 650)
        draw.rounded_rectangle((22, 20, panel_right, 244), radius=16, fill=(10, 18, 30, 228))
        title = "PALM-DOWN FULL SO(3) REORIENTATION" if args.phase == "rotate" else "PALM-DOWN CUBE SPIN"
        draw.text((42, 32), f"{title}  |  1g", font=bold_font, fill=(245, 248, 255, 255))
        contact_ok = contact_count > 0
        status_color = (34, 197, 94, 255) if contact_ok else (239, 68, 68, 255)
        draw.text((43, 58), "STATE MACHINE", font=small_font, fill=(160, 180, 205, 255))
        states = ("APPROACH", "HOLD", "NEXT GOAL")
        active_state = "NEXT GOAL" if switch_flash_remaining > 0 else machine_state
        box_y1, box_y2, box_w, box_gap = 78, 110, 132, 14
        for state_i, state_name in enumerate(states):
          x1 = 43 + state_i * (box_w + box_gap)
          x2 = x1 + box_w
          is_active = state_name == active_state
          box_color = (34, 105, 160, 255) if is_active else (35, 48, 65, 255)
          if is_active and state_name == "HOLD":
            box_color = (26, 128, 80, 255)
          draw.rounded_rectangle((x1, box_y1, x2, box_y2), radius=8, fill=box_color)
          text_box = draw.textbbox((0, 0), state_name, font=small_font)
          text_w = text_box[2] - text_box[0]
          draw.text((x1 + (box_w - text_w) / 2, 87), state_name, font=small_font, fill=(248, 250, 255, 255))
          if state_i < 2:
            mid_y = (box_y1 + box_y2) // 2
            draw.line((x2 + 2, mid_y, x2 + box_gap - 3, mid_y), fill=(125, 150, 180, 255), width=2)
            draw.polygon(((x2 + box_gap - 3, mid_y), (x2 + box_gap - 8, mid_y - 4), (x2 + box_gap - 8, mid_y + 4)), fill=(125, 150, 180, 255))
        if in_success_window:
          stage_text = f"HOLD WINDOW: {hold_progress}/{max(command.cfg.goal_switch_delay, 1)} steps"
          stage_color = (90, 235, 160, 255) if goal_error_rad < command.cfg.success_threshold else (255, 190, 95, 255)
        else:
          stage_text = f"APPROACH: {hold_counter}/{command.cfg.success_hold_steps} stable steps"
          stage_color = (115, 205, 255, 255)
        draw.text((43, 120), stage_text, font=small_font, fill=stage_color)
        if switch_flash_remaining > 0:
          draw.text((370, 120), "NEW RANDOM SO(3) GOAL", font=small_font, fill=(255, 220, 125, 255))
        draw.text((43, 144), f"GOAL ERROR: {np.degrees(goal_error_rad):.1f} deg  |  GOALS REACHED: {goal_count}", font=font, fill=(255, 220, 125, 255))
        draw.text((43, 166), "TARGET QUAT WORLD (w, x, y, z): " + ", ".join(f"{q:+.2f}" for q in goal_quat), font=small_font, fill=(205, 215, 230, 255))
        draw.ellipse((43, 190, 57, 204), fill=status_color)
        draw.text((66, 188), f"FINGERTIP CONTACT: {contact_count}  {'GRASP' if contact_ok else 'LOST'}", font=font, fill=status_color)
        turn_text = f"NET AXIAL ROTATION: {accumulated_spin:+.2f} rad ({accumulated_spin / (2 * np.pi):+.2f} turns)"
        draw.text((43, 214), turn_text, font=small_font, fill=(100, 210, 255, 255))
        frames.append(np.asarray(image))
        if bool(dones[args.env_index]):
          break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(str(args.output), frames, fps=round(1.0 / env.step_dt))
    if wrist_renderer is not None:
      args.wrist_video.parent.mkdir(parents=True, exist_ok=True)
      media.write_video(str(args.wrist_video), wrist_frames, fps=round(1.0 / env.step_dt))
    trajectory_path = args.output.with_suffix(".csv")
    with trajectory_path.open("w", newline="") as handle:
      writer = csv.DictWriter(handle, fieldnames=trajectory[0].keys())
      writer.writeheader()
      writer.writerows(trajectory)
    axial_rates = np.asarray([row["axial_rate_rad_s"] for row in trajectory])
    longest_stall_steps = 0
    current_stall_steps = 0
    for rate in axial_rates:
      if rate < 0.2:
        current_stall_steps += 1
        longest_stall_steps = max(longest_stall_steps, current_stall_steps)
      else:
        current_stall_steps = 0
    summary = {
      "duration_s": len(trajectory) * env.step_dt,
      "wrist_assembly": args.wrist_assembly,
      "cube_mass_kg": args.cube_mass_kg,
      "wrist_video": str(args.wrist_video) if args.wrist_video else None,
      "goal_reach_count": trajectory[-1]["goal_reach_count"],
      "final_goal_error_rad": trajectory[-1]["goal_error_rad"],
      "mean_axial_rate_rad_s": float(axial_rates.mean()),
      "positive_rate_fraction": float((axial_rates > 0.0).mean()),
      "rate_over_0_5_rad_s_fraction": float((axial_rates > 0.5).mean()),
      "longest_stall_s_below_0_2_rad_s": longest_stall_steps * env.step_dt,
      "fingertip_contact_fraction": float(
        np.mean([row["fingertip_contacts"] > 0 for row in trajectory])
      ),
      "net_spin_rad": accumulated_spin,
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"[INFO] Saved {len(frames)} frames to {args.output}")
    print(f"[INFO] Trajectory: {trajectory_path}")
    print(f"[INFO] Final accumulated axial rotation: {accumulated_spin:.3f} rad")
    print(json.dumps(summary, indent=2))
  finally:
    if wrist_renderer is not None:
      wrist_renderer.close()
    env.close()


if __name__ == "__main__":
  main()
