"""Record an annotated dodecahedron reorientation demo at a fixed wrist pitch.

Uses the environment of the gated tilt curriculum
(``scripts/train/train_dodecahedron_tilt_curriculum.py``), including its reset
grasp schedule, so the video matches the evaluated setting.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import mediapy as media
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.lab_api.math import quat_error_magnitude
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from train.train_dodecahedron_tilt_curriculum import make_cfg
from wuji_mjlab.tasks.dodecahedron.env_cfg import wuji_hand_dodecahedron_ppo_cfg
from wuji_mjlab.tasks.reorient.mdp.command_visualization import (
  ReorientCommandVisualization,
)
from wuji_mjlab.tasks.reorient.mdp.metrics import fingertip_contact_count


def default_blend(pitch: float) -> float:
  """Reset grasp blend of gated_v2: linear from -90 to +20 deg."""
  return min(max((pitch + 90.0) / 110.0, 0.0), 1.0)


def default_grip(pitch: float) -> float:
  """Extra distal curl of gated_v2: 0.1 to 0.2 rad over +20 to +90 deg."""
  return 0.0 if pitch < 20.0 else 0.1 + 0.1 * (pitch - 20.0) / 70.0


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--pitch", type=float, default=90.0)
  parser.add_argument("--blend", type=float, help="Default: gated_v2 schedule")
  parser.add_argument("--grip", type=float, help="Default: gated_v2 schedule")
  parser.add_argument("--seconds", type=float, default=30.0)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--width", type=int, default=1280)
  parser.add_argument("--height", type=int, default=720)
  parser.add_argument("--azimuth", type=float, default=135.0)
  parser.add_argument("--elevation", type=float, default=-12.0)
  parser.add_argument("--distance", type=float, default=0.55)
  args = parser.parse_args()
  if not args.checkpoint.is_file():
    parser.error(f"Checkpoint not found: {args.checkpoint}")
  blend = default_blend(args.pitch) if args.blend is None else args.blend
  grip = default_grip(args.pitch) if args.grip is None else args.grip

  configure_torch_backends()
  cfg = make_cfg(1, args.pitch, args.seed, blend, grip)
  cfg.viewer.width = args.width
  cfg.viewer.height = args.height
  cfg.viewer.distance = args.distance
  cfg.viewer.azimuth = args.azimuth
  cfg.viewer.elevation = args.elevation
  cfg.commands["reorient_command"].debug_vis = False

  env = ManagerBasedRlEnv(cfg, device="cuda:0", render_mode="rgb_array")
  try:
    agent = wuji_hand_dodecahedron_ppo_cfg()
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    obs, _ = wrapped.reset()
    actor_cfg = asdict(agent.actor)
    for key in ("class_name", "cnn_cfg", "rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
      actor_cfg.pop(key, None)
    template = TensorDict({"policy": torch.zeros_like(obs["policy"])}, batch_size=[1])
    policy = MLPModel(
      template, {"policy": ["policy"]}, "policy", wrapped.num_actions, **actor_cfg
    ).to("cuda:0")
    checkpoint = torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)
    policy.load_state_dict(checkpoint["actor_state_dict"])
    policy.eval()

    command = env.command_manager.get_term("reorient_command")
    robot = env.scene["robot"]
    obj = env.scene["object"]
    palm_body_id = int(robot.find_bodies(".*_palm_link")[0][0])
    # Palm down puts the hand above the object, so draw the goal below it.
    goal_vis = ReorientCommandVisualization(
      entity_name="object",
      show_frames=False,
      show_status=True,
      goal_z_offset=-0.10 if args.pitch > 0.0 else 0.10,
    )
    try:
      font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
      bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
      font = bold = ImageFont.load_default()

    def goal_error() -> float:
      return float(quat_error_magnitude(obj.data.root_link_quat_w[:1], command.goal_quat[:1])[0])

    def draw_goal(visualizer) -> None:
      goal_vis.draw_debug_visuals(
        visualizer=visualizer,
        num_envs=1,
        palm_pose_w=robot.data.body_link_pose_w[:, palm_body_id, :],
        object_pos_w=obj.data.root_link_pos_w,
        goal_quat_w=command.goal_quat,
        policy_status_for_env=lambda _: (goal_error(), goal_error() < command.cfg.success_threshold),
      )

    frames, trajectory = [], []
    dropped_at = None
    with torch.inference_mode():
      for step in range(int(args.seconds / env.step_dt)):
        action = policy(TensorDict({"policy": obs["policy"]}, batch_size=[1]))
        obs, _, dones, _ = wrapped.step(action)
        error = goal_error()
        goals = int(command.goal_reach_count[0])
        contacts = int(fingertip_contact_count(env)[0])
        state = "HOLD" if bool(command.in_success_window[0]) else "APPROACH"
        time_s = (step + 1) * env.step_dt
        trajectory.append({
          "time_s": time_s, "goal_error_rad": error, "goal_reach_count": goals,
          "fingertip_contacts": contacts, "state": state,
        })
        env._offline_renderer.update(env.sim.data, debug_vis_callback=draw_goal)
        rendered = env._offline_renderer.render()
        frame = rendered[0] if rendered.ndim == 4 else rendered
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rounded_rectangle((20, 20, 560, 168), radius=14, fill=(10, 18, 30, 225))
        orientation = "PALM DOWN" if args.pitch >= 89.0 else "PALM UP" if args.pitch <= -89.0 else "TILTED"
        draw.text((38, 32), f"DODECAHEDRON  |  WRIST PITCH {args.pitch:+.0f} deg ({orientation})", font=bold, fill=(245, 248, 255, 255))
        state_color = (90, 235, 160, 255) if state == "HOLD" else (115, 205, 255, 255)
        draw.text((38, 62), f"STATE: {state}", font=font, fill=state_color)
        draw.text((38, 86), f"GOAL ERROR: {np.degrees(error):5.1f} deg   GOALS REACHED: {goals}", font=font, fill=(255, 220, 125, 255))
        contact_color = (34, 197, 94, 255) if contacts else (239, 68, 68, 255)
        draw.text((38, 110), f"FINGERTIP CONTACTS: {contacts}", font=font, fill=contact_color)
        draw.text((38, 134), f"t = {time_s:5.1f} s   (translucent shape = target pose)", font=font, fill=(190, 200, 215, 255))
        frames.append(np.asarray(image))
        if bool(dones[0]):
          dropped_at = time_s
          break
  finally:
    env.close()

  args.output.parent.mkdir(parents=True, exist_ok=True)
  media.write_video(str(args.output), frames, fps=round(1.0 / env.step_dt))
  with args.output.with_suffix(".csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=trajectory[0].keys())
    writer.writeheader()
    writer.writerows(trajectory)
  summary = {
    "checkpoint": str(args.checkpoint), "pitch_deg": args.pitch, "grasp_blend": blend,
    "grip_offset_rad": grip, "seed": args.seed, "duration_s": len(trajectory) * env.step_dt,
    "dropped_at_s": dropped_at, "goal_reach_count": trajectory[-1]["goal_reach_count"],
    "fingertip_contact_fraction": float(np.mean([r["fingertip_contacts"] > 0 for r in trajectory])),
  }
  args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
