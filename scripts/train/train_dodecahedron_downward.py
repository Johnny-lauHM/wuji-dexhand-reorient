"""Curriculum fine-tune a privileged-state Teacher for palm-down rotation."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends
from wuji_mjlab.rl.runner import WujiOnPolicyRunner
from wuji_mjlab.tasks.dodecahedron.env_cfg import (
  wuji_hand_dodecahedron_env_cfg,
  wuji_hand_dodecahedron_ppo_cfg,
)
from wuji_mjlab.tasks.reorient import mdp

DEFAULT_TEACHER = Path(
  "logs/rsl_rl/wuji_dodecahedron_sim2real/finetune_500_20260922/final.pt"
)
DEFAULT_PITCHES = tuple(float(pitch) for pitch in range(-75, 91, 5))


def positive_int(value):
  number = int(value)
  if number <= 0:
    raise argparse.ArgumentTypeError("must be positive")
  return number


def parse_pitches(value):
  try:
    pitches = tuple(float(item) for item in value.split(","))
  except ValueError as exc:
    raise argparse.ArgumentTypeError("pitches must be comma-separated numbers") from exc
  if not pitches:
    raise argparse.ArgumentTypeError("at least one pitch is required")
  if any(right <= left for left, right in zip(pitches, pitches[1:], strict=False)):
    raise argparse.ArgumentTypeError("pitches must be strictly increasing")
  return pitches


def make_cfg(num_envs, pitch, seed, grip_offset, contact_reward_weight):
  cfg = wuji_hand_dodecahedron_env_cfg(
    num_envs=num_envs,
    sim2real_domain_randomization=False,
    wrist_pitch_deg=pitch,
    grip_offset_rad=grip_offset,
    downward_grasp=True,
    object_palm_x_m=0.030,
    object_palm_y_m=0.009,
    object_palm_z_m=0.1085,
  )
  cfg.seed = seed
  # Start every stage from the known grasp. The policy must then maintain the
  # grasp against the progressively inverted gravity direction.
  cfg.events["reset_robot_joints"].params["position_range"] = {".*": (0.0, 0.0)}
  cfg.events["reset_object_pose"].params["pos_noise"] = 0.0015
  cfg.terminations["cage_drop"].params["max_outside_steps"] = 14
  cfg.rewards["cage_escape"].weight = -750.0
  cfg.rewards["palm_detach"].weight = 2.0
  if contact_reward_weight > 0.0:
    cfg.rewards["fingertip_contact"] = RewardTermCfg(
      func=mdp.fingertip_contact_count,
      weight=contact_reward_weight,
      params={"sensor_cfg": SceneEntityCfg("tip_object_contact")},
    )
  return cfg


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
  parser.add_argument("--num-envs", type=positive_int, default=1024)
  parser.add_argument("--iterations-per-stage", type=positive_int, default=75)
  parser.add_argument(
    "--pitches", type=parse_pitches, default=DEFAULT_PITCHES,
    help="Strictly increasing wrist pitches in degrees",
  )
  parser.add_argument("--seed", type=int, default=84)
  parser.add_argument("--grip-offset", type=float, default=0.0)
  parser.add_argument("--contact-reward-weight", type=float, default=0.0)
  parser.add_argument("--learning-rate", type=float, default=2.0e-5)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output", type=Path)
  parser.add_argument("--smoke", action="store_true")
  args = parser.parse_args()
  if args.smoke:
    args.num_envs, args.iterations_per_stage = 32, 2
  if not args.teacher.is_file():
    parser.error(f"Teacher checkpoint not found: {args.teacher}")
  if not args.device.startswith("cuda") or not torch.cuda.is_available():
    parser.error("Training requires an available CUDA device")

  configure_torch_backends()
  output = args.output or (
    Path("logs/rsl_rl/wuji_dodecahedron_downward")
    / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
  )
  output.mkdir(parents=True, exist_ok=False)
  (output / "launch.json").write_text(json.dumps(vars(args), default=str, indent=2))

  source = args.teacher.resolve()
  stages = []
  for index, pitch in enumerate(args.pitches):
    stage_dir = output / f"stage_{index + 1}_{pitch:+05.1f}deg"
    stage_dir.mkdir()
    cfg = make_cfg(
      args.num_envs,
      pitch,
      args.seed + index,
      args.grip_offset,
      args.contact_reward_weight,
    )
    agent = wuji_hand_dodecahedron_ppo_cfg()
    agent.seed = args.seed + index
    agent.experiment_name = "wuji_dodecahedron_downward"
    agent.run_name = f"DownwardPitch{pitch:+.0f}"
    agent.logger = "tensorboard"
    agent.upload_model = False
    agent.max_iterations = args.iterations_per_stage
    agent.algorithm.learning_rate = args.learning_rate
    agent.algorithm.entropy_coef = 3.0e-4
    agent.algorithm.num_mini_batches = min(32, args.num_envs)
    if args.smoke:
      agent.num_steps_per_env = 8
      agent.algorithm.num_learning_epochs = 1
    dump_yaml(stage_dir / "params/env.yaml", asdict(cfg))
    dump_yaml(stage_dir / "params/agent.yaml", asdict(agent))

    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
      obs, _ = env.reset()
      if obs["policy"].shape != (args.num_envs, 207):
        raise RuntimeError(f"Unexpected policy shape: {obs['policy'].shape}")
      if obs["critic"].shape != (args.num_envs, 407):
        raise RuntimeError(f"Unexpected critic shape: {obs['critic'].shape}")
      if not all(torch.isfinite(value).all() for value in obs.values()):
        raise RuntimeError("Nonfinite initial observation")
      wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
      runner = WujiOnPolicyRunner(wrapped, asdict(agent), str(stage_dir), args.device)
      runner.load(
        str(source),
        load_cfg={
          "actor": True,
          "critic": True,
          "optimizer": False,
          "iteration": False,
          "rnd": False,
        },
        map_location=args.device,
      )
      runner.learn(args.iterations_per_stage, init_at_random_ep_len=False)
      checkpoint = stage_dir / "final.pt"
      runner.save(str(checkpoint))
    finally:
      env.close()
    stages.append({"pitch_deg": pitch, "checkpoint": str(checkpoint.resolve())})
    source = checkpoint.resolve()

  final_checkpoint = output / "final.pt"
  shutil.copy2(source, final_checkpoint)
  report = {
    "source_teacher": str(args.teacher.resolve()),
    "checkpoint": str(final_checkpoint.resolve()),
    "observation_source": "privileged object state + proprioception",
    "target_wrist_pitch_deg": args.pitches[-1],
    "palm_orientation": "down" if args.pitches[-1] == 90.0 else "curriculum",
    "num_envs": args.num_envs,
    "iterations_per_stage": args.iterations_per_stage,
    "stages": stages,
  }
  (output / "result.json").write_text(json.dumps(report, indent=2))
  print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
  main()
