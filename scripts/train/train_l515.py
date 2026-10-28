"""Train the dodecahedron from single-view RGB-D; no physical camera needed."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v3 as iio
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends
from wuji_mjlab.rl.runner import WujiOnPolicyRunner
from wuji_mjlab.tasks.dodecahedron.l515 import make_l515_configs


def positive_int(value):
  number = int(value)
  if number <= 0:
    raise argparse.ArgumentTypeError("must be positive")
  return number


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--num-envs", type=positive_int, default=64)
  parser.add_argument("--iterations", type=positive_int, default=5000)
  parser.add_argument("--width", type=positive_int, default=128)
  parser.add_argument("--height", type=positive_int, default=96)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output", type=Path)
  parser.add_argument("--resume", type=Path, help="Only this RGB-D model's checkpoint")
  parser.add_argument(
    "--smoke", action="store_true", help="4 envs, 2 updates + reload check"
  )
  args = parser.parse_args()
  if args.smoke:
    args.num_envs, args.iterations = 4, 2
  if not args.device.startswith("cuda") or not torch.cuda.is_available():
    parser.error("GPU camera rendering requires an available CUDA device")
  configure_torch_backends()
  cfg, agent = make_l515_configs(args.num_envs, args.width, args.height, args.seed)
  agent.max_iterations = args.iterations
  if args.smoke:
    agent.num_steps_per_env = 8
    agent.algorithm.num_learning_epochs = 1
  output = args.output or Path(
    "logs/rsl_rl"
  ) / agent.experiment_name / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
  output.mkdir(parents=True, exist_ok=False)
  dump_yaml(output / "params/env.yaml", asdict(cfg))
  dump_yaml(output / "params/agent.yaml", asdict(agent))
  (output / "launch.json").write_text(json.dumps(vars(args), default=str, indent=2))
  env = ManagerBasedRlEnv(cfg, device=args.device)
  try:
    obs, _ = env.reset()
    for name, value in obs.items():
      if not torch.isfinite(value).all():
        raise RuntimeError(f"Nonfinite initial observation: {name}")
    if obs["policy"].shape != (args.num_envs, 186):
      raise RuntimeError(f"Unexpected actor state shape: {obs['policy'].shape}")
    if obs["rgbd"].shape != (args.num_envs, 5, args.height, args.width):
      raise RuntimeError(f"Unexpected image shape: {obs['rgbd'].shape}")
    data = env.scene["l515"].data
    if args.smoke:
      # A floor-only view can have valid depth too. Require near-field hand/
      # object pixels in every independent world, plus inspect camera_rgb.png.
      visible = (
        ((data.depth >= 0.25) & (data.depth < 0.55)).flatten(start_dim=1).sum(dim=1)
      )
      if not (visible > 10).all():
        raise RuntimeError(
          f"No near-field hand/object in initial camera views: {visible}"
        )
    iio.imwrite(output / "camera_rgb.png", data.rgb[0].cpu().numpy())
    iio.imwrite(
      output / "camera_depth.png",
      (obs["rgbd"][0, 3].cpu().numpy() * 255).astype("uint8"),
    )
    if not (obs["rgbd"][:, 4].mean() > 0):
      raise RuntimeError("Camera has no valid depth in the configured range")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = WujiOnPolicyRunner(wrapped, asdict(agent), str(output), args.device)
    if args.resume:
      runner.load(str(args.resume.resolve()), map_location=args.device)
    before = None
    if args.smoke:
      before = [p.detach().clone() for p in runner.alg.actor.cnns.parameters()]
    runner.learn(args.iterations, init_at_random_ep_len=True)
    checkpoint = output / "final.pt"
    runner.save(str(checkpoint))
    report = {
      "observation_shapes": {k: list(v.shape) for k, v in obs.items()},
      "checkpoint": str(checkpoint),
      "iterations": args.iterations,
      "convergence_validated": False,
    }
    if args.smoke:
      after = list(runner.alg.actor.cnns.parameters())
      if not all(torch.isfinite(p).all() for p in runner.alg.actor.parameters()):
        raise RuntimeError("Nonfinite trained actor")
      if not any(not torch.equal(a, b) for a, b in zip(before, after, strict=True)):
        raise RuntimeError("CNN weights were not updated by PPO")
      observation = wrapped.get_observations()
      with torch.inference_mode():
        expected = runner.get_inference_policy()(observation).clone()
      replay = WujiOnPolicyRunner(wrapped, asdict(agent), device=args.device)
      replay.load(str(checkpoint), map_location=args.device)
      with torch.inference_mode():
        actual = replay.get_inference_policy()(observation)
      torch.testing.assert_close(actual, expected)
      report.update(cnn_updated=True, checkpoint_replay="passed")
    (output / "result.json").write_text(json.dumps(report, indent=2))
    print(f"Output: {output.resolve()}")
  finally:
    env.close()


if __name__ == "__main__":
  main()
