"""Fine-tune the state Teacher under moderate sim-to-real randomization."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends
from wuji_mjlab.rl.runner import WujiOnPolicyRunner
from wuji_mjlab.tasks.dodecahedron.env_cfg import (
  wuji_hand_dodecahedron_env_cfg,
  wuji_hand_dodecahedron_ppo_cfg,
)

DEFAULT_TEACHER = Path(
  "logs/rsl_rl/wuji_dodecahedron/2026-09-12_15-55-31_Dodecahedron/model_4999.pt"
)


def positive_int(value):
  number = int(value)
  if number <= 0:
    raise argparse.ArgumentTypeError("must be positive")
  return number


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
  parser.add_argument("--num-envs", type=positive_int, default=1024)
  parser.add_argument("--iterations", type=positive_int, default=500)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output", type=Path)
  parser.add_argument("--smoke", action="store_true")
  args = parser.parse_args()
  if args.smoke:
    args.num_envs, args.iterations = 32, 2
  if not args.teacher.is_file():
    parser.error(f"Teacher checkpoint not found: {args.teacher}")
  if not args.device.startswith("cuda") or not torch.cuda.is_available():
    parser.error("Training requires an available CUDA device")

  configure_torch_backends()
  cfg = wuji_hand_dodecahedron_env_cfg(
    num_envs=args.num_envs, sim2real_domain_randomization=True
  )
  cfg.seed = args.seed
  agent = wuji_hand_dodecahedron_ppo_cfg()
  agent.seed = args.seed
  agent.experiment_name = "wuji_dodecahedron_sim2real"
  agent.run_name = "DodecahedronSim2Real"
  agent.logger = "tensorboard"
  agent.upload_model = False
  agent.max_iterations = args.iterations
  agent.algorithm.learning_rate = 3.0e-5
  agent.algorithm.entropy_coef = 5.0e-4
  agent.algorithm.num_mini_batches = min(32, args.num_envs)
  if args.smoke:
    agent.num_steps_per_env = 8
    agent.algorithm.num_learning_epochs = 1

  output = args.output or (
    Path("logs/rsl_rl")
    / agent.experiment_name
    / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
  )
  output.mkdir(parents=True, exist_ok=False)
  dump_yaml(output / "params/env.yaml", asdict(cfg))
  dump_yaml(output / "params/agent.yaml", asdict(agent))
  (output / "launch.json").write_text(json.dumps(vars(args), default=str, indent=2))

  env = ManagerBasedRlEnv(cfg, device=args.device)
  try:
    obs, _ = env.reset()
    expected = {
      "policy": (args.num_envs, 207),
      "critic": (args.num_envs, 407),
    }
    for name, shape in expected.items():
      if obs[name].shape != shape:
        raise RuntimeError(f"Unexpected {name} shape: {obs[name].shape}, want {shape}")
      if not torch.isfinite(obs[name]).all():
        raise RuntimeError(f"Nonfinite initial observation: {name}")

    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = WujiOnPolicyRunner(wrapped, asdict(agent), str(output), args.device)
    runner.load(
      str(args.teacher.resolve()),
      load_cfg={
        "actor": True,
        "critic": True,
        "optimizer": False,
        "iteration": False,
        "rnd": False,
      },
      map_location=args.device,
    )
    source = torch.load(args.teacher, map_location=args.device, weights_only=False)
    loaded_actor = runner.alg.actor.state_dict()
    if not all(
      torch.equal(value, loaded_actor[name])
      for name, value in source["actor_state_dict"].items()
    ):
      raise RuntimeError("Pretrained Teacher actor was not loaded exactly")

    runner.learn(args.iterations, init_at_random_ep_len=True)
    checkpoint = output / "final.pt"
    runner.save(str(checkpoint))
    report = {
      "source_teacher": str(args.teacher.resolve()),
      "checkpoint": str(checkpoint.resolve()),
      "num_envs": args.num_envs,
      "iterations": args.iterations,
      "domain_randomization": True,
      "actor_sensor_corruption": True,
      "learning_rate": agent.algorithm.learning_rate,
    }
    (output / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
