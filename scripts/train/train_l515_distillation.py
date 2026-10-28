"""Distill the converged simulation-state policy into an RGB-D policy."""
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
from rsl_rl.runners import DistillationRunner
from wuji_mjlab.tasks.dodecahedron.l515 import make_l515_distillation_configs

DEFAULT_TEACHER = Path("logs/rsl_rl/wuji_dodecahedron/2026-09-12_15-55-31_Dodecahedron/model_4999.pt")


def positive_int(value):
  number = int(value)
  if number <= 0:
    raise argparse.ArgumentTypeError("must be positive")
  return number


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
  parser.add_argument("--num-envs", type=positive_int, default=64)
  parser.add_argument("--iterations", type=positive_int, default=5000)
  parser.add_argument("--width", type=positive_int, default=128)
  parser.add_argument("--height", type=positive_int, default=96)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output", type=Path)
  parser.add_argument("--resume", type=Path, help="Resume a distillation checkpoint")
  parser.add_argument("--validation-stage", action="store_true", help="Teacher data only; evaluate student and stop for review")
  parser.add_argument("--smoke", action="store_true", help="4 envs and 2 updates")
  args = parser.parse_args()
  if args.smoke:
    args.num_envs, args.iterations = 4, 2
  if not args.device.startswith("cuda") or not torch.cuda.is_available():
    parser.error("GPU camera rendering requires an available CUDA device")
  if not args.teacher.is_file():
    parser.error(f"teacher checkpoint not found: {args.teacher}")

  configure_torch_backends()
  cfg, agent = make_l515_distillation_configs(args.num_envs, args.width, args.height, args.seed)
  agent["max_iterations"] = args.iterations
  if args.validation_stage:
    agent["algorithm"]["teacher_action_prob_start"] = 1.0
    agent["algorithm"]["teacher_action_prob_end"] = 1.0
  if args.smoke:
    agent["num_steps_per_env"] = 8
    agent["algorithm"]["gradient_length"] = 4
  output = args.output or Path("logs/rsl_rl") / agent["experiment_name"] / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
  output.mkdir(parents=True, exist_ok=False)
  dump_yaml(output / "params/env.yaml", asdict(cfg))
  dump_yaml(output / "params/agent.yaml", agent)
  (output / "launch.json").write_text(json.dumps(vars(args), default=str, indent=2))

  env = ManagerBasedRlEnv(cfg, device=args.device)
  try:
    obs, _ = env.reset()
    expected = {
      "teacher": (args.num_envs, 207),
      "policy": (args.num_envs, 186),
      "rgbd": (args.num_envs, 15, args.height, args.width),
      "visual_privileged": (args.num_envs, 27),
    }
    for name, shape in expected.items():
      if obs[name].shape != shape:
        raise RuntimeError(f"Unexpected {name} shape: {obs[name].shape}, want {shape}")
    if not all(torch.isfinite(value).all() for value in obs.values()):
      raise RuntimeError("Nonfinite initial observation")
    iio.imwrite(output / "camera_rgb.png", env.scene["l515"].data.rgb[0].cpu().numpy())
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent["clip_actions"])
    runner = DistillationRunner(wrapped, agent, str(output), args.device)
    if args.resume:
      runner.load(str(args.resume.resolve()), map_location=args.device)
    else:
      runner.load(str(args.teacher.resolve()), load_cfg={"teacher": True, "iteration": False}, map_location=args.device)
    student = runner.alg.student
    if hasattr(student, "teacher_obs_normalizer"):
      if any(p.requires_grad for p in student.mlp.parameters()):
        raise RuntimeError("Student teacher action path is not frozen")
      with torch.no_grad():
        probe = wrapped.get_observations()
        rebuilt = student.reconstruct_teacher_observation(probe, probe["visual_privileged"])
        expected = torch.cat((probe["policy"][:, :120], probe["teacher"][:, 120:147], probe["policy"][:, 120:180]), -1)
        torch.testing.assert_close(rebuilt, expected)
        torch.testing.assert_close(student.mlp(student.teacher_obs_normalizer(rebuilt)), runner.alg.teacher.mlp(runner.alg.teacher.obs_normalizer(expected)))
      action_before = {k: v.detach().clone() for k, v in student.mlp.state_dict().items()}
    teacher_before = {name: value.detach().clone() for name, value in runner.alg.teacher.state_dict().items()}
    student_before = [p.detach().clone() for p in runner.alg.student.parameters()]
    runner.learn(args.iterations, init_at_random_ep_len=True)
    if hasattr(student, "teacher_obs_normalizer"):
      if not all(torch.equal(action_before[k], v) for k, v in student.mlp.state_dict().items()):
        raise RuntimeError("Frozen action path changed")
    checkpoint = output / "final.pt"
    runner.save(str(checkpoint))
    teacher_unchanged = all(torch.equal(value, teacher_before[name]) for name, value in runner.alg.teacher.state_dict().items())
    student_updated = any(not torch.equal(before, after) for before, after in zip(student_before, runner.alg.student.parameters(), strict=True))
    if not teacher_unchanged:
      raise RuntimeError("Teacher weights changed during distillation")
    if not student_updated:
      raise RuntimeError("Student weights did not update")
    report = {"observation_shapes": {name: list(value.shape) for name, value in obs.items()}, "teacher_checkpoint": str(args.teacher.resolve()), "checkpoint": str(checkpoint.resolve()), "iterations": args.iterations, "teacher_unchanged": teacher_unchanged, "student_updated": student_updated, "convergence_validated": False}
    (output / "result.json").write_text(json.dumps(report, indent=2))
    if args.validation_stage:
      from rsl_rl.models.cnn_model import project_rotation_6d_rows
      runner.alg.eval_mode()
      wrapped.seed(args.seed + 1000)
      with torch.inference_mode():
        current, _ = wrapped.reset()
      alive = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
      success = torch.zeros_like(alive)
      angle_sum = position_sum = samples = 0.0
      with torch.inference_mode():
        for _ in range(env.max_episode_length):
          prediction = student.predict_auxiliary(current)
          target = current["visual_privileged"]
          _, predicted_r = project_rotation_6d_rows(prediction[:, 9:])
          _, target_r = project_rotation_6d_rows(target[:, 9:])
          relative = predicted_r @ target_r.transpose(-1, -2)
          angles = torch.acos(((relative.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1, 1))
          angle_sum += angles[alive].sum().item()
          position_sum += (prediction[:, :9]-target[:, :9]).abs()[alive].sum().item()
          samples += alive.sum().item()
          current, _, dones, _ = wrapped.step(student(current))
          success |= alive & (env.command_manager.get_term("reorient_command").goal_reach_count > 0)
          alive &= ~dones.bool()
          if not alive.any():
            break
      validation = {"seed": args.seed+1000, "teacher_probability": 0, "successes": int(success.sum()), "num_envs": args.num_envs, "rotation_mean_deg": angle_sum/max(samples*3,1)*180/3.141592653589793, "position_mae_m": position_sum/max(samples*9,1), "automatic_curriculum_advance": False}
      (output / "validation.json").write_text(json.dumps(validation, indent=2))
      print(validation)
    print(f"Output: {output.resolve()}")
  finally:
    env.close()


if __name__ == "__main__":
  main()
