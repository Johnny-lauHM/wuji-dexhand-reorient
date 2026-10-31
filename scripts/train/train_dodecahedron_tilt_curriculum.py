"""Gated wrist-tilt curriculum: palm-up dodecahedron Teacher to palm-down.

Each stage fine-tunes the privileged-state Teacher at one fixed wrist pitch and
must pass a held-out evaluation before the pitch increases. A failed stage is
retried; repeated failure halves the pitch step. The reset grasp blends from
the palm-up Teacher pose to the enclosing downward grasp as the wrist tilts, so
the initial-state distribution never jumps between consecutive stages.

Every training and evaluation run is a separate process: Warp does not release
device memory when an environment is closed, so stages in one process run out
of memory on small GPUs.

Pitch convention follows ``wuji_hand_dodecahedron_env_cfg``: -90 deg is palm
up, +90 deg is palm down.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

PALM_UP_OBJECT_POS = (0.060, 0.010, 0.140)
GRASP_OBJECT_POS = (0.030, 0.009, 0.1085)
EVAL_SEED = 4321


def grasp_blend(pitch: float, start: float, end: float, shape: str = "linear") -> float:
  """Weight of the downward grasp at a wrist pitch.

  Linear is the default: a probe at -60 deg showed a 0.15 blend already lifts
  survival from 0.52 to 0.97, while blends above 0.3 suppress rotation for a
  policy that has not adapted to them. Smoothstep grows too slowly early on.
  """
  t = min(max((pitch - start) / (end - start), 0.0), 1.0)
  return t if shape == "linear" else t * t * (3.0 - 2.0 * t)


def goal_directed_rotation_rate(env) -> torch.Tensor:
  """Object angular velocity projected onto the shortest goal-error axis."""
  from mjlab.utils.lab_api.math import quat_inv, quat_mul

  obj = env.scene["object"]
  goal = env.command_manager.get_term("reorient_command").goal_quat
  error_quat = quat_mul(goal, quat_inv(obj.data.root_link_quat_w))
  error_quat = torch.where(error_quat[:, :1] < 0.0, -error_quat, error_quat)
  axis = error_quat[:, 1:4]
  axis_norm = axis.norm(dim=-1, keepdim=True)
  axis = axis / axis_norm.clamp_min(1.0e-6)
  rate = (obj.data.root_link_ang_vel_w * axis).sum(dim=-1)
  return torch.where(axis_norm.squeeze(-1) > 0.1, rate.clamp(-2.0, 2.0), 0.0)


def alive_reward(env) -> torch.Tensor:
  return torch.ones(env.num_envs, device=env.device)


def make_cfg(num_envs: int, pitch: float, seed: int, blend: float, grip: float = 0.0):
  from mjlab.managers.reward_manager import RewardTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from wuji_mjlab.tasks.dodecahedron.env_cfg import wuji_hand_dodecahedron_env_cfg
  from wuji_mjlab.tasks.reorient import mdp
  from wuji_mjlab.tasks.reorient.downward_grasp import DOWNWARD_GRASP_JOINT_POS

  object_pos = [
    (1.0 - blend) * up + blend * down
    for up, down in zip(PALM_UP_OBJECT_POS, GRASP_OBJECT_POS, strict=True)
  ]
  cfg = wuji_hand_dodecahedron_env_cfg(
    num_envs=num_envs,
    wrist_pitch_deg=pitch,
    object_palm_x_m=object_pos[0],
    object_palm_y_m=object_pos[1],
    object_palm_z_m=object_pos[2],
  )
  cfg.seed = seed
  robot_state = cfg.scene.entities["robot"].init_state
  # Palm-up keys are regexes (".*_finger1_joint1"); the grasp uses full names.
  palm_up = {key.split("_", 1)[1]: value for key, value in robot_state.joint_pos.items()}
  robot_state.joint_pos = {
    name: (1.0 - blend) * palm_up[name.split("_", 1)[1]] + blend * angle
    for name, angle in DOWNWARD_GRASP_JOINT_POS.items()
  }
  # Extra curl of the distal joints, as in the palm-down cube grasp (+0.3).
  for name in robot_state.joint_pos:
    if name.endswith(("joint3", "joint4")):
      robot_state.joint_pos[name] += grip
  cfg.events["reset_robot_joints"].params["position_range"] = {".*": (0.0, 0.0)}
  cfg.events["reset_object_pose"].params["pos_noise"] = 0.0015
  cfg.terminations["cage_drop"].params["max_outside_steps"] = 14
  cfg.rewards["cage_escape"].weight = -750.0
  # Holding against gravity matters more as the palm turns over.
  cfg.rewards["palm_detach"].weight = 0.5 + 1.5 * blend
  cfg.rewards["fingertip_contact"] = RewardTermCfg(
    func=mdp.fingertip_contact_count,
    weight=0.5 + 1.5 * blend,
    params={"sensor_cfg": SceneEntityCfg("tip_object_contact")},
  )
  cfg.rewards["alive"] = RewardTermCfg(func=alive_reward, weight=2.0)
  cfg.rewards["directed_rotation"] = RewardTermCfg(
    func=goal_directed_rotation_rate, weight=5.0
  )
  return cfg


def train_stage(args) -> None:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.utils.os import dump_yaml
  from mjlab.utils.torch import configure_torch_backends
  from wuji_mjlab.rl.runner import WujiOnPolicyRunner
  from wuji_mjlab.tasks.dodecahedron.env_cfg import wuji_hand_dodecahedron_ppo_cfg

  configure_torch_backends()
  args.output.mkdir(parents=True, exist_ok=False)
  cfg = make_cfg(args.num_envs, args.pitch, args.seed, args.blend, args.grip)
  agent = wuji_hand_dodecahedron_ppo_cfg()
  agent.seed = args.seed
  agent.experiment_name = "wuji_dodecahedron_tilt"
  agent.run_name = f"TiltPitch{args.pitch:+.0f}"
  agent.max_iterations = args.iterations
  agent.algorithm.learning_rate = args.learning_rate
  agent.algorithm.entropy_coef = args.entropy_coef
  agent.algorithm.num_mini_batches = min(32, args.num_envs)
  if args.smoke:
    agent.num_steps_per_env = 8
    agent.algorithm.num_learning_epochs = 1
  dump_yaml(args.output / "params/env.yaml", asdict(cfg))
  dump_yaml(args.output / "params/agent.yaml", asdict(agent))

  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    obs, _ = env.reset()
    if obs["policy"].shape[-1] != 207 or obs["critic"].shape[-1] != 407:
      raise RuntimeError(f"Unexpected shapes: {obs['policy'].shape}, {obs['critic'].shape}")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = WujiOnPolicyRunner(wrapped, asdict(agent), str(args.output), "cuda:0")
    runner.load(
      str(args.source),
      load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False},
      map_location="cuda:0",
    )
    runner.learn(args.iterations, init_at_random_ep_len=False)
    runner.save(str(args.output / "final.pt"))
  finally:
    env.close()


def eval_stage(args) -> None:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.utils.torch import configure_torch_backends
  from rsl_rl.models import MLPModel
  from tensordict import TensorDict
  from wuji_mjlab.tasks.dodecahedron.env_cfg import wuji_hand_dodecahedron_ppo_cfg

  configure_torch_backends()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  cfg = make_cfg(args.num_envs, args.pitch, args.seed, args.blend, args.grip)
  agent = wuji_hand_dodecahedron_ppo_cfg()
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    obs, _ = wrapped.reset()
    actor_cfg = asdict(agent.actor)
    for key in ("class_name", "cnn_cfg", "rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
      actor_cfg.pop(key, None)
    template = TensorDict({"policy": torch.zeros_like(obs["policy"])}, batch_size=[args.num_envs])
    policy = MLPModel(
      template, {"policy": ["policy"]}, "policy", wrapped.num_actions, **actor_cfg
    ).to("cuda:0")
    checkpoint = torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)
    policy.load_state_dict(checkpoint["actor_state_dict"])
    policy.eval()

    command = env.command_manager.get_term("reorient_command")
    active = torch.ones(args.num_envs, dtype=torch.bool, device="cuda:0")
    goal_counts = torch.zeros(args.num_envs, dtype=torch.int32, device="cuda:0")
    termination_step = torch.full_like(goal_counts, -1)
    steps = int(args.seconds / env.step_dt)
    with torch.inference_mode():
      for step in range(steps):
        action = policy(TensorDict({"policy": obs["policy"]}, batch_size=[args.num_envs]))
        obs, _, dones, _ = wrapped.step(action)
        goal_counts += (active & command.success_achieved).int()
        ended = active & dones.bool()
        termination_step[ended] = step + 1
        active &= ~dones.bool()
    drops = termination_step[termination_step > 0].float() * env.step_dt
    report = {
      "checkpoint": str(args.checkpoint),
      "pitch_deg": args.pitch,
      "grasp_blend": args.blend,
      "num_envs": args.num_envs,
      "horizon_s": args.seconds,
      "survival_rate": float(active.float().mean()),
      "mean_goals_per_trial": float(goal_counts.float().mean()),
      "goal_rate_per_min": float(goal_counts.float().mean()) * 60.0 / args.seconds,
      "median_drop_s": float(drops.median()) if drops.numel() else None,
    }
  finally:
    env.close()
  args.output.write_text(json.dumps(report, indent=2))
  print(json.dumps(report), flush=True)


class Curriculum:
  def __init__(self, args):
    self.args = args
    self.root = args.output.resolve()
    self.manifest_path = self.root / "manifest.json"
    if self.manifest_path.is_file():
      self.state = json.loads(self.manifest_path.read_text())
      print(f"[curriculum] resuming at pitch {self.state['current_pitch']}", flush=True)
    else:
      self.root.mkdir(parents=True, exist_ok=False)
      self.state = {
        "teacher": str(args.teacher.resolve()),
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "func"},
        "current_pitch": args.start_pitch,
        "step": args.step,
        "best": str(args.teacher.resolve()),
        "run_index": 0,
        "elapsed_s": 0.0,
        "accepted": [],
        "attempts": [],
      }
    self.started = time.time() - self.state["elapsed_s"]

  def save(self) -> None:
    self.state["elapsed_s"] = time.time() - self.started
    self.manifest_path.write_text(json.dumps(self.state, indent=2))

  def grip(self, pitch: float) -> float:
    """Extra distal curl once the hand is fully closed; a probe at +27.5 deg
    showed 0.1 rad lifts survival from 0.56 to 1.0."""
    if pitch < self.args.grip_start_pitch:
      return 0.0
    t = (pitch - self.args.grip_start_pitch) / (self.args.final_pitch - self.args.grip_start_pitch)
    return self.args.grip_min + t * (self.args.grip_max - self.args.grip_min)

  def blend(self, pitch: float) -> float:
    return grasp_blend(pitch, self.args.blend_start, self.args.blend_end, self.args.blend_shape)

  def run(self, mode: str, log_name: str, extra: list[str]) -> None:
    command = [sys.executable, str(Path(__file__).resolve()), mode, *extra]
    with (self.root / f"{log_name}.log").open("w") as log:
      result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
      raise RuntimeError(f"{mode} failed, see {self.root / log_name}.log")

  def evaluate(self, name: str, checkpoint: str, pitch: float) -> dict:
    output = self.root / "eval" / f"{name}.json"
    self.run("eval", f"eval_{name}", [
      "--checkpoint", checkpoint, "--pitch", str(pitch), "--blend", str(self.blend(pitch)),
      "--grip", str(self.grip(pitch)),
      "--num-envs", str(self.args.eval_envs), "--seconds", str(self.args.eval_seconds),
      "--output", str(output),
    ])
    return json.loads(output.read_text())

  def passed(self, report: dict) -> bool:
    return (
      report["survival_rate"] >= self.args.min_survival
      and report["mean_goals_per_trial"] >= self.args.min_goals
    )

  def score(self, report: dict) -> float:
    return report["survival_rate"] + 0.1 * report["mean_goals_per_trial"]

  def train(self, source: str, pitch: float) -> tuple[str, str]:
    index = self.state["run_index"] = self.state["run_index"] + 1
    name = f"run{index:03d}_pitch{pitch:+06.1f}"
    extra = [
      "--source", source, "--pitch", str(pitch), "--blend", str(self.blend(pitch)),
      "--grip", str(self.grip(pitch)),
      "--num-envs", str(self.args.num_envs), "--iterations", str(self.args.iterations),
      "--seed", str(self.args.seed + index), "--learning-rate", str(self.args.learning_rate),
      "--entropy-coef", str(self.args.entropy_coef), "--output", str(self.root / "runs" / name),
    ]
    if self.args.smoke:
      extra.append("--smoke")
    print(f"[curriculum] train {name} from {Path(source).parent.name}", flush=True)
    self.run("train", name, extra)
    return str(self.root / "runs" / name / "final.pt"), name

  def out_of_time(self) -> bool:
    return time.time() - self.started > self.args.budget_hours * 3600.0

  def main(self) -> None:
    s = self.state
    if not s["accepted"]:
      report = self.evaluate("teacher", s["best"], s["current_pitch"])
      s["accepted"].append({"pitch": s["current_pitch"], "checkpoint": s["best"], "eval": report, "forced": False})
      self.save()
    while s["current_pitch"] < self.args.final_pitch and not self.out_of_time():
      target = min(s["current_pitch"] + s["step"], self.args.final_pitch)
      # After a failed gate, the best attempt at the harder pitch usually
      # already handles the halfway pitch: evaluate it first and train from it.
      source = s.pop("candidate", None) or s["best"]
      best_try = None
      if source != s["best"]:
        name = f"carry_{Path(source).parent.name}_to{target:+06.1f}"
        report = self.evaluate(name, source, target)
        s["attempts"].append({"pitch": target, "run": name, "round": 0, "eval": report})
        print(
          f"[curriculum] {name}: survival={report['survival_rate']:.2f} "
          f"goals={report['mean_goals_per_trial']:.2f}", flush=True,
        )
        best_try = (source, report)
      rounds = 0
      for attempt in range(1, self.args.max_rounds + 1):
        if best_try is not None and self.passed(best_try[1]):
          break
        rounds = attempt
        checkpoint, name = self.train(source, target)
        report = self.evaluate(name, checkpoint, target)
        s["attempts"].append({"pitch": target, "run": name, "round": attempt, "eval": report})
        print(
          f"[curriculum] {name}: survival={report['survival_rate']:.2f} "
          f"goals={report['mean_goals_per_trial']:.2f}", flush=True,
        )
        if best_try is None or self.score(report) > self.score(best_try[1]):
          best_try = (checkpoint, report)
        self.save()
        if self.passed(report) or self.out_of_time():
          break
        source = checkpoint
      checkpoint, report = best_try
      if self.passed(report):
        s["current_pitch"], s["best"] = target, checkpoint
        s["accepted"].append({"pitch": target, "checkpoint": checkpoint, "eval": report, "forced": False})
        if rounds <= 1:
          # Passed without retries: the pitch step can grow back.
          s["step"] = min(s["step"] * 2.0, self.args.step)
      elif s["step"] > self.args.min_step:
        s["step"] = max(s["step"] / 2.0, self.args.min_step)
        s["candidate"] = checkpoint
        print(f"[curriculum] gate failed at {target}; step -> {s['step']}", flush=True)
      else:
        # Minimum step already reached: keep moving with the best attempt so
        # the run always ends palm-down; flagged in the manifest.
        s["current_pitch"], s["best"] = target, checkpoint
        s["accepted"].append({"pitch": target, "checkpoint": checkpoint, "eval": report, "forced": True})
        print(f"[curriculum] forced advance to {target}", flush=True)
      self.save()

    sweep = {}
    for pitch in self.args.sweep_pitches:
      sweep[str(pitch)] = self.evaluate(f"final_sweep_{pitch:+.0f}", s["best"], pitch)
    s["final_sweep"] = sweep
    s["finished"] = s["current_pitch"] >= self.args.final_pitch
    self.save()
    print(f"[curriculum] done, final checkpoint {s['best']}", flush=True)


def parse_pitch_list(value: str) -> list[float]:
  return [float(item) for item in value.split(",")]


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  sub = parser.add_subparsers(dest="mode", required=True)

  run = sub.add_parser("run", help="Run the gated curriculum")
  run.add_argument("--teacher", type=Path, required=True)
  run.add_argument("--output", type=Path, required=True)
  run.add_argument("--start-pitch", type=float, default=-90.0)
  run.add_argument("--final-pitch", type=float, default=90.0)
  run.add_argument("--step", type=float, default=15.0)
  run.add_argument("--min-step", type=float, default=5.0)
  run.add_argument("--blend-start", type=float, default=-90.0)
  run.add_argument("--blend-end", type=float, default=90.0)
  run.add_argument("--blend-shape", choices=("linear", "smooth"), default="linear")
  run.add_argument("--grip-start-pitch", type=float, default=20.0)
  run.add_argument("--grip-min", type=float, default=0.1)
  run.add_argument("--grip-max", type=float, default=0.2)
  run.add_argument("--num-envs", type=int, default=2048)
  run.add_argument("--iterations", type=int, default=150, help="Iterations per training round")
  run.add_argument("--max-rounds", type=int, default=4)
  run.add_argument("--learning-rate", type=float, default=5.0e-5)
  run.add_argument("--entropy-coef", type=float, default=1.0e-3)
  run.add_argument("--seed", type=int, default=84)
  run.add_argument("--eval-envs", type=int, default=64)
  run.add_argument("--eval-seconds", type=float, default=20.0)
  run.add_argument("--min-survival", type=float, default=0.8)
  run.add_argument("--min-goals", type=float, default=2.0, help="Mean goals per eval trial")
  run.add_argument("--budget-hours", type=float, default=6.5)
  run.add_argument("--sweep-pitches", type=parse_pitch_list, default=[-90.0, -45.0, 0.0, 45.0, 90.0])
  run.add_argument("--smoke", action="store_true")

  train = sub.add_parser("train")
  evaluate = sub.add_parser("eval")
  for p in (train, evaluate):
    p.add_argument("--pitch", type=float, required=True)
    p.add_argument("--blend", type=float, required=True)
    p.add_argument("--grip", type=float, default=0.0)
    p.add_argument("--num-envs", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
  train.add_argument("--source", type=Path, required=True)
  train.add_argument("--iterations", type=int, required=True)
  train.add_argument("--seed", type=int, required=True)
  train.add_argument("--learning-rate", type=float, required=True)
  train.add_argument("--entropy-coef", type=float, required=True)
  train.add_argument("--smoke", action="store_true")
  evaluate.add_argument("--checkpoint", type=Path, required=True)
  evaluate.add_argument("--seconds", type=float, required=True)
  evaluate.add_argument("--seed", type=int, default=EVAL_SEED)

  args = parser.parse_args()
  if not torch.cuda.is_available():
    parser.error("Requires CUDA")
  if args.mode == "run":
    if args.smoke:
      args.num_envs, args.iterations, args.max_rounds = 32, 2, 1
      args.eval_envs, args.eval_seconds = 8, 1.0
    Curriculum(args).main()
  elif args.mode == "train":
    train_stage(args)
  else:
    eval_stage(args)


if __name__ == "__main__":
  main()
