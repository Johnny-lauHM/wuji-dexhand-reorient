"""Long curriculum run for full-SO(3) palm-down cube reorientation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN = REPO_ROOT / "scripts/train/train_cube_downward.py"
EVAL = REPO_ROOT / "scripts/eval_cube_downward.py"


def run_logged(command: list[str], log_path: Path) -> None:
  with log_path.open("w") as log_file:
    process = subprocess.run(
      command,
      cwd=REPO_ROOT,
      stdout=log_file,
      stderr=subprocess.STDOUT,
      check=False,
    )
  if process.returncode:
    raise subprocess.CalledProcessError(process.returncode, command)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--resume", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=2048)
  parser.add_argument("--hold-iterations", type=int, default=300)
  parser.add_argument("--small-goal-iterations", type=int, default=500)
  parser.add_argument("--medium-goal-iterations", type=int, default=1000)
  parser.add_argument("--full-so3-iterations", type=int, default=3500)
  parser.add_argument("--cube-edge-m", type=float, default=0.045)
  parser.add_argument("--seed", type=int, default=84)
  args = parser.parse_args()

  if args.output.exists():
    parser.error(f"Output directory already exists: {args.output}")
  if not args.resume.is_file():
    parser.error(f"Resume checkpoint does not exist: {args.resume}")
  if not (0.03 <= args.cube_edge_m <= 0.054):
    parser.error("cube-edge-m must be between 0.030 and 0.054 meters")
  if min(
    args.num_envs,
    args.hold_iterations,
    args.small_goal_iterations,
    args.medium_goal_iterations,
    args.full_so3_iterations,
  ) <= 0:
    parser.error("environment and iteration counts must be positive")

  root = args.output.resolve()
  root.mkdir(parents=True)
  manifest = {
    "cube_edge_m": args.cube_edge_m,
    "cube_mass_kg_at_baseline_density": 0.120 * (args.cube_edge_m / 0.054) ** 3,
    "num_envs": args.num_envs,
    "seed": args.seed,
    "resume": str(args.resume.resolve()),
    "stages": [],
    "goal_distribution": "uniform full SO(3) in the final stage",
  }

  stages = [
    ("01_hold_45mm", "hold", args.hold_iterations, None),
    ("02_rotate_to_08rad", "rotate", args.small_goal_iterations, 0.8),
    ("03_rotate_to_16rad", "rotate", args.medium_goal_iterations, 1.6),
    ("04_rotate_full_so3", "rotate", args.full_so3_iterations, None),
  ]
  source = args.resume.resolve()
  for stage_name, phase, iterations, goal_angle in stages:
    stage_output = root / stage_name
    command = [
      sys.executable,
      str(TRAIN),
      "--phase",
      phase,
      "--resume",
      str(source),
      "--num-envs",
      str(args.num_envs),
      "--iterations",
      str(iterations),
      "--seed",
      str(args.seed),
      "--cube-edge-m",
      str(args.cube_edge_m),
      "--output",
      str(stage_output),
    ]
    if goal_angle is not None:
      command.extend(("--goal-angle", str(goal_angle)))
    print(f"[START] {stage_name}: {phase}, {iterations} iterations", flush=True)
    run_logged(command, root / f"{stage_name}.log")
    source = stage_output / "final.pt"

    eval_output = root / f"eval_{stage_name}"
    eval_command = [
      sys.executable,
      str(EVAL),
      "--checkpoint",
      str(source),
      "--phase",
      phase,
      "--num-envs",
      "64",
      "--seconds",
      "30" if stage_name == "04_rotate_full_so3" else "10",
      "--seed",
      str(args.seed + 1000),
      "--cube-edge-m",
      str(args.cube_edge_m),
      "--output",
      str(eval_output),
    ]
    if goal_angle is not None:
      eval_command.extend(("--goal-angle", str(goal_angle)))
    run_logged(eval_command, root / f"eval_{stage_name}.log")
    report = json.loads((eval_output / "evaluation.json").read_text())
    manifest["stages"].append(
      {
        "name": stage_name,
        "phase": phase,
        "iterations": iterations,
        "goal_angle_rad": goal_angle,
        "checkpoint": str(source),
        "evaluation": str(eval_output / "evaluation.json"),
        "survival_rate": report["survival_rate"],
        "first_goal_success_rate": report["first_goal_success_rate"],
        "mean_min_goal_error_rad": report["mean_min_goal_error_rad"],
      }
    )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
      f"[DONE] {stage_name}: survival={report['survival_rate']:.3f}, "
      f"first-goal={report['first_goal_success_rate']:.3f}, "
      f"min-error={report['mean_min_goal_error_rad']:.3f} rad",
      flush=True,
    )

  (root / "result.json").write_text(
    json.dumps({"checkpoint": str(source), "manifest": str(root / "manifest.json")}, indent=2)
  )
  print(f"[COMPLETE] final full-SO(3) checkpoint: {source}", flush=True)


if __name__ == "__main__":
  main()
