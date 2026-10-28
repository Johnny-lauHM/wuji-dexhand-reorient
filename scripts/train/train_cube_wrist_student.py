"""Distill the palm-down teacher into a 207D sensor-observation student."""

from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
from dataclasses import asdict
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.train_cube_downward import make_cfg
from wuji_mjlab.tasks.reorient.config.wuji_hand.rsl_rl.ppo import (
  wuji_hand_reorient_ppo_runner_cfg,
)

ROOT = Path(__file__).resolve().parents[2]
TEACHER = (
  ROOT
  / "logs/rsl_rl/wuji_cube_downward/anypose_rewardfix_20260923_v1/rotate_full_so3_8000/final.pt"
)


def model(dim, n, device):
  ac = asdict(wuji_hand_reorient_ppo_runner_cfg().actor)
  for k in ["class_name", "cnn_cfg", "rnn_type", "rnn_hidden_dim", "rnn_num_layers"]:
    ac.pop(k, None)
  return MLPModel(
    TensorDict({"policy": torch.zeros(n, dim, device=device)}, batch_size=[n]),
    {"policy": ["policy"]},
    "policy",
    20,
    **ac,
  ).to(device)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--output", type=Path, required=True)
  p.add_argument("--num-envs", type=int, default=256)
  p.add_argument("--rollout-steps", type=int, default=384)
  p.add_argument("--updates", type=int, default=3000)
  p.add_argument("--seed", type=int, default=9401)
  a = p.parse_args()
  a.output.mkdir(parents=True, exist_ok=False)
  configure_torch_backends()
  torch.manual_seed(a.seed)
  teacher = model(413, a.num_envs, "cuda:0")
  teacher.load_state_dict(
    torch.load(TEACHER, map_location="cuda:0", weights_only=False)["actor_state_dict"]
  )
  teacher.eval()
  student = model(207, a.num_envs, "cuda:0")
  state = teacher.state_dict()
  adapt = {
    k: (
      v[:, :207].clone()
      if k
      in [
        "obs_normalizer._mean",
        "obs_normalizer._var",
        "obs_normalizer._std",
        "mlp.0.weight",
      ]
      else v.clone()
    )
    for k, v in state.items()
  }
  student.load_state_dict(adapt)
  cfg = make_cfg(
    "rotate",
    a.num_envs,
    a.seed,
    cube_edge_m=0.045,
    cube_mass_kg=0.028192,
    wrist_assembly=True,
  )
  terms = list(cfg.observations["policy"].terms)
  assert terms[:5] == [
    "joint_angles",
    "qpos_error",
    "cube_pos_in_tag",
    "cube_ori_error",
    "action_history",
  ]
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  xs = []
  ys = []
  start = time.perf_counter()
  try:
    wrapped = RslRlVecEnvWrapper(
      env, clip_actions=wuji_hand_reorient_ppo_runner_cfg().clip_actions
    )
    obs, _ = wrapped.reset()
    with torch.inference_mode():
      for t in range(a.rollout_steps):
        target = teacher(TensorDict({"policy": obs["policy"]}, batch_size=[a.num_envs]))
        xs.append(obs["policy"][:, :207].clone())
        ys.append(target.clone())
        obs, _, _, _ = wrapped.step(target)
        if t % 128 == 0:
          print(f"collect {t}/{a.rollout_steps}", flush=True)
    x = torch.cat(xs).detach().clone()
    y = torch.cat(ys).detach().clone()
    xs.clear()
    ys.clear()
    split = int(len(x) * 0.9)
    trainx = x[:split]
    trainy = y[:split]
    validx = x[split:]
    validy = y[split:]
    optimizer = torch.optim.Adam(student.parameters(), lr=2e-4)
    best = float("inf")
    for update in range(a.updates):
      ids = torch.randint(len(trainx), (1024,), device="cuda:0")
      batch = trainx[ids].clone()
      batch[:, 120:129] += torch.randn_like(batch[:, 120:129]) * 0.0006
      batch[:, 129:147] += torch.randn_like(batch[:, 129:147]) * 0.005
      predicted = student(TensorDict({"policy": batch}, batch_size=[len(batch)]))
      loss = torch.nn.functional.mse_loss(predicted, trainy[ids])
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
      optimizer.step()
      if update % 250 == 0 or update == a.updates - 1:
        with torch.no_grad():
          v = float(
            torch.nn.functional.mse_loss(
              student(TensorDict({"policy": validx}, batch_size=[len(validx)])), validy
            )
          )
        print(f"update {update} loss={loss.item():.6f} validation={v:.6f}", flush=True)
        if v < best:
          best = v
          torch.save(
            {
              "actor_state_dict": student.state_dict(),
              "input_dim": 207,
              "teacher": str(TEACHER),
              "training": "State-supervised behavior cloning; camera closed-loop evaluation required",
              "update": update,
            },
            a.output / "student.pt",
          )
    report = {
      "samples": len(x),
      "updates": a.updates,
      "validation_action_mse": best,
      "elapsed_s": time.perf_counter() - start,
      "seed": a.seed,
      "input_dim": 207,
      "student_inputs": [
        "joint angle history",
        "command target error history",
        "visual cube position history",
        "visual goal orientation error history",
        "previous action history",
      ],
      "training_pose_source": "Privileged teacher pose used only for supervised training; evaluation uses actual camera detection.",
    }
    (a.output / "training.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
