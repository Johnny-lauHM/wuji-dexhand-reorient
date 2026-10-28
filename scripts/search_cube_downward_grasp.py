"""Screen fixed palm-down cube grasp starts with batched zero-action rollouts."""

from __future__ import annotations

import itertools

import torch
from mjlab.envs import ManagerBasedRlEnv
from wuji_mjlab.tasks.reorient.config.wuji_hand.env_cfgs import (
  wuji_hand_cube_downward_env_cfg,
)
from wuji_mjlab.tasks.reorient.downward_grasp import DOWNWARD_GRASP_JOINT_POS


def main() -> None:
  candidates = list(itertools.product((0.02, 0.03, 0.04), (0.075, 0.09, 0.105, 0.12), (0.0, 0.15, 0.3), (0.5, 1.0)))
  repeats = 8
  num_envs = len(candidates) * repeats
  cfg = wuji_hand_cube_downward_env_cfg(num_envs=num_envs)
  cfg.seed = 321
  cfg.events = {name: term for name, term in cfg.events.items() if term.mode == "reset"}
  cfg.events["reset_robot_joints"].params["position_range"] = {".*": (0.0, 0.0)}
  cfg.events["reset_object_pose"].params["pos_noise"] = 0.0
  cfg.curriculum.clear()
  cfg.observations["policy"].enable_corruption = False
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    env.reset()
    robot, cube = env.scene["robot"], env.scene["object"]
    action = env.action_manager.get_term("joint_pos")
    q_official = robot.data.default_joint_pos[0].clone()
    q_grip = torch.tensor(
      [DOWNWARD_GRASP_JOINT_POS[name] for name in robot.joint_names],
      device=env.device,
    )
    all_q = []
    all_pos = []
    for x, z, offset, blend in candidates:
      q = q_official * (1.0 - blend) + q_grip * blend
      q = q.clone()
      for joint_index, name in enumerate(robot.joint_names):
        if name.endswith(("joint3", "joint4")):
          q[joint_index] += offset
      all_q.extend([q] * repeats)
      all_pos.extend([(z, 0.009, 0.5 - x)] * repeats)
    joint_pos = torch.stack(all_q).clamp(
      robot.data.soft_joint_pos_limits[:, :, 0],
      robot.data.soft_joint_pos_limits[:, :, 1],
    )
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
    action._default_joint_pos = joint_pos.clone()
    action._prev_target = joint_pos.clone()
    pose = cube.data.root_link_pose_w.clone()
    pose[:, :3] = torch.tensor(all_pos, device=env.device) + env.scene.env_origins
    cube.write_root_link_pose_to_sim(pose)
    alive = torch.ones(num_envs, device=env.device, dtype=torch.bool)
    death_step = torch.full((num_envs,), 201, device=env.device, dtype=torch.int32)
    with torch.inference_mode():
      for step in range(200):
        _, _, terminated, truncated, _ = env.step(
          torch.zeros(num_envs, 20, device=env.device)
        )
        ended = alive & (terminated | truncated)
        death_step[ended] = step + 1
        alive &= ~(terminated | truncated)
    survival = (death_step.reshape(-1, repeats) > 200).sum(dim=1).cpu().tolist()
    mean_steps = death_step.reshape(-1, repeats).float().mean(dim=1).cpu().tolist()
    ranked = sorted(
      zip(candidates, survival, mean_steps, strict=True),
      key=lambda item: (item[1], item[2]),
      reverse=True,
    )
    for candidate, count, mean in ranked[:20]:
      print(f"candidate={candidate} survival={count}/{repeats} mean_steps={mean:.1f}")
  finally:
    env.close()


if __name__ == "__main__":
  main()
