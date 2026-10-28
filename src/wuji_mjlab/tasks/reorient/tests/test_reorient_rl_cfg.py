# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

from dataclasses import asdict

from wuji_mjlab.tasks.reorient.config.wuji_hand.env_cfgs import (
  wuji_hand_cube_downward_env_cfg,
)
from wuji_mjlab.tasks.reorient.config.wuji_hand.rsl_rl.ppo import (
  wuji_hand_reorient_ppo_runner_cfg,
)


def test_reorient_rl_cfg_sets_explicit_min_std():
  runner_cfg = wuji_hand_reorient_ppo_runner_cfg()

  dist_cfg = asdict(runner_cfg)["actor"]["distribution_cfg"]
  assert dist_cfg["min_std"] == 0.2


def test_cube_downward_teacher_has_inverted_gravity_and_full_state():
  cfg = wuji_hand_cube_downward_env_cfg(num_envs=4)
  wrist = cfg.scene.entities["robot"].init_state
  cube = cfg.scene.entities["object"].init_state

  assert wrist.rot[2] > 0.0
  assert cube.pos[2] < wrist.pos[2]
  assert "cube_angvel" in cfg.observations["policy"].terms
  assert "true_joint_vel" in cfg.observations["policy"].terms
  assert cfg.events["reset_robot_pose"].params["pose_range"] == {}


def test_cube_downward_can_load_official_privileged_actor():
  cfg = wuji_hand_cube_downward_env_cfg(
    num_envs=4, official_actor_compatible=True
  )
  policy = cfg.observations["policy"]

  assert not policy.enable_corruption
  assert "cube_angvel" not in policy.terms
  assert "cube_angvel" in cfg.observations["critic"].terms
  assert all(term.params.get("injection_prob", 0.0) == 0.0 for term in policy.terms.values())
