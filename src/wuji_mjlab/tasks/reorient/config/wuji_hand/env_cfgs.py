# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji Hand Reorient environment configurations.

Thin binding layer: attaches the Wuji Hand robot + cube object entities
and the robot-specific viewer target to the task cfg produced by
``make_reorient_env_cfg``. All task design (sensors, rewards, DR events,
play-mode overrides) lives in ``reorient_terms`` / ``reorient_env_cfg``.
"""

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg

from wuji_mjlab.assets.objects.inhand_object.object_cfg import get_inhand_object_cfg
from wuji_mjlab.assets.robots.wuji_hand.wuji_hand_cfg import get_wuji_hand_cfg
from wuji_mjlab.tasks.reorient.downward_grasp import DOWNWARD_GRASP_JOINT_POS
from wuji_mjlab.tasks.reorient.reorient_constants import (
  REORIENT_CUBE_INIT_STATE,
  REORIENT_ROBOT_INIT_STATE,
  REORIENT_ROBOT_ROOT_POS,
)
from wuji_mjlab.tasks.reorient.reorient_env_cfg import make_reorient_env_cfg


def wuji_hand_reorient_env_cfg(
  play: bool = False, num_envs: int = 8192
) -> ManagerBasedRlEnvCfg:
  """Create Wuji Hand Reorient task configuration."""
  cfg = make_reorient_env_cfg(play=play, num_envs=num_envs)
  cfg.scene.entities = {
    "robot": get_wuji_hand_cfg(),
    "object": get_inhand_object_cfg(),
  }
  cfg.scene.entities["robot"].init_state = REORIENT_ROBOT_INIT_STATE
  cfg.scene.entities["object"].init_state = REORIENT_CUBE_INIT_STATE
  cfg.viewer.body_name = "right_palm_link"
  return cfg


def wuji_hand_cube_downward_env_cfg(
  play: bool = False,
  num_envs: int = 4096,
  object_palm_pos: tuple[float, float, float] = (0.030, 0.009, 0.105),
  official_actor_compatible: bool = False,
  cube_edge_m: float | None = None,
  wrist_assembly: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Official cube reorientation task with a fixed palm-down wrist.

  The hand's grasp direction is local +X. R_y(+90 degrees) maps that
  direction toward world -Z, so gravity pulls the cube away from the palm.
  """
  cfg = wuji_hand_reorient_env_cfg(play=play, num_envs=num_envs)
  if wrist_assembly:
    cfg.scene.entities["robot"].spec_fn = get_wuji_hand_cfg(wrist_assembly=True).spec_fn
  cfg.scene.entities["object"] = get_inhand_object_cfg(cube_edge_m)
  cfg.scene.entities["robot"].init_state = replace(
    cfg.scene.entities["robot"].init_state,
    rot=(math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0),
    joint_pos={
      name: angle + (0.3 if name.endswith(("joint3", "joint4")) else 0.0)
      for name, angle in DOWNWARD_GRASP_JOINT_POS.items()
    },
  )
  x, y, z = object_palm_pos
  cfg.scene.entities["object"].init_state = replace(
    cfg.scene.entities["object"].init_state,
    pos=(REORIENT_ROBOT_ROOT_POS[0] + z, REORIENT_ROBOT_ROOT_POS[1] + y,
         REORIENT_ROBOT_ROOT_POS[2] - x),
  )
  cfg.events["reset_robot_pose"].params["pose_range"] = {}
  cfg.events["reset_robot_joints"].params["position_range"] = {".*": (0.0, 0.0)}
  cfg.events["reset_object_pose"].params["pos_noise"] = 0.0015
  if official_actor_compatible:
    # The release actor has 207 inputs. Here its object pose terms are exact
    # simulator values, so it is still a privileged-state teacher.
    cfg.observations["policy"].enable_corruption = False
    for term in cfg.observations["policy"].terms.values():
      if "injection_prob" in term.params:
        term.params["injection_prob"] = 0.0
  else:
    # The scratch teacher consumes all 413 critic features, including object
    # velocity and physical parameters that are unavailable on the real hand.
    cfg.observations["policy"] = cfg.observations["critic"]
  return cfg
