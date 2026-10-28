"""Simulation-only SO(3) reorientation, sharing the existing Wuji controller."""

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg

from wuji_mjlab.assets.objects.dodecahedron.object_cfg import get_dodecahedron_cfg
from wuji_mjlab.tasks.reorient.config.wuji_hand.env_cfgs import (
  wuji_hand_reorient_env_cfg,
)
from wuji_mjlab.tasks.reorient.config.wuji_hand.rsl_rl.ppo import (
  wuji_hand_reorient_ppo_runner_cfg,
)
from wuji_mjlab.tasks.reorient.downward_grasp import DOWNWARD_GRASP_JOINT_POS
from wuji_mjlab.tasks.reorient.reorient_constants import (
  REORIENT_CUBE_INIT_STATE,
  REORIENT_ROBOT_ROOT_POS,
)


def wuji_hand_dodecahedron_env_cfg(
  play: bool = False,
  num_envs: int = 4096,
  sim2real_domain_randomization: bool = False,
  wrist_pitch_deg: float = -90.0,
  grip_offset_rad: float = 0.0,
  downward_grasp: bool = False,
  object_palm_x_m: float = 0.060,
  object_palm_y_m: float = 0.010,
  object_palm_z_m: float = 0.140,
) -> ManagerBasedRlEnvCfg:
  cfg = wuji_hand_reorient_env_cfg(play=play, num_envs=num_envs)
  cfg.scene.entities["object"] = get_dodecahedron_cfg()
  # Rotate the fixed wrist about world Y while preserving the object pose in
  # the palm frame. -90 deg is the original palm-up task; +90 deg is the
  # fully inverted palm-down task.
  pitch = math.radians(wrist_pitch_deg)
  wrist_quat = (math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0)
  robot_state = cfg.scene.entities["robot"].init_state
  joint_pos = (
    dict(DOWNWARD_GRASP_JOINT_POS)
    if downward_grasp
    else dict(robot_state.joint_pos)
  )
  if grip_offset_rad:
    for name in joint_pos:
      if name.endswith(("joint3", "joint4")):
        joint_pos[name] += grip_offset_rad
  cfg.scene.entities["robot"].init_state = replace(
    robot_state, rot=wrist_quat, joint_pos=joint_pos
  )
  object_in_palm = (object_palm_x_m, object_palm_y_m, object_palm_z_m)
  cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch)
  object_pos = (
    REORIENT_ROBOT_ROOT_POS[0]
    + cos_pitch * object_in_palm[0]
    + sin_pitch * object_in_palm[2],
    REORIENT_ROBOT_ROOT_POS[1] + object_in_palm[1],
    REORIENT_ROBOT_ROOT_POS[2]
    - sin_pitch * object_in_palm[0]
    + cos_pitch * object_in_palm[2],
  )
  cfg.scene.entities["object"].init_state = replace(
    REORIENT_CUBE_INIT_STATE, pos=object_pos
  )
  cfg.viewer.entity_name = "object"
  cfg.viewer.body_name = "cube"
  cfg.viewer.distance = 0.65
  cfg.viewer.elevation = -30.0
  cfg.viewer.max_extra_envs = 0
  if play:
    cfg.scene.num_envs = 1
  # The nominal task stays deterministic. The opt-in profile retains a moderate
  # subset of the base task DR for frozen-policy robustness evaluation and
  # sim-to-real fine-tuning, without interval pushes or mesh-size changes.
  if sim2real_domain_randomization:
    dr_events = {
      "object_com",
      "robot_friction",
      "contact_params_palm_thumb",
      "contact_params_fingers",
      "object_mass",
      "pd_gains",
      "robot_dof_damping",
      "robot_dof_armature",
      "robot_dof_frictionloss",
      "encoder_bias",
      "robot_link_mass",
    }
    cfg.events = {
      name: term
      for name, term in cfg.events.items()
      if term.mode == "reset" or name in dr_events
    }
    cfg.events["object_com"].params["ranges"] = (-0.0015, 0.0015)
    cfg.events["robot_friction"].params["ranges"] = (0.85, 1.15)
    cfg.events["object_mass"].params["scale_range"] = (0.8, 1.2)
    cfg.events["pd_gains"].params["kp_range"] = (0.9, 1.1)
    cfg.events["pd_gains"].params["kd_range"] = (0.8, 1.25)
    cfg.events["robot_dof_damping"].params["ranges"] = (0.7, 1.4)
    cfg.events["robot_dof_armature"].params["ranges"] = (0.9, 1.1)
    cfg.events["robot_dof_frictionloss"].params["ranges"] = (0.75, 1.3)
    cfg.events["encoder_bias"].params["bias_range"] = (-0.005, 0.005)
    # Scale each link mass and inertia by the same sample. Independent body
    # mass randomization leaves an inconsistent inertia tensor.
    cfg.events["robot_link_mass"].func = cfg.events["object_mass"].func
    cfg.events["robot_link_mass"].params = {
      "asset_cfg": cfg.events["robot_link_mass"].params["asset_cfg"],
      "scale_range": (0.9, 1.1),
    }
  else:
    cfg.events = {
      name: term for name, term in cfg.events.items() if term.mode == "reset"
    }
  cfg.curriculum.clear()
  cfg.events["reset_robot_pose"].params["pose_range"] = {}
  cfg.events["reset_object_pose"].params["pos_noise"] = 0.003
  for name, group in cfg.observations.items():
    # Sim-to-real fine-tuning keeps actor sensor noise enabled. Critic and
    # nominal evaluation observations remain clean.
    group.enable_corruption = sim2real_domain_randomization and name == "policy"
    if not sim2real_domain_randomization:
      for term in group.terms.values():
        if "injection_prob" in term.params:
          term.params["injection_prob"] = 0.0
  # Keep the critic topology checkpoint-compatible (407 dimensions). The
  # randomized physical parameters affect transitions without being revealed.
  cfg.observations["critic"].terms.pop("dr_params")
  return cfg


def wuji_hand_dodecahedron_ppo_cfg():
  cfg = wuji_hand_reorient_ppo_runner_cfg(run_name="Dodecahedron")
  cfg.experiment_name = "wuji_dodecahedron"
  cfg.logger = "tensorboard"
  cfg.upload_model = False
  return cfg
