"""Single-view RGB-D training baseline; not a calibrated L515 simulator."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

import numpy as np
import torch
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_inv, quat_mul
from scipy.spatial.transform import Rotation

from wuji_mjlab.assets.objects.dodecahedron.object_cfg import apriltag_dodecahedron_spec

from .env_cfg import wuji_hand_dodecahedron_env_cfg, wuji_hand_dodecahedron_ppo_cfg


def camera_quaternion(position, target):
  """MuJoCo camera looks along local -Z, with local +Y up."""
  backward = np.asarray(position) - np.asarray(target)
  backward /= np.linalg.norm(backward)
  right = np.cross([0.0, 0.0, 1.0], backward)
  right /= np.linalg.norm(right)
  up = np.cross(backward, right)
  xyzw = Rotation.from_matrix(np.column_stack((right, up, backward))).as_quat()
  return tuple(float(x) for x in xyzw[[3, 0, 1, 2]])


def goal_in_camera(env, camera_quat):
  """Goal command only: never reads the actual object's pose."""
  goal = env.command_manager.get_term("reorient_command").goal_quat
  camera = torch.tensor(camera_quat, device=goal.device).expand_as(goal)
  rotation = matrix_from_quat(quat_mul(quat_inv(camera), goal))
  return rotation.flatten(start_dim=1)[:, 3:]


def encode_rgbd(rgb, depth, noise_std=0.0, missing_prob=0.0):
  """BCHW: RGB [0,1], optical-axis depth / 1 m, valid-depth mask.

  Fixed range gates and whole-image input need no truth segmentation or pose.
  Noise is a provisional augmentation, not measured L515 error statistics.
  """
  valid = torch.isfinite(depth) & (depth >= 0.25) & (depth <= 1.0)
  measured = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
  if noise_std:
    measured = measured + torch.randn_like(measured) * noise_std
  if missing_prob:
    valid = valid & (torch.rand_like(measured) >= missing_prob)
  measured = torch.where(valid, measured.clamp(0.25, 1.0), 0.0)
  return (
    torch.cat((rgb.float() / 255.0, measured, valid.float()), dim=-1)
    .permute(0, 3, 1, 2)
    .contiguous()
  )


def rgbd_observation(env, noise_std=0.002, missing_prob=0.02):
  data = env.scene["l515"].data
  return encode_rgbd(data.rgb, data.depth, noise_std, missing_prob)


class RGBDHistoryObservation:
  """Three RGB-D frames with reset-safe per-environment backfilling."""

  def __init__(self, cfg, env):
    del cfg
    camera = env.cfg.scene.sensors[-1]
    self.history = torch.zeros(
      (env.num_envs, 3, 5, camera.height, camera.width),
      device=env.device,
    )
    self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.last_step = None

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self.history[ids] = 0
    self.initialized[ids] = False
    self.last_step = None

  def __call__(self, env, noise_std=0.002, missing_prob=0.02):
    # During manager shape discovery the sensor may not have rendered yet.
    if not hasattr(env, "common_step_counter"):
      return self.history.flatten(1, 2)
    step = env.common_step_counter
    if self.last_step == step:
      return self.history.flatten(1, 2)
    frame = rgbd_observation(env, noise_std, missing_prob)
    self.history[:, :-1] = self.history[:, 1:].clone()
    self.history[:, -1] = frame
    new_ids = ~self.initialized
    if new_ids.any():
      self.history[new_ids] = frame[new_ids, None].expand(-1, 3, -1, -1, -1)
      self.initialized[new_ids] = True
    self.last_step = step
    return self.history.flatten(1, 2)


def make_l515_configs(
  num_envs=64,
  width=128,
  height=96,
  seed=42,
  sim2real_domain_randomization=False,
):
  """Build independent configs, leaving the state-based task untouched."""
  cfg = wuji_hand_dodecahedron_env_cfg(
    num_envs=num_envs,
    sim2real_domain_randomization=sim2real_domain_randomization,
  )
  cfg.seed = seed
  # Separate MJWarp worlds need no grid offset. A worldbody camera otherwise
  # observes the origin while the hands are translated to different grid cells.
  cfg.scene.env_spacing = 0.0
  cfg.scene.entities["object"].spec_fn = apriltag_dodecahedron_spec
  target = (-0.140, 0.010, 0.560)
  position = (0.060, -0.210, 0.820)
  quaternion = camera_quaternion(position, target)
  camera = CameraSensorCfg(
    name="l515",
    pos=position,
    quat=quaternion,
    fovy=55.0,
    width=width,
    height=height,
    data_types=("rgb", "depth"),
    use_textures=True,
    enabled_geom_groups=(0, 1, 2),
  )
  cfg.scene.sensors = (*cfg.scene.sensors, camera)
  policy = cfg.observations["policy"]
  policy.terms.pop("cube_pos_in_tag")
  policy.terms.pop("cube_ori_error")
  policy.terms["goal_in_camera"] = ObservationTermCfg(
    func=goal_in_camera,
    params={"camera_quat": quaternion},
  )
  policy.enable_corruption = True
  cfg.observations["rgbd"] = ObservationGroupCfg(
    terms={
      "image": ObservationTermCfg(
        func=RGBDHistoryObservation,
        # A stateful term preserves BCHW layout while keeping reset-safe history.
        params={"noise_std": 0.002, "missing_prob": 0.02},
      )
    },
    concatenate_dim=1,
  )
  agent = wuji_hand_dodecahedron_ppo_cfg()
  agent.seed = seed
  agent.experiment_name = "wuji_dodecahedron_l515_rgbd"
  agent.obs_groups = {"actor": ("policy", "rgbd"), "critic": ("critic",)}
  agent.actor.class_name = "CNNModel"
  agent.actor.cnn_cfg = {
    "output_channels": (16, 32, 32),
    "kernel_size": (5, 3, 3),
    "stride": (2, 2, 2),
    "padding": "zeros",
    "flatten": True,
  }
  agent.num_steps_per_env = 24
  agent.algorithm.num_mini_batches = min(8, num_envs)
  return cfg, agent


def make_l515_distillation_configs(num_envs=64, width=128, height=96, seed=42):
  """Build synchronized RGB-D student and simulation-state teacher views."""
  cfg, _ = make_l515_configs(num_envs, width, height, seed)
  # Rebuild the unmodified state observation: make_l515_configs already
  # removed privileged pose terms from policy.
  state_cfg = wuji_hand_dodecahedron_env_cfg(num_envs=num_envs)
  cfg.observations["teacher"] = deepcopy(state_cfg.observations["policy"])
  cfg.observations["visual_privileged"] = ObservationGroupCfg(
    terms={
      "cube_pos": deepcopy(
        state_cfg.observations["critic"].terms["cube_pos_in_tag_clean"]
      ),
      "cube_ori_error": deepcopy(
        state_cfg.observations["critic"].terms["cube_ori_error_clean"]
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )
  for term in cfg.observations["visual_privileged"].terms.values():
    term.history_length = 3
    term.flatten_history_dim = True

  ppo = asdict(wuji_hand_dodecahedron_ppo_cfg())
  teacher = deepcopy(ppo["actor"])
  teacher.pop("cnn_cfg", None)
  student = ppo["actor"]
  for model in (student, teacher):
    model.pop("rnn_type", None)
    model.pop("rnn_hidden_dim", None)
    model.pop("rnn_num_layers", None)
  student["class_name"] = "VisionStateTeacherModel"
  student["auxiliary_output_dim"] = 27
  student["cnn_cfg"] = {
    "output_channels": (16, 32, 32),
    "kernel_size": (5, 3, 3),
    "stride": (2, 2, 2),
    "padding": "zeros",
    "flatten": True,
  }
  return cfg, {
    "seed": seed,
    "num_steps_per_env": 24,
    "max_iterations": 5000,
    "obs_groups": {"student": ("policy", "rgbd"), "teacher": ("teacher",)},
    "save_interval": 50,
    "experiment_name": "wuji_dodecahedron_l515_distillation",
    "run_name": "DodecahedronL515Distillation",
    "logger": "tensorboard",
    "wandb_project": "wuji_reorient_mjlab",
    "wandb_tags": (),
    "clip_actions": None,
    "upload_model": False,
    "student": student,
    "teacher": teacher,
    "algorithm": {
      "class_name": "Distillation",
      "num_learning_epochs": 2,
      "gradient_length": 8,
      "learning_rate": 1.0e-4,
      "max_grad_norm": 1.0,
      "loss_type": "huber",
      "optimizer": "adam",
      "auxiliary_target_group": "visual_privileged",
      "auxiliary_loss_coef": 1.0,
      "auxiliary_loss_type": "rotation_6d",
      "auxiliary_position_loss_coef": 10.0,
      "auxiliary_rotation_loss_coef": 1.0,
      "teacher_action_prob_start": 1.0,
      "teacher_action_prob_end": 0.0,
      "teacher_action_prob_warmup_updates": 500,
      "teacher_action_prob_decay_updates": 4000,
      "rnd_cfg": None,
      "symmetry_cfg": None,
    },
  }
