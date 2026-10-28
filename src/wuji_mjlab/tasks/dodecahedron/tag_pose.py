"""Observed AprilTag pose policy: never reads the object's simulator state."""

from __future__ import annotations

import time

import numpy as np
import torch
from mjlab.managers.observation_manager import ObservationTermCfg
from scipy.spatial.transform import Rotation

from wuji_mjlab.assets.objects.dodecahedron.object_cfg import DODECAHEDRON_XML
from wuji_mjlab.perception.apriltag import PoseCache
from wuji_mjlab.perception.batch_apriltag import BatchAprilTagEstimator
from wuji_mjlab.tasks.reorient.reorient_constants import (
  REORIENT_ROBOT_ROOT_POS,
  REORIENT_ROBOT_ROOT_ROT,
  TAG_IN_PALM_POS,
  TAG_IN_PALM_QUAT_WXYZ,
)

from .env_cfg import wuji_hand_dodecahedron_ppo_cfg
from .l515 import make_l515_configs


def quat_matrix(wxyz):
  q = np.asarray(wxyz)
  return Rotation.from_quat(q[..., [1, 2, 3, 0]]).as_matrix()


def teacher_observation_from_apriltag_policy(policy: torch.Tensor) -> torch.Tensor:
  """Rebuild the state Teacher input from proprioception and tag-pose history."""
  if policy.ndim != 2 or policy.shape[1] != 213:
    raise ValueError(f"Expected a (batch, 213) tag-pose policy, got {policy.shape}")
  pose_history = policy[:, 180:].reshape(policy.shape[0], 3, 11)
  positions = pose_history[:, :, :3].reshape(policy.shape[0], 9)
  orientations = pose_history[:, :, 3:9].reshape(policy.shape[0], 18)
  return torch.cat(
    (policy[:, :120], positions, orientations, policy[:, 120:180]), dim=-1
  )


class AprilTagPoseObservation:
  def __init__(self, cfg, env):
    p = cfg.params
    workers = min(p.get("detector_workers", 1), env.num_envs)
    self.estimator = BatchAprilTagEstimator(p["board_path"], workers)
    self.cache = PoseCache(
      env.num_envs,
      p["timeout_s"],
      p["max_position_speed_m_s"],
      p["max_rotation_speed_rad_s"],
    )
    camera = env.cfg.scene.sensors[-1]
    focal = camera.height / (2 * np.tan(np.deg2rad(camera.fovy) / 2))
    self.K = np.array(
      [[focal, 0, camera.width / 2], [0, focal, camera.height / 2], [0, 0, 1]]
    )
    # Fixed-mount calibration from configuration, never measured object/root pose.
    palm_R = quat_matrix(REORIENT_ROBOT_ROOT_ROT)
    self.tag_R_world = (palm_R @ quat_matrix(TAG_IN_PALM_QUAT_WXYZ)).T
    tag_position = np.asarray(REORIENT_ROBOT_ROOT_POS) + palm_R @ np.asarray(
      TAG_IN_PALM_POS
    )
    self.camera_R_tag = (
      self.tag_R_world @ quat_matrix(camera.quat) @ np.diag([1.0, -1.0, -1.0])
    )
    self.camera_position_tag = self.tag_R_world @ (
      np.asarray(camera.pos) - tag_position
    )
    self.last_step = None
    self.output = torch.zeros((env.num_envs, 11), device=env.device)
    self.last_poses = [None] * env.num_envs
    rng = np.random.default_rng((getattr(env.cfg, "seed", 0) or 0) + 515)
    self.rng = rng
    position_bias_range = p.get("position_bias_range_m", 0.0)
    rotation_bias_range = p.get("rotation_bias_range_rad", 0.0)
    self.position_noise_std = p.get("position_noise_std_m", 0.0)
    self.rotation_noise_std = p.get("rotation_noise_std_rad", 0.0)
    self.extra_dropout_prob = p.get("extra_dropout_prob", 0.0)
    self.position_bias = rng.uniform(
      -position_bias_range, position_bias_range, (env.num_envs, 3)
    )
    bias_axes = rng.normal(size=(env.num_envs, 3))
    bias_axes /= np.linalg.norm(bias_axes, axis=1, keepdims=True).clip(1.0e-9)
    bias_angles = rng.uniform(-rotation_bias_range, rotation_bias_range, env.num_envs)
    self.rotation_bias = Rotation.from_rotvec(
      bias_axes * bias_angles[:, None]
    ).as_matrix()
    self.latency_steps = int(p.get("latency_steps", 0))
    self.delay_history = np.zeros(
      (env.num_envs, self.latency_steps + 1, 11), dtype=np.float64
    )
    self.delay_initialized = np.zeros(env.num_envs, dtype=bool)
    self.frames = self.detected = 0
    self.detector_seconds = 0.0
    self.transfer_seconds = 0.0
    self.detection_seconds = 0.0

  def close(self):
    self.estimator.close()

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    if isinstance(ids, torch.Tensor):
      ids = ids.cpu().numpy()
    self.cache.reset(ids)
    self.delay_history[ids] = 0
    self.delay_initialized[ids] = False
    self.last_step = None

  def __call__(
    self,
    env,
    board_path,
    timeout_s,
    max_position_speed_m_s,
    max_rotation_speed_rad_s,
    detector_workers=1,
    position_bias_range_m=0.0,
    rotation_bias_range_rad=0.0,
    position_noise_std_m=0.0,
    rotation_noise_std_rad=0.0,
    extra_dropout_prob=0.0,
    latency_steps=0,
  ):
    del (
      board_path,
      timeout_s,
      max_position_speed_m_s,
      max_rotation_speed_rad_s,
      detector_workers,
      position_bias_range_m,
      rotation_bias_range_rad,
      position_noise_std_m,
      rotation_noise_std_rad,
      extra_dropout_prob,
      latency_steps,
    )
    # Shape discovery happens before the first reset/sensor update.
    if not hasattr(env, "common_step_counter"):
      return self.output
    step = env.common_step_counter
    if self.last_step == step:
      return self.output
    start = time.perf_counter()
    rgb = env.scene["l515"].data.rgb.cpu().numpy()
    detection_start = time.perf_counter()
    self.transfer_seconds += detection_start - start
    poses = self.estimator.estimate_batch(rgb, self.K)
    if self.extra_dropout_prob:
      poses = [
        None if self.rng.random() < self.extra_dropout_prob else pose for pose in poses
      ]
    self.detection_seconds += time.perf_counter() - detection_start
    self.detector_seconds += time.perf_counter() - start
    self.frames += len(poses)
    fresh, usable, age = self.cache.update(
      poses, step * env.step_dt, self.camera_R_tag, self.camera_position_tag
    )
    self.last_poses = [
      pose if good else None for pose, good in zip(poses, fresh, strict=True)
    ]
    self.detected += int(fresh.sum())
    # Goal is an externally available command. Error uses the estimated rotation.
    goal_world = (
      env.command_manager.get_term("reorient_command").goal_quat.cpu().numpy()
    )
    goal_tag = self.tag_R_world @ quat_matrix(goal_world)
    observed_position = self.cache.position + self.position_bias
    if self.position_noise_std:
      observed_position += self.rng.normal(
        0.0, self.position_noise_std, observed_position.shape
      )
    observed_rotation = self.rotation_bias @ self.cache.rotation
    if self.rotation_noise_std:
      axes = self.rng.normal(size=(env.num_envs, 3))
      axes /= np.linalg.norm(axes, axis=1, keepdims=True).clip(1.0e-9)
      angles = self.rng.normal(0.0, self.rotation_noise_std, env.num_envs)
      observed_rotation = (
        Rotation.from_rotvec(axes * angles[:, None]).as_matrix() @ observed_rotation
      )
    error = observed_rotation @ np.swapaxes(goal_tag, -1, -2)
    pose_terms = np.concatenate(
      (observed_position, error.reshape(-1, 9)[:, 3:]), axis=1
    )
    pose_terms[~usable] = 0
    result = np.column_stack((pose_terms, fresh.astype(float), age))
    self.delay_history[:, :-1] = self.delay_history[:, 1:]
    self.delay_history[:, -1] = result
    new_ids = ~self.delay_initialized
    self.delay_history[new_ids] = result[new_ids, None, :]
    self.delay_initialized[new_ids] = True
    delayed = self.delay_history[:, 0]
    self.output = torch.as_tensor(delayed, dtype=torch.float32, device=env.device)
    self.last_step = step
    return self.output


def make_tag_pose_configs(
  num_envs=4,
  width=1280,
  height=720,
  seed=42,
  detector_workers=8,
  sim2real_domain_randomization=False,
):
  cfg, _ = make_l515_configs(
    num_envs,
    width,
    height,
    seed,
    sim2real_domain_randomization=sim2real_domain_randomization,
  )
  camera = cfg.scene.sensors[-1]
  camera.data_types = ("rgb",)
  camera.fovy = 43.0
  cfg.observations.pop("rgbd")
  policy = cfg.observations["policy"]
  policy.terms.pop("goal_in_camera")
  policy.terms["apriltag_pose"] = ObservationTermCfg(
    func=AprilTagPoseObservation,
    params={
      "board_path": str(DODECAHEDRON_XML.with_name("apriltag_board.json")),
      "timeout_s": 0.25,
      "max_position_speed_m_s": 0.5,
      "max_rotation_speed_rad_s": 12.0,
      "detector_workers": detector_workers,
      "position_bias_range_m": 0.002 if sim2real_domain_randomization else 0.0,
      "rotation_bias_range_rad": (
        float(np.deg2rad(2.0)) if sim2real_domain_randomization else 0.0
      ),
      "position_noise_std_m": 0.0005 if sim2real_domain_randomization else 0.0,
      "rotation_noise_std_rad": (
        float(np.deg2rad(0.3)) if sim2real_domain_randomization else 0.0
      ),
      "extra_dropout_prob": 0.02 if sim2real_domain_randomization else 0.0,
      "latency_steps": 1 if sim2real_domain_randomization else 0,
    },
    history_length=3,
  )
  agent = wuji_hand_dodecahedron_ppo_cfg()
  agent.seed = seed
  agent.experiment_name = "wuji_dodecahedron_apriltag_pose"
  agent.algorithm.num_mini_batches = min(8, num_envs)
  return cfg, agent
