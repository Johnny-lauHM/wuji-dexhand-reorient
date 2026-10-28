"""Observation source boundaries and physically meaningful camera preprocessing."""

from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from wuji_mjlab.tasks.dodecahedron.env_cfg import wuji_hand_dodecahedron_env_cfg
from wuji_mjlab.tasks.dodecahedron.l515 import (
  camera_quaternion,
  encode_rgbd,
  goal_in_camera,
  make_l515_configs,
)
from wuji_mjlab.tasks.dodecahedron.tag_pose import (
  teacher_observation_from_apriltag_policy,
)


def test_apriltag_history_reconstructs_teacher_order():
  policy = torch.arange(213, dtype=torch.float32).unsqueeze(0)
  pose_history = policy[:, 180:].reshape(1, 3, 11)
  expected = torch.cat(
    (
      policy[:, :120],
      pose_history[:, :, :3].reshape(1, 9),
      pose_history[:, :, 3:9].reshape(1, 18),
      policy[:, 120:180],
    ),
    dim=-1,
  )
  actual = teacher_observation_from_apriltag_policy(policy)
  assert actual.shape == (1, 207)
  torch.testing.assert_close(actual, expected)


def test_camera_axes_point_at_workspace():
  position, target = np.array([0.06, -0.21, 0.82]), np.array([-0.14, 0.01, 0.56])
  quat = np.array(camera_quaternion(position, target))
  rot = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
  local = rot.T @ (target - position)
  np.testing.assert_allclose(local[:2], 0, atol=1e-10)
  assert local[2] < -0.25


def test_invalid_depth_mask_and_no_mutation():
  depth = torch.tensor([[[[float("nan")], [0.2], [0.5], [2.0], [float("inf")]]]])
  rgb = torch.full((1, 1, 5, 3), 255, dtype=torch.uint8)
  encoded = encode_rgbd(rgb, depth)
  assert encoded.shape == (1, 5, 1, 5)
  assert torch.isfinite(encoded).all()
  torch.testing.assert_close(encoded[0, 4, 0], torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]))
  assert encoded[0, 3, 0, 2] == 0.5
  assert torch.isnan(depth[0, 0, 0, 0])
  assert (rgb == 255).all()
  assert (encode_rgbd(rgb, depth, missing_prob=1)[:, 3:] == 0).all()


def test_goal_needs_command_but_no_object_state():
  # A scene/object is deliberately absent. This is a commanded target, not an
  # object-relative rotation error smuggled into an otherwise visual actor.
  env = SimpleNamespace(
    command_manager=SimpleNamespace(
      get_term=lambda _: SimpleNamespace(goal_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    )
  )
  torch.testing.assert_close(
    goal_in_camera(env, (1.0, 0.0, 0.0, 0.0)),
    torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]),
  )


def test_visual_actor_has_no_object_truth_and_baseline_is_unchanged():
  cfg, agent = make_l515_configs(num_envs=4)
  assert agent.obs_groups["actor"] == ("policy", "rgbd")
  assert set(cfg.observations["policy"].terms) == {
    "noisy_joint_angles",
    "qpos_error",
    "action_history",
    "goal_in_camera",
  }
  assert cfg.scene.env_spacing == 0
  assert cfg.scene.sensors[-1].data_types == ("rgb", "depth")
  baseline = wuji_hand_dodecahedron_env_cfg(num_envs=4)
  assert "cube_pos_in_tag" in baseline.observations["policy"].terms
  assert not baseline.observations["policy"].enable_corruption
  assert all(sensor.name != "l515" for sensor in baseline.scene.sensors)


def test_sim2real_domain_randomization_is_opt_in_and_moderate():
  nominal = wuji_hand_dodecahedron_env_cfg(num_envs=4)
  randomized = wuji_hand_dodecahedron_env_cfg(
    num_envs=4, sim2real_domain_randomization=True
  )
  assert all(term.mode == "reset" for term in nominal.events.values())
  assert "object_mass" in randomized.events
  assert "encoder_bias" in randomized.events
  assert "object_disturbance_force" not in randomized.events
  assert "object_size" not in randomized.events
  assert randomized.events["object_mass"].params["scale_range"] == (0.8, 1.2)
  assert "robot_link_inertia" not in randomized.events
  assert randomized.events["robot_link_mass"].func is randomized.events["object_mass"].func
  assert randomized.events["robot_link_mass"].params["scale_range"] == (0.9, 1.1)
  assert randomized.events["object_com"].params["ranges"] == (-0.0015, 0.0015)
  assert randomized.events["encoder_bias"].params["bias_range"] == (-0.005, 0.005)
  assert randomized.observations["policy"].enable_corruption
  assert not randomized.observations["critic"].enable_corruption
  assert "dr_params" not in randomized.observations["critic"].terms


def test_wrist_pitch_preserves_object_pose_in_palm_frame():
  palm_up = wuji_hand_dodecahedron_env_cfg(num_envs=1, wrist_pitch_deg=-90.0)
  palm_down = wuji_hand_dodecahedron_env_cfg(num_envs=1, wrist_pitch_deg=90.0)
  np.testing.assert_allclose(
    palm_up.scene.entities["robot"].init_state.rot,
    (np.sqrt(0.5), 0.0, -np.sqrt(0.5), 0.0),
  )
  np.testing.assert_allclose(
    palm_down.scene.entities["robot"].init_state.rot,
    (np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0),
  )
  np.testing.assert_allclose(
    palm_up.scene.entities["object"].init_state.pos, (-0.14, 0.01, 0.56)
  )
  np.testing.assert_allclose(
    palm_down.scene.entities["object"].init_state.pos, (0.14, 0.01, 0.44)
  )
