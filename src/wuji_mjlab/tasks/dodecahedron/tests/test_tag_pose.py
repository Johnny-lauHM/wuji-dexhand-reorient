"""Real marker decoding plus missing-observation and privileged-state boundaries."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from wuji_mjlab.assets.objects.dodecahedron.object_cfg import DODECAHEDRON_XML
from wuji_mjlab.perception.apriltag import AprilTagBoardEstimator, BoardPose, PoseCache
from wuji_mjlab.tasks.dodecahedron.tag_pose import (
  AprilTagPoseObservation,
  make_tag_pose_configs,
)


def test_bounded_pose_hold_and_independent_reset():
  cache = PoseCache(2, timeout_s=0.25)
  pose = BoardPose(np.eye(3), np.array([0.0, 0.0, 0.4]), (1,), 0.0)
  fresh, usable, age = cache.update([pose, None], 0.0, np.eye(3), np.zeros(3))
  assert fresh.tolist() == [True, False]
  assert usable.tolist() == [True, False]
  np.testing.assert_allclose(age, [0.0, 1.0])
  fresh, usable, age = cache.update([None, pose], 0.10, np.eye(3), np.zeros(3))
  assert fresh.tolist() == [False, True]
  assert usable.tolist() == [True, True]
  np.testing.assert_allclose(age, [0.4, 0.0])
  cache.reset([1])
  _, usable, age = cache.update([None, None], 0.25, np.eye(3), np.zeros(3))
  assert not usable.any()
  np.testing.assert_allclose(age, [1.0, 1.0])
  assert cache.last_seen[0] == 0.0
  assert np.isneginf(cache.last_seen[1])


def test_blank_and_unknown_id_have_no_pose():
  cv2 = pytest.importorskip("cv2")
  tracker = AprilTagBoardEstimator(DODECAHEDRON_XML.with_name("apriltag_board.json"))
  K = np.array([[500.0, 0, 200], [0, 500, 200], [0, 0, 1]])
  blank = np.full((400, 400, 3), 255, dtype=np.uint8)
  assert tracker.estimate(blank, K) is None
  dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
  # Wrist ID 0 is visible, but cannot be mistaken for an object face.
  blank[80:320, 80:320] = cv2.aruco.generateImageMarker(dictionary, 0, 240)[..., None]
  assert tracker.estimate(blank, K) is None


def test_pose_jump_is_rejected_using_only_visual_history():
  cache = PoseCache(1, timeout_s=0.25)
  good = BoardPose(np.eye(3), np.array([0.0, 0.0, 0.4]), (1,), 0.0)
  bad = BoardPose(np.diag([-1.0, -1.0, 1.0]), np.array([0.1, 0.0, 0.4]), (1,), 0.0)
  cache.update([good], 0, np.eye(3), np.zeros(3))
  fresh, usable, _ = cache.update([bad], 0.05, np.eye(3), np.zeros(3))
  assert not fresh[0] and usable[0]
  assert cache.rejected_jumps == 1
  np.testing.assert_allclose(cache.position[0], good.translation)
  # After a long loss, reacquisition has no stale pose gate to trap the tracker.
  fresh, _, _ = cache.update([bad], 0.3, np.eye(3), np.zeros(3))
  assert fresh[0]


def test_real_marker_pixels_recover_known_metric_pose():
  cv2 = pytest.importorskip("cv2")
  tracker = AprilTagBoardEstimator(DODECAHEDRON_XML.with_name("apriltag_board.json"))
  face = tracker.faces[1]
  # Front-facing face 1: tag occupies 240 px at focal 500 px.
  right, up, normal = (
    np.array(face[k]) for k in ("right_object", "up_object", "normal_object")
  )
  rot = np.vstack((right, -up, -normal))
  distance = 0.018 * 500 / 240
  expected_t = np.array([0.0, 0.0, distance]) - rot @ np.array(face["center_object_m"])
  rgb = np.full((400, 400, 3), 255, dtype=np.uint8)
  dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
  rgb[80:320, 80:320] = cv2.aruco.generateImageMarker(dictionary, 1, 240)[..., None]
  pose = tracker.estimate(
    rgb, np.array([[500.0, 0, 199.5], [0, 500, 199.5], [0, 0, 1]])
  )
  assert pose is not None
  assert pose.ids == (1,)
  np.testing.assert_allclose(pose.translation, expected_t, atol=0.001)
  np.testing.assert_allclose(pose.rotation, rot, atol=0.03)


def test_observation_can_run_without_object_or_robot_truth():
  pytest.importorskip("cv2")
  cfg, agent = make_tag_pose_configs(num_envs=1)
  assert agent.obs_groups["actor"] == ("policy",)
  assert set(cfg.observations) == {"policy", "critic"}
  assert set(cfg.observations["policy"].terms) == {
    "noisy_joint_angles",
    "qpos_error",
    "action_history",
    "apriltag_pose",
  }
  term = cfg.observations["policy"].terms["apriltag_pose"]
  # Only a camera and externally supplied goal exist; accessing object/robot
  # simulator state would raise KeyError.
  env = SimpleNamespace(
    cfg=cfg,
    num_envs=1,
    device="cpu",
    common_step_counter=0,
    step_dt=0.05,
    scene={
      "l515": SimpleNamespace(
        data=SimpleNamespace(rgb=torch.zeros((1, 32, 32, 3), dtype=torch.uint8))
      )
    },
    command_manager=SimpleNamespace(
      get_term=lambda _: SimpleNamespace(goal_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    ),
  )
  observation = AprilTagPoseObservation(term, env)
  pose = BoardPose(np.eye(3), np.array([0.0, 0.0, 0.4]), (1,), 0.0)
  observation.estimator = SimpleNamespace(estimate_batch=lambda *_: [pose])
  initial = observation(env, **term.params).clone()
  assert initial[0, 9] == 1 and initial[0, 10] == 0
  observation.estimator = SimpleNamespace(estimate_batch=lambda *_: [None])
  # Repeated read in one control step cannot age/re-sample the observation.
  torch.testing.assert_close(observation(env, **term.params), initial)
  env.common_step_counter = 1
  missing = observation(env, **term.params)
  torch.testing.assert_close(missing[:, :9], initial[:, :9])
  assert missing[0, 9] == 0 and missing[0, 10] == pytest.approx(0.2)
  env.common_step_counter = 5
  expired = observation(env, **term.params)
  assert (expired[0, :10] == 0).all() and expired[0, 10] == 1
  observation.reset(torch.tensor([0]))
  assert np.isneginf(observation.cache.last_seen[0])
