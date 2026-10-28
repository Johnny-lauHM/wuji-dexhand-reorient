"""Compare the sensor adapter with the established observation contract."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from mjlab.envs import ManagerBasedRlEnv
from wuji_mjlab.tasks.reorient.config.wuji_hand.env_cfgs import (
  wuji_hand_cube_downward_env_cfg,
)
from wuji_mjlab.perception.cube_wrist import SensorHistory, calibration
from wuji_mjlab.tasks.reorient.reorient_constants import (
  TAG_IN_PALM_POS,
  TAG_IN_PALM_QUAT_WXYZ,
)


def test_sensor_adapter_matches_teacher_observation_history():
  """Truth is a test oracle only; the adapter API accepts no environment."""
  cfg = wuji_hand_cube_downward_env_cfg(
    num_envs=1, cube_edge_m=0.045, wrist_assembly=True
  )
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    obs, _ = env.reset()
    history = SensorHistory()
    palm_R = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    tag_R = Rotation.from_quat(
      np.array(TAG_IN_PALM_QUAT_WXYZ)[[1, 2, 3, 0]]
    ).as_matrix()
    world_tag = palm_R @ tag_R
    tag_origin = np.array(
      cfg.scene.entities["robot"].init_state.pos
    ) + palm_R @ np.array(TAG_IN_PALM_POS)
    for step in range(3):
      robot = env.scene["robot"]
      obj = env.scene["object"]
      position = world_tag.T @ (obj.data.root_link_pos_w[0].cpu().numpy() - tag_origin)
      quat = obj.data.root_link_quat_w[0].cpu().numpy()
      rotation = world_tag.T @ Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
      goal = env.command_manager.get_term("reorient_command").goal_quat[0].cpu().numpy()
      goal = world_tag.T @ Rotation.from_quat(goal[[1, 2, 3, 0]]).as_matrix()
      x = history.update(
        robot.data.joint_pos[0].cpu().numpy(),
        robot.data.soft_joint_pos_limits[0].cpu().numpy(),
        env.action_manager.get_term("joint_pos").processed_action[0].cpu().numpy(),
        env.action_manager.prev_action[0].cpu().numpy(),
        position,
        rotation,
        goal,
      )
      np.testing.assert_allclose(x, obs["policy"][0, :207].cpu().numpy(), atol=2e-5)
      if step < 2:
        obs, _, _, _, _ = env.step(
          torch.full((1, 20), 0.01 * (step + 1), device=env.device)
        )
  finally:
    env.close()


def test_factory_camera_calibration_is_finite_and_right_handed():
  R, p, tag_R, K = calibration()
  np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-7)
  assert np.isclose(np.linalg.det(R), 1.0)
  assert np.isfinite(p).all() and np.isfinite(K).all()
  assert K[0, 0] > 0 and K[1, 1] > 0
