"""Sensor-only cube observations and fixed wrist-camera calibration."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from wuji_mjlab.assets.robots.wuji_hand.wuji_hand_cfg import get_wuji_hand_cfg
from wuji_mjlab.tasks.reorient.reorient_constants import (
  TAG_IN_PALM_POS,
  TAG_IN_PALM_QUAT_WXYZ,
)


def calibration():
  # Factory/configured rigid geometry only. No object or rollout state is used.
  m = get_wuji_hand_cfg(wrist_assembly=True).spec_fn().compile()
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  c = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "d405_b")
  P = d.cam_xpos[c].copy()
  C = d.cam_xmat[c].reshape(3, 3) @ np.diag([1, -1, -1])
  T = Rotation.from_quat(np.array(TAG_IN_PALM_QUAT_WXYZ)[[1, 2, 3, 0]]).as_matrix()
  R = T.T @ C
  p = T.T @ (P - np.array(TAG_IN_PALM_POS))
  intr = m.cam_intrinsic[c]
  ss = m.cam_sensorsize[c]
  K = np.array(
    [
      [intr[0] / ss[0] * 640, 0, 319.5 - intr[2] / ss[0] * 640],
      [0, intr[1] / ss[1] * 480, 239.5 - intr[3] / ss[1] * 480],
      [0, 0, 1.0],
    ]
  )
  return R, p, T, K


class SensorHistory:
  """Only measured joint angles, known targets/actions, visual pose, and supplied goal."""

  def __init__(self):
    self.history = None

  def update(self, q, limits, target, last_action, position, rotation, goal):
    center = limits.mean(axis=1)
    half = (limits[:, 1] - limits[:, 0]) / 2
    angles = np.clip((q - center) / (half + 1e-6), -1, 1)
    error = angles - np.clip((target - center) / (half + 1e-6), -1, 1)
    ori = (rotation @ goal.T).reshape(9)[3:]
    terms = [angles, error, position, ori, last_action]
    if self.history is None:
      self.history = [np.tile(t, (3, 1)) for t in terms]
    else:
      for h, t in zip(self.history, terms):
        h[:-1] = h[1:]
        h[-1] = t
    return np.concatenate([h.ravel() for h in self.history]).astype(np.float32)
