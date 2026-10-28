"""Fiducial board PnP. Inputs are pixels, intrinsics and known marker geometry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class BoardPose:
  rotation: np.ndarray  # object -> OpenCV optical camera (right/down/forward)
  translation: np.ndarray  # object origin in optical camera, meters
  ids: tuple[int, ...]
  reprojection_error_px: float


class AprilTagBoardEstimator:
  def __init__(self, board_path: str | Path, max_error_px: float = 3.0):
    import cv2

    self.cv2 = cv2
    self.board = json.loads(Path(board_path).read_text())
    dictionaries = {"tag36h11": cv2.aruco.DICT_APRILTAG_36h11, "aruco4x4_50": cv2.aruco.DICT_4X4_50}
    if self.board["family"] not in dictionaries:
      raise ValueError("Unsupported calibrated board family")
    self.faces = {f["id"]: f for f in self.board["faces"]}
    self.max_error_px = max_error_px
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    self.detector = cv2.aruco.ArucoDetector(
      cv2.aruco.getPredefinedDictionary(dictionaries[self.board["family"]]), params
    )

  def estimate(
    self, rgb: np.ndarray, K: np.ndarray, distortion=None
  ) -> BoardPose | None:
    """RGB uint8 image; IDs absent from the calibrated board are ignored."""
    cv2 = self.cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = self.detector.detectMarkers(gray)
    if ids is None:
      return None
    known = [
      (int(tag_id), corner[0])
      for tag_id, corner in zip(ids.flatten(), corners, strict=True)
      if int(tag_id) in self.faces
    ]
    if not known:
      return None
    object_points = np.concatenate(
      [self.faces[i]["corners_object_m"] for i, _ in known]
    ).astype(np.float64)
    image_points = np.concatenate([points for _, points in known]).astype(np.float64)
    dist = (
      np.zeros(5) if distortion is None else np.asarray(distortion, dtype=np.float64)
    )
    K = np.asarray(K, dtype=np.float64)
    try:
      if len(known) > 1:
        ok, rv, tv, inliers = cv2.solvePnPRansac(
          object_points,
          image_points,
          K,
          dist,
          iterationsCount=100,
          reprojectionError=self.max_error_px,
          flags=cv2.SOLVEPNP_SQPNP,
        )
        if not ok or inliers is None:
          return None
        # Keep complete markers, never silently assemble a pose from isolated
        # corners belonging to misidentified or partially inconsistent tags.
        inlier_set = set(inliers.flatten().tolist())
        keep = [
          j for j in range(len(known)) if all(4 * j + k in inlier_set for k in range(4))
        ]
        if not keep:
          return None
        object_points = np.concatenate([object_points[4 * j : 4 * j + 4] for j in keep])
        image_points = np.concatenate([image_points[4 * j : 4 * j + 4] for j in keep])
        known = [known[j] for j in keep]
      # IPPE enumerates the planar alternatives for a single visible face.
      flag = cv2.SOLVEPNP_IPPE if len(known) == 1 else cv2.SOLVEPNP_SQPNP
      result = cv2.solvePnPGeneric(object_points, image_points, K, dist, flags=flag)
      if not result[0]:
        return None
      candidates = []
      for rv, tv in zip(result[1], result[2], strict=True):
        rot = cv2.Rodrigues(rv)[0]
        translation = tv.flatten()
        transformed = (rot @ object_points.T).T + translation
        if not np.isfinite(transformed).all() or (transformed[:, 2] <= 0).any():
          continue
        # The printed face must point toward the camera, not into the object.
        if any(
          np.dot(
            rot @ np.array(self.faces[i]["normal_object"]),
            rot @ np.array(self.faces[i]["center_object_m"]) + translation,
          )
          >= 0
          for i, _ in known
        ):
          continue
        projected = cv2.projectPoints(object_points, rv, tv, K, dist)[0].reshape(-1, 2)
        errors = np.linalg.norm(projected - image_points, axis=1)
        if errors.max() <= self.max_error_px:
          candidates.append(
            BoardPose(
              rot, translation, tuple(i for i, _ in known), float(errors.mean())
            )
          )
      return (
        min(candidates, key=lambda p: p.reprojection_error_px) if candidates else None
      )
    except cv2.error:
      # A failed/degenerate PnP solve is a missing measurement, not a truth fallback.
      return None


class PoseCache:
  """Bounded hold-last behavior with explicit validity/age and episode resets."""

  def __init__(
    self,
    count: int,
    timeout_s: float,
    max_position_speed_m_s=0.5,
    max_rotation_speed_rad_s=12.0,
  ):
    if timeout_s <= 0:
      raise ValueError("timeout_s must be positive")
    self.timeout_s = timeout_s
    self.max_position_speed_m_s = max_position_speed_m_s
    self.max_rotation_speed_rad_s = max_rotation_speed_rad_s
    self.rejected_jumps = 0
    self.position = np.zeros((count, 3))
    self.rotation = np.tile(np.eye(3), (count, 1, 1))
    self.last_seen = np.full(count, -np.inf)

  def reset(self, ids):
    self.position[ids] = 0
    self.rotation[ids] = np.eye(3)
    self.last_seen[ids] = -np.inf

  def update(self, poses, now_s, camera_to_tag_rotation, camera_origin_tag):
    fresh = np.zeros(len(poses), dtype=bool)
    for i, pose in enumerate(poses):
      if pose is not None:
        position = camera_to_tag_rotation @ pose.translation + camera_origin_tag
        rotation = camera_to_tag_rotation @ pose.rotation
        elapsed = now_s - self.last_seen[i]
        if 0 <= elapsed < self.timeout_s:
          displacement = np.linalg.norm(position - self.position[i])
          delta_R = rotation @ self.rotation[i].T
          angle = np.arccos(np.clip((np.trace(delta_R) - 1) / 2, -1, 1))
          if (
            displacement > 0.005 + self.max_position_speed_m_s * elapsed
            or angle > 0.1 + self.max_rotation_speed_rad_s * elapsed
          ):
            self.rejected_jumps += 1
            continue
        self.position[i] = position
        self.rotation[i] = rotation
        self.last_seen[i] = now_s
        fresh[i] = True
    age = np.maximum(now_s - self.last_seen, 0)
    usable = age < self.timeout_s
    return fresh, usable, np.minimum(age / self.timeout_s, 1)
