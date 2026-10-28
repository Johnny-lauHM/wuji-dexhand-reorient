"""Render every physical face and decode its AprilTag; verify metric board PnP."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from wuji_mjlab.assets.objects.dodecahedron.object_cfg import (
  DODECAHEDRON_XML,
  apriltag_dodecahedron_spec,
  dodecahedron_spec,
)


def verify(output):
  output.mkdir(parents=True, exist_ok=True)
  board = json.loads(DODECAHEDRON_XML.with_name("apriltag_board.json").read_text())
  spec = apriltag_dodecahedron_spec()
  spec.worldbody.add_camera(name="verify", pos=(0, 0, 0.4), fovy=43)
  spec.worldbody.add_light(
    pos=(0, 0, 0), dir=(0, 0, -1), ambient=(0.7, 0.7, 0.7), diffuse=(0, 0, 0)
  )
  spec.visual.global_.offwidth = 1280
  spec.visual.global_.offheight = 720
  model = spec.compile()
  base = dodecahedron_spec(board["radius_m"]).compile()
  # The tags only change surface appearance, never contact geometry or inertia.
  for field in (
    "body_mass",
    "body_inertia",
    "mesh_vert",
    "mesh_face",
    "geom_contype",
    "geom_conaffinity",
  ):
    np.testing.assert_allclose(getattr(model, field), getattr(base, field), atol=1e-7)
  data = mujoco.MjData(model)
  detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
  )
  f = 720 / (2 * np.tan(np.deg2rad(43) / 2))
  K = np.array([[f, 0, 640], [0, f, 360], [0, 0, 1]])
  records, tiles = [], []
  with mujoco.Renderer(model, height=720, width=1280) as renderer:
    for face in board["faces"]:
      normal, right, up = (
        np.array(face[key]) for key in ("normal_object", "right_object", "up_object")
      )
      center = np.array(face["center_object_m"])
      position = center + normal * 0.38
      rotation = np.column_stack((right, up, normal))
      model.cam_pos[0] = position
      model.cam_quat[0] = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
      mujoco.mj_forward(model, data)
      renderer.update_scene(data, camera="verify")
      rgb = renderer.render()
      corners, ids, _ = detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
      found = [] if ids is None else ids.flatten().tolist()
      if face["id"] not in found:
        Image.fromarray(rgb).save(output / f"failed_{face['id']}.png")
        raise RuntimeError(f"Face {face['id']} is not decodable: found {found}")
      pixels = corners[found.index(face["id"])][0].astype(np.float64)
      object_points = np.array(face["corners_object_m"])
      # Independent analytic projection checks UV orientation AND metric size.
      R_cv = np.diag([1, -1, -1]) @ rotation.T
      xyz = (R_cv @ (object_points - position).T).T
      projected = (K @ xyz.T).T
      projected = projected[:, :2] / projected[:, 2:]
      error = float(np.linalg.norm(projected - pixels, axis=1).max())
      if error > 2.0:
        raise RuntimeError(f"Tag {face['id']} corner geometry mismatch: {error:.2f} px")
      ok, rvec, tvec = cv2.solvePnP(
        object_points, pixels, K, np.zeros(5), flags=cv2.SOLVEPNP_SQPNP
      )
      if not ok:
        raise RuntimeError("PnP failed")
      expected_translation = -R_cv @ position
      translation_error = float(np.linalg.norm(tvec.flatten() - expected_translation))
      rotation_error = float(
        Rotation.from_matrix(cv2.Rodrigues(rvec)[0] @ R_cv.T).magnitude()
      )
      if translation_error > 0.015 or rotation_error > 0.15:
        raise RuntimeError(
          f"PnP disagrees with render pose: {translation_error}, {rotation_error}"
        )
      records.append(
        {
          "id": face["id"],
          "max_corner_error_px": error,
          "translation_error_m": translation_error,
          "rotation_error_rad": rotation_error,
        }
      )
      Image.fromarray(rgb).save(output / f"face_{face['id']:02d}.png")
      tile = Image.fromarray(rgb).crop((500, 220, 780, 500))
      tiles.append(tile)
  sheet = Image.new("RGB", (4 * 280, 3 * 280), "white")
  for i, tile in enumerate(tiles):
    sheet.paste(tile, (i % 4 * 280, i // 4 * 280))
  sheet.save(output / "all_faces.png")
  (output / "verification.json").write_text(
    json.dumps({"physics_unchanged": True, "faces": records}, indent=2)
  )
  print(f"Verified all 12 rendered tag IDs, metric corners and PnP: {output}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, default=Path("logs/apriltag_verification"))
  verify(parser.parse_args().output)
