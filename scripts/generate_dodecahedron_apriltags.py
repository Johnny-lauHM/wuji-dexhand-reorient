"""Generate a metric AprilTag texture, board geometry and 1:1 SVG print sheet.

Uses OpenCV's AprilTag 36h11 dictionary, not decorative/AI-generated patterns.
The numbered baseline mesh and its physics are preserved in a separate variant.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ASSET = (
  Path(__file__).resolve().parents[1] / "src/wuji_mjlab/assets/objects/dodecahedron"
)


def face_geometry():
  lines = (ASSET / "dodecahedron.obj").read_text().splitlines()
  vertices = np.array(
    [[float(x) for x in s.split()[1:]] for s in lines if s.startswith("v ")]
  )
  triangles = np.array(
    [
      [int(x.split("/")[0]) - 1 for x in s.split()[1:]]
      for s in lines
      if s.startswith("f ")
    ]
  )
  groups = {}
  for index, (a, b, c) in enumerate(triangles):
    normal = np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])
    normal /= np.linalg.norm(normal)
    groups.setdefault(tuple(np.round(normal, 8)), []).append(index)
  if len(groups) != 12:
    raise ValueError("Expected twelve planar faces")
  return vertices, triangles, sorted(groups.items())


def generate(radius_m=0.035, tag_size_m=0.018):
  if not np.isfinite([radius_m, tag_size_m]).all() or min(radius_m, tag_size_m) <= 0:
    raise ValueError("Radius and tag size must be finite and positive")
  vertices, triangles, groups = face_geometry()
  dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
  # 8 black-border cells plus one white cell on each side. Exact integer cells.
  cell_px, tile_px = 36, 768
  black_px = 8 * cell_px
  meters_per_px = tag_size_m / black_px
  atlas = Image.new("RGB", (4 * tile_px, 3 * tile_px), "white")
  uv = []
  tri_uv = np.empty_like(triangles)
  faces = []
  svg = [
    '<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="297mm" viewBox="0 0 210 297">',
    '<rect width="210" height="297" fill="white"/>',
    '<text x="15" y="12" font-size="4">AprilTag 36h11 — print at 100%, no fit-to-page</text>',
    f'<text x="15" y="19" font-size="3">Black square: {tag_size_m * 1000:g} mm; object circumradius: {radius_m * 1000:g} mm</text>',
  ]
  for face_index, (key, tri_ids) in enumerate(groups):
    tag_id = face_index + 1  # ID 0 reserved for the existing wrist tag.
    ids = sorted(set(triangles[tri_ids].flatten()))
    points = vertices[ids] * radius_m
    center = points.mean(axis=0)
    normal = np.array(key) / np.linalg.norm(key)
    up = np.array([0.0, 0.0, 1.0])
    up -= np.dot(up, normal) * normal
    up /= np.linalg.norm(up)
    right = np.cross(up, normal)
    polygon = np.column_stack(((points - center) @ right, (points - center) @ up))
    polygon = polygon[np.argsort(np.arctan2(polygon[:, 1], polygon[:, 0]))]
    # White quiet zone must also fit entirely inside every pentagon edge.
    half_white = tag_size_m * 10 / 8 / 2
    for corner in (
      (-half_white, -half_white),
      (half_white, -half_white),
      (half_white, half_white),
      (-half_white, half_white),
    ):
      for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        edge, delta = b - a, np.asarray(corner) - a
        if edge[0] * delta[1] - edge[1] * delta[0] <= 0:
          raise ValueError(f"Tag plus quiet zone does not fit face {tag_id}")
    col, row = face_index % 4, face_index // 4
    lookup = {}
    for vertex_id in ids:
      delta = vertices[vertex_id] * radius_m - center
      px = col * tile_px + tile_px / 2 + delta @ right / meters_per_px
      py = row * tile_px + tile_px / 2 - delta @ up / meters_per_px
      if not (
        col * tile_px <= px <= (col + 1) * tile_px
        and row * tile_px <= py <= (row + 1) * tile_px
      ):
        raise ValueError("Face exceeds its texture tile; use a larger tag size")
      lookup[vertex_id] = len(uv)
      uv.append((px / (4 * tile_px), 1 - py / (3 * tile_px)))
    for index in tri_ids:
      tri_uv[index] = [lookup[x] for x in triangles[index]]
    bits = cv2.aruco.generateImageMarker(dictionary, tag_id, 8)
    image = Image.fromarray(bits).resize((black_px, black_px), Image.Resampling.NEAREST)
    atlas.paste(
      image,
      (
        col * tile_px + (tile_px - black_px) // 2,
        row * tile_px + (tile_px - black_px) // 2,
      ),
    )
    half = tag_size_m / 2
    # OpenCV detector canonical corner order: top-left, top-right, bottom-right, bottom-left.
    corners = [
      center + x * right + y * up
      for x, y in ((-half, half), (half, half), (half, -half), (-half, -half))
    ]
    faces.append(
      {
        "id": tag_id,
        "face_index": face_index + 1,
        "center_object_m": center.tolist(),
        "normal_object": normal.tolist(),
        "right_object": right.tolist(),
        "up_object": up.tolist(),
        "corners_object_m": np.asarray(corners).tolist(),
        "vertices_object_m": points.tolist(),
      }
    )
    # Print marker inside a dashed face outline, with external ID/top cue.
    x0, y0 = 35 + (face_index % 3) * 67, 48 + (face_index // 3) * 57
    outline = " ".join(f"{x0 + x * 1000:.6f},{y0 - y * 1000:.6f}" for x, y in polygon)
    svg.append(
      f'<polygon points="{outline}" fill="none" stroke="#aaa" stroke-width="0.15" stroke-dasharray="1,1"/>'
    )
    cell_mm = tag_size_m * 1000 / 8
    for y, x in np.argwhere(bits == 0):
      svg.append(
        f'<rect x="{x0 + (x - 4) * cell_mm:.6f}" y="{y0 + (y - 4) * cell_mm:.6f}" width="{cell_mm:.6f}" height="{cell_mm:.6f}"/>'
      )
    svg.append(
      f'<text x="{x0}" y="{y0 - 24}" text-anchor="middle" font-size="3">ID {tag_id} — TOP</text>'
    )
  svg.extend(
    [
      '<path d="M15 279h50m-50 -2v4m50 -4v4" stroke="black" stroke-width="0.2"/>',
      '<text x="15" y="286" font-size="3">Verify this line measures 50 mm before cutting.</text>',
      "</svg>",
    ]
  )
  atlas.save(ASSET / "apriltag36h11.png")
  lines = [
    "# AprilTag variant: same collision vertices and triangles; metric per-face UVs."
  ]
  lines += ["v " + " ".join(f"{x:.12g}" for x in p) for p in vertices]
  lines += ["vt " + " ".join(f"{x:.12g}" for x in p) for p in uv]
  lines += [
    "f " + " ".join(f"{v + 1}/{t + 1}" for v, t in zip(tri, tex, strict=True))
    for tri, tex in zip(triangles, tri_uv, strict=True)
  ]
  (ASSET / "dodecahedron_apriltag.obj").write_text("\n".join(lines) + "\n")
  tree = ET.parse(ASSET / "dodecahedron.xml")
  tree.find("asset/mesh").set("file", "dodecahedron_apriltag.obj")
  tree.find("asset/mesh").set("scale", " ".join([str(radius_m)] * 3))
  tree.find("asset/texture").set("file", "apriltag36h11.png")
  tree.find("asset/material").set("specular", "0")
  body = tree.find("worldbody/body")
  for site in list(body.findall("site")):
    if site.get("name", "").startswith("orientation_"):
      body.remove(site)
  ET.indent(tree)
  tree.write(ASSET / "dodecahedron_apriltag.xml", encoding="unicode")
  manifest = {
    "family": "tag36h11",
    "radius_m": radius_m,
    "tag_size_m": tag_size_m,
    "quiet_zone_size_m": tag_size_m * 10 / 8,
    "corner_order": "top-left,top-right,bottom-right,bottom-left (canonical decoded marker)",
    "object_frame": "source mesh/body frame, before MuJoCo mesh principal-axis compilation",
    "faces": faces,
  }
  (ASSET / "apriltag_board.json").write_text(json.dumps(manifest, indent=2) + "\n")
  (ASSET / "apriltag_print_A4.svg").write_text("\n".join(svg) + "\n")
  return manifest


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--radius-mm", type=float, default=35)
  parser.add_argument("--tag-size-mm", type=float, default=18)
  args = parser.parse_args()
  generate(args.radius_mm / 1000, args.tag_size_mm / 1000)
  print(f"Generated AprilTag assets in {ASSET}")
