"""Bake face IDs 1–12 and UVs without changing collision vertices or triangles."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "src/wuji_mjlab/assets/objects/dodecahedron"


def main():
  path = ASSET / "dodecahedron.xml"
  tree = ET.parse(
    path, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
  )
  root = tree.getroot()
  asset = root.find("asset")
  mesh = asset.find("mesh")
  if mesh.get("file"):
    lines = (ASSET / mesh.get("file")).read_text().splitlines()
    vertices = np.array(
      [[float(x) for x in line.split()[1:]] for line in lines if line.startswith("v ")]
    )
    triangles = np.array(
      [
        [int(x.split("/")[0]) - 1 for x in line.split()[1:]]
        for line in lines
        if line.startswith("f ")
      ]
    )
  else:
    vertices = np.fromstring(mesh.get("vertex"), sep=" ").reshape(-1, 3)
    triangles = np.fromstring(mesh.get("face"), sep=" ", dtype=int).reshape(-1, 3)
  groups = {}
  for tri_idx, (a, b, c) in enumerate(triangles):
    normal = np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])
    normal /= np.linalg.norm(normal)
    key = tuple(np.round(normal, 8))
    groups.setdefault(key, []).append(tri_idx)
  assert len(groups) == 12
  size = 384
  atlas = Image.new("RGB", (4 * size, 3 * size))
  draw = ImageDraw.Draw(atlas)
  font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 170)
  palette = [
    "#F6C5B5",
    "#B6DEFA",
    "#C5E6B1",
    "#F8D88A",
    "#D8C5F0",
    "#A9E2D7",
    "#F0BDD9",
    "#C4D3F5",
    "#E0E5A0",
    "#F5C69C",
    "#B9E8EA",
    "#E7C8B4",
  ]
  texcoords = []
  facetexcoords = np.empty_like(triangles)
  for face_idx, (key, triangle_ids) in enumerate(sorted(groups.items())):
    ids = sorted(set(triangles[triangle_ids].flatten()))
    assert len(ids) == 5
    points = vertices[ids]
    center = points.mean(axis=0)
    normal = np.array(key)
    normal /= np.linalg.norm(normal)
    up = np.array([0.0, 0.0, 1.0])
    up -= np.dot(up, normal) * normal
    up /= np.linalg.norm(up)
    right = np.cross(up, normal)
    radius = np.linalg.norm(points - center, axis=1).max()
    col, row = face_idx % 4, face_idx // 4
    lookup = {}
    for vertex_id in ids:
      delta = (vertices[vertex_id] - center) / radius
      u = (col + 0.5 + 0.46 * np.dot(delta, right)) / 4
      v = 1 - (row + 0.5 - 0.46 * np.dot(delta, up)) / 3
      lookup[vertex_id] = len(texcoords)
      texcoords.append((u, v))
    for idx in triangle_ids:
      facetexcoords[idx] = [lookup[x] for x in triangles[idx]]
    x, y = col * size, row * size
    draw.rectangle((x, y, x + size, y + size), fill=palette[face_idx])
    # A central light disk and dark type remain legible under scene lighting.
    draw.ellipse(
      (x + 69, y + 69, x + 315, y + 315), fill="#FFF9EF", outline="#243247", width=6
    )
    draw.text(
      (x + size / 2, y + size / 2 - 5),
      str(face_idx + 1),
      font=font,
      fill="#172439",
      anchor="mm",
    )
    draw.line((x + 151, y + 287, x + 233, y + 287), fill="#172439", width=7)
  atlas.save(ASSET / "face_labels.png")
  lines = ["# Regular dodecahedron; independent per-face UVs for labels 1–12."]
  lines += ["v " + " ".join(f"{x:.12g}" for x in point) for point in vertices]
  lines += ["vt " + " ".join(f"{x:.9g}" for x in uv) for uv in texcoords]
  lines += [
    "f " + " ".join(f"{v + 1}/{t + 1}" for v, t in zip(tri, uv, strict=True))
    for tri, uv in zip(triangles, facetexcoords, strict=True)
  ]
  (ASSET / "dodecahedron.obj").write_text("\n".join(lines) + "\n")
  for attr in ("vertex", "face", "texcoord", "facetexcoord"):
    mesh.attrib.pop(attr, None)
  mesh.set("file", "dodecahedron.obj")
  if asset.find("texture[@name='face_labels']") is None:
    ET.SubElement(
      asset, "texture", name="face_labels", type="2d", file="face_labels.png"
    )
    ET.SubElement(
      asset,
      "material",
      name="face_labels",
      texture="face_labels",
      specular="0.15",
      shininess="0.1",
    )
  geom = root.find("worldbody/body/geom[@name='cube']")
  geom.set("material", "face_labels")
  geom.set("rgba", "1 1 1 1")
  ET.indent(tree)
  tree.write(path, encoding="unicode")
  with path.open("a") as f:
    f.write("\n")


if __name__ == "__main__":
  main()
