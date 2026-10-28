"""A solid regular dodecahedron with a single convex collision mesh."""

from functools import partial
from math import isfinite
from pathlib import Path

import mujoco
from mjlab.entity import EntityCfg

DODECAHEDRON_XML = Path(__file__).with_name("dodecahedron.xml")
DEFAULT_RADIUS_M = 0.035
DEFAULT_MASS_KG = 0.120


def dodecahedron_spec(
  radius_m: float = DEFAULT_RADIUS_M, mass_kg: float = DEFAULT_MASS_KG
) -> mujoco.MjSpec:
  """Build at a fixed circumradius and mass; MuJoCo computes solid inertia.

  ``cube``/``cube_mesh`` are compatibility names for shared reorient terms,
  not box geometry. Scaling happens before compilation, never via geom_size.
  """
  if not isfinite(radius_m) or radius_m <= 0:
    raise ValueError("radius_m must be finite and positive")
  if not isfinite(mass_kg) or mass_kg <= 0:
    raise ValueError("mass_kg must be finite and positive")
  spec = mujoco.MjSpec.from_file(str(DODECAHEDRON_XML))
  spec.mesh("cube_mesh").scale = [radius_m] * 3
  spec.geom("cube").mass = mass_kg
  # Visual-only body axes scale with the mesh. They have no mass or contacts.
  for site in spec.sites:
    if site.name.startswith("orientation_"):
      site.fromto = site.fromto * (radius_m / DEFAULT_RADIUS_M)
  return spec


def get_dodecahedron_cfg(
  radius_m: float = DEFAULT_RADIUS_M, mass_kg: float = DEFAULT_MASS_KG
) -> EntityCfg:
  return EntityCfg(spec_fn=partial(dodecahedron_spec, radius_m, mass_kg))


def apriltag_dodecahedron_spec() -> mujoco.MjSpec:
  """Metric tag variant generated with its matching board/print manifest."""
  return mujoco.MjSpec.from_file(
    str(DODECAHEDRON_XML.with_name("dodecahedron_apriltag.xml"))
  )
