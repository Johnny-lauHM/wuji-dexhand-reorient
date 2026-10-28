"""Asset physics and task isolation checks, using real MuJoCo compilation."""

import mujoco
import numpy as np
import pytest
from scipy.spatial import ConvexHull
from wuji_mjlab.assets.objects.dodecahedron.object_cfg import dodecahedron_spec
from wuji_mjlab.tasks.dodecahedron.env_cfg import wuji_hand_dodecahedron_env_cfg
from wuji_mjlab.tasks.reorient.config.wuji_hand.env_cfgs import (
  wuji_hand_reorient_env_cfg,
)


def test_regular_solid_and_collision_geometry():
  model = dodecahedron_spec().compile()
  assert model.ngeom == 1
  assert model.geom_type[0] == mujoco.mjtGeom.mjGEOM_MESH
  assert model.geom_contype[0] != 0
  vertices = model.mesh_vert
  assert len(vertices) == 20
  np.testing.assert_allclose(np.linalg.norm(vertices, axis=1), 0.035, rtol=1e-6)
  hull = ConvexHull(vertices)
  planes = np.unique(np.round(hull.equations, 5), axis=0)
  assert len(planes) == 12
  # Each supporting plane contains five vertices: actual pentagonal faces.
  for plane in planes:
    distances = vertices @ plane[:3] + plane[3]
    assert np.count_nonzero(np.abs(distances) < 1e-5) == 5
  np.testing.assert_allclose(model.body_mass[1], 0.120)
  assert np.all(model.body_inertia[1] > 0)
  np.testing.assert_allclose(model.body_inertia[1], model.body_inertia[1, 0])
  # Closed outward triangle winding and a nonzero solid volume.
  triangles = vertices[model.mesh_face]
  signed_volume = (
    np.einsum(
      "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
    ).sum()
    / 6
  )
  assert signed_volume == pytest.approx(hull.volume, rel=1e-5)


def test_physical_scaling_before_compile():
  base = dodecahedron_spec().compile()
  scaled = dodecahedron_spec(radius_m=0.070, mass_kg=0.240).compile()
  np.testing.assert_allclose(scaled.body_mass[1], 2 * base.body_mass[1])
  np.testing.assert_allclose(scaled.body_inertia[1], 8 * base.body_inertia[1])
  np.testing.assert_allclose(np.linalg.norm(scaled.mesh_vert, axis=1), 0.070, rtol=1e-6)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_dimensions(value):
  with pytest.raises(ValueError):
    dodecahedron_spec(radius_m=value)
  with pytest.raises(ValueError):
    dodecahedron_spec(mass_kg=value)


def test_nominal_physics_and_clean_observations_do_not_modify_cube_task():
  for play in (False, True):
    cfg = wuji_hand_dodecahedron_env_cfg(play=play, num_envs=8)
    assert all(term.mode == "reset" for term in cfg.events.values())
    assert not cfg.curriculum
    assert "dr_params" not in cfg.observations["critic"].terms
    for group in cfg.observations.values():
      assert not group.enable_corruption
      assert all(
        term.params.get("injection_prob", 0) == 0 for term in group.terms.values()
      )
    obj = cfg.scene.entities["object"].spec_fn().compile()
    assert obj.geom_type[0] == mujoco.mjtGeom.mjGEOM_MESH
  cube = wuji_hand_reorient_env_cfg(num_envs=8)
  assert "object_size" in cube.events
  assert "dr_params" in cube.observations["critic"].terms
  assert cube.observations["policy"].enable_corruption
