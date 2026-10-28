# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import torch
from wuji_mjlab.tasks.reorient.mdp.event_impl.state import get_reorient_event_state
from wuji_mjlab.tasks.reorient.mdp.events import reset_disturbance_caches
from wuji_mjlab.tasks.reorient.mdp.observations import (
  perturbation_direction,
  perturbation_velocity,
)
from wuji_mjlab.tasks.reorient.tests.fakes import make_fake_event_env


def test_reset_disturbance_caches_only_clears_selected_envs():
  env = make_fake_event_env(num_envs=4)
  state = get_reorient_event_state(env)
  state.pert_force_dir = torch.arange(
    12, device=env.device, dtype=torch.float32
  ).reshape(4, 3)
  state.pert_velocity_cache = torch.arange(
    24, device=env.device, dtype=torch.float32
  ).reshape(4, 6)

  reset_disturbance_caches(
    env, torch.tensor([1, 3], dtype=torch.long, device=env.device)
  )

  assert torch.equal(
    state.pert_force_dir[0],  # type: ignore[index]
    torch.tensor([0.0, 1.0, 2.0], device=env.device),
  )
  assert torch.equal(state.pert_force_dir[1], torch.zeros(3, device=env.device))  # type: ignore[index]
  assert torch.equal(
    state.pert_force_dir[2],  # type: ignore[index]
    torch.tensor([6.0, 7.0, 8.0], device=env.device),
  )
  assert torch.equal(state.pert_force_dir[3], torch.zeros(3, device=env.device))  # type: ignore[index]
  assert torch.equal(
    state.pert_velocity_cache[0],  # type: ignore[index]
    torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], device=env.device),
  )
  assert torch.equal(state.pert_velocity_cache[1], torch.zeros(6, device=env.device))  # type: ignore[index]
  assert torch.equal(
    state.pert_velocity_cache[2],  # type: ignore[index]
    torch.tensor([12.0, 13.0, 14.0, 15.0, 16.0, 17.0], device=env.device),
  )
  assert torch.equal(state.pert_velocity_cache[3], torch.zeros(6, device=env.device))  # type: ignore[index]


def test_perturbation_observations_read_shared_event_state():
  env = make_fake_event_env(num_envs=2)
  state = get_reorient_event_state(env)
  state.pert_force_dir[0] = torch.tensor([0.1, 0.2, 0.3], device=env.device)  # type: ignore[index]
  state.pert_velocity_cache[1] = torch.tensor(  # type: ignore[index]
    [1.0, 2.0, 3.0, 0.0, 0.0, 0.0], device=env.device
  )

  direction = perturbation_direction(env)
  velocity = perturbation_velocity(env)

  assert direction is state.pert_force_dir
  assert velocity is state.pert_velocity_cache
  assert torch.equal(direction[0], torch.tensor([0.1, 0.2, 0.3], device=env.device))
  assert torch.equal(
    velocity[1],
    torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], device=env.device),
  )
