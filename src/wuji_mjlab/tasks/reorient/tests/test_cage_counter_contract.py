# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Behavior contracts for the shared cage counter ownership."""

from types import SimpleNamespace

import torch
from wuji_mjlab.tasks.reorient import mdp
from wuji_mjlab.tasks.reorient.mdp import terminations
from wuji_mjlab.tasks.reorient.mdp.cage import CageEscapePenalty, cage_drop


def test_cage_drop_uses_shared_counter_behavior():
  env = SimpleNamespace(
    _cage_penalty_counter=torch.tensor([9.0, 10.0]),
    num_envs=2,
    device="cpu",
  )

  result = cage_drop(env, max_outside_steps=10)

  assert torch.equal(result, torch.tensor([False, True]))


def test_cage_public_compatibility_reexports_remain_available():
  assert mdp.CageEscapePenalty is CageEscapePenalty
  assert mdp.cage_drop is cage_drop
  assert terminations.CageEscapePenalty is CageEscapePenalty
  assert terminations.cage_drop is cage_drop
