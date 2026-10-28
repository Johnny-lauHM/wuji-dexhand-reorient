# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Wuji Hand reorient task registration."""

from mjlab.tasks.registry import register_mjlab_task

from wuji_mjlab.rl.runner import WujiOnPolicyRunner

from .env_cfgs import wuji_hand_cube_downward_env_cfg, wuji_hand_reorient_env_cfg
from .rsl_rl.ppo import wuji_hand_reorient_ppo_runner_cfg

# Canonical release configuration: 8192 envs × 5000 iters reproduces the
# weights published in v0.1.0. Needs ~20 GB GPU memory.
register_mjlab_task(
  task_id="WujiHand_Reorient",
  env_cfg=wuji_hand_reorient_env_cfg(num_envs=8192),
  play_env_cfg=wuji_hand_reorient_env_cfg(play=True),
  rl_cfg=wuji_hand_reorient_ppo_runner_cfg(max_iterations=5000),
  runner_cls=WujiOnPolicyRunner,
)

# Lower-VRAM variant: 4096 envs × 7500 iters. Trains on smaller GPUs but
# converges to a visibly weaker policy — expect occasional cube drops and
# finger-jam behavior on harder reorientations.
register_mjlab_task(
  task_id="WujiHand_Reorient_Light",
  env_cfg=wuji_hand_reorient_env_cfg(num_envs=4096),
  play_env_cfg=wuji_hand_reorient_env_cfg(play=True),
  rl_cfg=wuji_hand_reorient_ppo_runner_cfg(
    run_name="Reorient_Light", max_iterations=7500
  ),
  runner_cls=WujiOnPolicyRunner,
)


register_mjlab_task(
  task_id="WujiHand_Cube_Downward_Teacher",
  env_cfg=wuji_hand_cube_downward_env_cfg(num_envs=4096),
  play_env_cfg=wuji_hand_cube_downward_env_cfg(play=True),
  rl_cfg=wuji_hand_reorient_ppo_runner_cfg(
    run_name="Cube_Downward_Teacher", max_iterations=5000
  ),
  runner_cls=WujiOnPolicyRunner,
)


# Isolated wrist rig: canonical hand with the extracted connector and D405.
register_mjlab_task(
  task_id="WujiHand_Cube_Downward_Wrist_Teacher",
  env_cfg=wuji_hand_cube_downward_env_cfg(
    num_envs=4096, cube_edge_m=0.045,
    object_palm_pos=(0.0255, 0.009, 0.105), wrist_assembly=True,
  ),
  play_env_cfg=wuji_hand_cube_downward_env_cfg(
    play=True, cube_edge_m=0.045,
    object_palm_pos=(0.0255, 0.009, 0.105), wrist_assembly=True,
  ),
  rl_cfg=wuji_hand_reorient_ppo_runner_cfg(
    run_name="Cube_Downward_Wrist_Teacher", max_iterations=5000
  ),
  runner_cls=WujiOnPolicyRunner,
)
