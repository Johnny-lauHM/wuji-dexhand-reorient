"""Register the independent simulation-only dodecahedron task."""

from mjlab.tasks.registry import register_mjlab_task

from wuji_mjlab.rl.runner import WujiOnPolicyRunner

from .env_cfg import wuji_hand_dodecahedron_env_cfg, wuji_hand_dodecahedron_ppo_cfg

register_mjlab_task(
  task_id="WujiHand_Dodecahedron",
  env_cfg=wuji_hand_dodecahedron_env_cfg(),
  play_env_cfg=wuji_hand_dodecahedron_env_cfg(play=True),
  rl_cfg=wuji_hand_dodecahedron_ppo_cfg(),
  runner_cls=WujiOnPolicyRunner,
)
