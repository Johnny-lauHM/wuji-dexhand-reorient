# 十二面体 SO(3) 手内重定向（仅 mjlab）

任务 ID：`WujiHand_Dodecahedron`。Wuji Hand 将正十二面体转到随机三维目标姿态，
误差低于 0.2 rad 并连续保持 5 个控制步后记一次成功，约 1 秒后切换目标。
这是目标姿态重定向；不包含指定角速度的持续自转或真实手部署。

## 运行

在仓库根目录，使用原有 Pixi 环境：

```bash
pixi run train --task WujiHand_Dodecahedron
pixi run play --task WujiHand_Dodecahedron --checkpoint-file <checkpoint.pt>
pixi run python scripts/dodecahedron_harness.py
```

默认训练为 4096 环境、5000 次 PPO 更新、TensorBoard 本地日志，不上传模型。
显存使用需要以实际运行测量；可用 `--env.scene.num-envs 512` 先做小规模检查。

本次实现也在独立 `.venv` 下验证，可用以下方式重建这个环境：

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match -r scripts/requirements-dodecahedron.txt

.venv/bin/python scripts/train/train_rsl_rl.py --task WujiHand_Dodecahedron
.venv/bin/python scripts/play/play_rsl_rl.py \
  --task WujiHand_Dodecahedron --checkpoint-file <checkpoint.pt> --viewer viser
```

训练输出位于 `logs/rsl_rl/wuji_dodecahedron/<run>/`。
这里的策略需要重新训练；原方块权重不能作为十二面体已收敛的证据，
且新任务的 Critic 输入维度不同，不能直接完整恢复方块训练状态。

## 本地 harness

```bash
# 真实 GPU 仿真、目标状态机、掉落重置、两次 PPO 更新、保存和重新加载。
.venv/bin/python scripts/dodecahedron_harness.py

# 用已有十二面体 checkpoint 做第一目标评测，不再训练。
.venv/bin/python scripts/dodecahedron_harness.py \
  --checkpoint <checkpoint.pt> --num-envs 128 --steps 1000 \
  --output outputs/dodecahedron_eval

# 几何、惯量缩放和配置隔离测试。
.venv/bin/pytest -q src/wuji_mjlab/tasks/dodecahedron/tests
```

默认输出 `outputs/dodecahedron_harness/report.json`、`initial_scene.png`、
`smoke_policy.pt` 和 `ppo/` 日志。无 EGL 时可加 `--no-render`。
任何验收失败都会返回非零退出码，并在 JSON 中记录已完成阶段与错误。

验收要求物理状态、观测和奖励有限，存在指尖接触，目标需要保持后才算成功，
强制逃出手部会重置，PPO 确实修改参数，重新创建 runner 后加载得到相同动作。
评测对每个环境只统计 reset 后的第一个目标，后续切换/重置不重复计数。
默认评测窗口只有 10 秒；`--steps 1000` 为 50 秒。
`status: passed` 表示软件验收通过，**不代表策略达到任意旋转成功率要求**。
两个 PPO iteration 的 smoke checkpoint 仅用于验证管线。

## 实现与边界

- 物体是 20 顶点、12 个正五边形面、36 个三角面的单一凸碰撞 mesh，
  不是方块碰撞体套十二面体外观。默认外接半径 35 mm、质量 120 g。
- MuJoCo 根据实体网格计算惯量；资产工厂支持编译前指定半径和质量。
  `cube`、`cube_mesh`、`cube_center` 是兼容共享传感器/可视化的内部名称。
- 物体自带无碰撞、无质量的 RGB 轴标记，用于区分对称外形的自身朝向。
  12 个面还带有固定编号 1–12 与配色；手内物体和目标模型共用贴图，
  编号下方的底线帮助区分旋转后的 6/9。可运行
  `python scripts/generate_dodecahedron_labels.py` 重建贴图和逐面 UV。
  成功判定使用完整物体四元数，不对几何对称姿态取最小误差。
- 初始位置为 `(-0.140, 0.010, 0.560)` m，相对每个环境原点；
  向远端手指移动以产生指尖接触。位置扰动为 ±3 mm，初始朝向仍均匀采样 SO(3)。
  起始阶段允许物体落入掌部，学习后通过原有奖励鼓励远端手指操控。
- 复用原有 20 关节位置动作、EMA 平滑、20 Hz 控制和 PPO 网络。
  Actor 输入 207 维；Critic 输入 407 维。无物理域随机化、外力扰动、
  观测噪声或位姿异常注入，移除了依赖随机化模型数组的 `dr_params` 观测。
- 原来的 `WujiHand_Reorient` / `WujiHand_Reorient_Light` 配置不变。
  训练与播放均使用新任务注册，不调用原有方块专用 ONNX/真实手评测脚本。

这里只建立了可训练、可评测的任务；达到高成功率需要正式训练和独立种子评测。
