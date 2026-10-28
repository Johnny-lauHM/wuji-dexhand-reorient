# L515、十二面体 AprilTag 与重训入口

本次交付包含可识别的 AprilTag 物体资产、实物打印图、几何标定文件、逐面渲染/PnP 验证，以及两条独立训练入口：AprilTag 检测位姿 → MLP，RGB-D → CNN。AprilTag 路线已实际运行仿真图像检测和 PPO 更新；尚未接通真实 L515、完成真机闭环或验证策略收敛。

## 十二面体标记

使用 `tag36h11` ID 1–12，一面一个；ID 0 留给既有手腕标记。默认对应外接半径 35 mm（外接直径 70 mm）的正十二面体。黑色检测方框边长 18 mm，含一格白边的完整方形为 22.5 mm。生成器逐边检查完整白边落在五边形内部。

文件位于 `src/wuji_mjlab/assets/objects/dodecahedron/`：

- `dodecahedron_apriltag.xml` / `.obj`：独立物体变体，复用原顶点与碰撞三角形，只有 UV/材质改变；去掉可视化轴 site。
- `apriltag36h11.png`：真实字典编码生成的纹理 atlas，非装饰性二维码。
- `apriltag_board.json`：每个 ID 的面中心、外法向、图像向右/向上方向及四个检测角点，单位米，均在原始物体 body 坐标系中。
- `apriltag_print_A4.svg`：A4 矢量打印图。按 100% 实际尺寸打印，关闭适应纸张；先用尺量下方 50 mm 校验线，再量标记黑框。虚线是面轮廓参考，文字在面外，完整保留白边。TOP 对应 manifest 的 `up_object`。

当前尺寸是按仓库模型生成的；打印前测量实物。若实物外接尺寸不同，先以实测值重新生成，不能仅缩放打印而继续使用旧的 board JSON。每面 ID、朝向及中心偏移必须与文件一致；贴纸偏移/旋转应回写实测标定。尽量用薄、哑光贴纸；贴纸对手指摩擦的影响需要另行测量，视觉纹理不会自动修改仿真摩擦。

本项目按 OpenCV 的 canonical marker 朝向生成打印图与四角坐标。已抽查 ID 1 与 AprilRobotics 官方位图属于同一编码，但相差 180° 排版旋转；直接换用网上下载的同 ID 图案时必须核对朝向，不能只看 ID 一致。

安装与重生成（仓库根目录）：

```bash
uv pip install --python .venv/bin/python -r scripts/requirements-apriltag.txt
.venv/bin/python scripts/generate_dodecahedron_apriltags.py --radius-mm 35 --tag-size-mm 18
.venv/bin/python scripts/verify_dodecahedron_apriltags.py
```

验证程序在 0.38 m、1280×720 的正对视角逐面渲染，用 OpenCV 的 AprilTag 字典检测器识别，再核对角点投影及 PnP 位姿。输出 `logs/apriltag_verification/all_faces.png` 与 `verification.json`。这是理想逐面识别验证，不代表手指遮挡、运动模糊下的实机识别率。

## 缩小 sim-to-real 的控制路径

优先目标是：L515 RGB → AprilTag 角点检测 → 利用 board JSON 做物体 PnP → 转到手腕 Tag 坐标系 → 关节/位姿策略。深度可以辅助异常检查；标记位姿本身不要求把深度直接喂给 actor。

一个标记即可给出候选物体位姿；多面同时可见时使用同一物体坐标系中的角点联合估计，并做异常角点拒绝。检测器的角点顺序必须对应 JSON 的 top-left、top-right、bottom-right、bottom-left；不同 AprilTag 库要显式转换顺序。OpenCV 光学相机坐标是右、下、前，MuJoCo 相机是右、上、后，转换不可省略。

仿真和实物应使用同一追踪器、内参/畸变处理、尺寸、帧时间定义和丢失策略。之后根据 L515 录制数据拟合位姿噪声、延迟和连续失踪分布，再训练状态策略。只给真实位姿加任意独立噪声，或仅把纹理换成 AprilTag，都不能算已验证感知差异很小。

## AprilTag → 位姿 → policy：实际检测重训

入口为 `scripts/train/train_apriltag_pose.py`，不再使用 `train_l515.py`。每个环境在每个控制步渲染 1280×720 RGB，使用 OpenCV tag36h11 检测器识别标记，按 board JSON 做 PnP。多标记使用 RANSAC 拒绝异常角点并保留完整标记；单标记枚举 IPPE 平面候选、检查正深度/可见面朝向，并按重投影误差选解。代码位于 `src/wuji_mjlab/perception/apriltag.py`，其输入只有图像、相机内参/畸变和已知标记几何，可以供后续实机适配器复用。

actor 输入 **213 维**，每项堆叠 3 帧：

- 20 个归一化关节角、20 个关节目标误差、20 个原始动作。
- 3 维检测物体位置与 6 维“检测旋转相对目标”的误差，均在手腕 Tag 坐标系。
- 1 维当前帧测量是否被接受、1 维位姿年龄（按 0.25 s 截断并归一化）。

没有图像、物体真值位姿或真值 mask 进入 actor/PPO rollout；critic 仍使用 407 维特权输入。物体真值只用于 critic、奖励及独立诊断，诊断误差不会反馈到检测器。位姿与旋转误差同次采集，不分别采样噪声。与旧 207 维 actor 不同，新增有效性/年龄且位姿历史排列改变，旧权重不能直接续训。

当前相机/手腕均固定，相机到 Tag 的变换来自配置中的安装标定，不读取仿真实时物体或手腕位姿。真实部署必须用实际相机/手腕外参替换；此版没有用图像动态追踪手腕 ID 0，不适用于手腕自由移动而仍使用固定外参的情况。

识别/PnP 失败时，本帧有效性为 0；最近成功位姿最多保留 0.25 s，之后输入位置/旋转误差全部置零，年龄为 1，不回退真值。短时跳变根据上一接受测量拒绝：位置变化不得超过 `5 mm + 0.5 m/s × Δt`，旋转不得超过 `0.1 rad + 12 rad/s × Δt`。超过丢失超时后允许重新定位，episode reset 清除各环境缓存。阈值是起步工程配置，实际运动速度、噪声与重获行为需要用实机数据校准；历史门限不能保证拒绝所有初始误识别。

短验证包含探测期、PPO 更新和重载检查：

```bash
cd wuji-mjlab
.venv/bin/python -u scripts/train/train_apriltag_pose.py --smoke
```

扩大到 16 环境做两次正常长度更新：

```bash
.venv/bin/python -u scripts/train/train_apriltag_pose.py --num-envs 16 --iterations 2
```

从随机参数开始的长训命令（本次未启动）：

```bash
.venv/bin/python -u scripts/train/train_apriltag_pose.py --num-envs 16 --iterations 5000
```

默认每轮 40 步。输出在 `logs/rsl_rl/wuji_dodecahedron_apriltag_pose/<timestamp>/`，包含起始相机图、配置、结果 JSON、TensorBoard、定期 checkpoint 与 `final.pt`。`--resume <同接口 checkpoint>` 可续训。训练前有 32 步零动作探测：保存识别比例、位置/旋转误差中位数/P95/最大值、异常帧及吞吐；识别帧的误差是条件指标，缺失帧单独计数。无有效测量或中位数几何误差过大时会停止并保存失败原因。

每个新 run 还会保存 `apriltag_board.json` 和 `calibration.json`，固定该次使用的标记尺寸/四角坐标、内参、相机到 Tag 变换及 OpenCV 版本。`pose_overlay_*.png` 显示检测出的物体坐标轴；不是仿真真值叠加。续训前应核对新旧 run 的 board/标定与 observation 语义，而不只是网络维度。

低维存储减少了显存压力，但检测与渲染仍按每个环境逐帧执行。现已使用有界线程池并行检测，默认 `--detector-workers 8`（实际线程数不超过环境数），每个工作线程持有独立检测器，并固定每帧 RANSAC 随机种子以避免调度改变结果。等待所有测量完成后按环境原顺序返回，不增加额外观测延迟；退出训练时关闭线程池。`--detector-workers 1` 可复现串行路径。不能把旧状态策略 4096 环境的速度直接套过来。

### 加速验证（2026-09-13）

在用户允许临时暂停现有训练后测量，测试完成恢复原进程。未降低 1280×720 分辨率，未修改相机视野、PPO/物理步长、位姿门限或 actor 结构；没有用真值代替检测。

- 相同录制图像、每批 16 帧、预热后 5 次：1/2/4/8 线程分别约 78/155/296/429 帧/秒。串行与并行的 ID、失效位置和位姿在对照容差内一致；这是检测部分吞吐，不是训练吞吐。
- 同样 16 环境、2 次 PPO 更新、每轮 40 步：1/4/8 线程耗时约 23.55/11.27/10.00 秒；8 线程相对串行快约 2.36 倍。32 步零动作 probe 的吞吐约 56.0/120.6/138.0 个环境步/秒，接受率均为 97.85%，未出现诊断阈值外异常。
- 64 环境、8 线程通过 2 次 PPO 更新，训练约 32.61 秒（5120 环境步），probe 约 158.6 个环境步/秒、接受率 98.24%。扩大 batch 改变每轮样本量，不能把 64 环境两次更新耗时直接与 16 环境相比。
- 17 项测试通过，包括并行检测环境顺序、缺帧保留、重复运行的一致性和线程池关闭。

原始记录：`logs/apriltag_speed/workers.json`，以及 `serial/`、`parallel4/`、`parallel8/`、`env64_workers8/` 下的 `result.json`；汇总为 `logs/apriltag_speed/results.tsv`。这是本机短时单种子测试，不是长训效果或所有机器上的速度保证。

显式使用加速参数：

```bash
.venv/bin/python -u scripts/train/train_apriltag_pose.py --num-envs 16 --detector-workers 8 --iterations 5000
```

64 环境已完成短验证，也可将 `--num-envs` 改为 `64`。正在运行的旧 Python 进程不会热加载修改；需在合适的 checkpoint 边界停止旧进程，再用 `--resume <checkpoint>` 启动加速版。不要同时启动两份长训来比较速度。此次只暂停/恢复旧进程，没有丢弃进度或自动切换其代码。

检测部分可复测：

```bash
.venv/bin/python scripts/benchmark_apriltag_workers.py \
  --recording logs/rsl_rl/wuji_dodecahedron_apriltag_pose/20260912_235806_906682 \
  --output logs/apriltag_speed/replay.json
```

本版以 20 Hz 仿真步同步采样，不模拟实际 L515 30 Hz 与控制异步时序，也未加入曝光/运动模糊或额外检测延迟；年龄按仿真时间计算，运行墙钟耗时不会自动变成仿真延迟。物理参数仍为标称值，不能把通过 smoke 当成 sim-to-real gap 已验证足够小。

## RGB-D → CNN 对照脚本

`scripts/train/train_l515.py` 是直接图像输入的独立实验基线，默认使用上述 AprilTag 物体。它**没有在训练循环内运行 AprilTag 检测/PnP**；这与上面的位姿接口路线分开。

actor 输入为 186 维状态（20 关节角、20 关节目标误差、20 原始动作各 3 帧，另加 6 维相机坐标系目标旋转）及 5×96×128 图像（RGB、归一化深度、有效深度 mask）。没有真实物体位置、当前旋转、旋转误差、完整 mesh 点云或真值分割。CNN 与 MLP 一起参加 PPO 更新；critic 保留原有 407 维特权输入。

相机固定在手外约 0.40 m，垂直视野起步用 55°；128×96 是低分辨率工程基线，不是 L515 硬件 profile，也不足以保证标记检测。真实 RGB 内参/视野与深度内参不同，需标定后匹配。仿真按 20 Hz 控制步采图，0–1 步共同图像延迟，默认 2 mm 深度噪声和 2% 独立无效深度；尚未模拟真实 30 Hz 时序、运动模糊、曝光变化、连续遮挡和 L515 材料相关噪声。物理参数沿用十二面体标称配置，尚无物理随机化。这是可跑通的训练入口，不是实机就绪配置。

先运行短验证，会检查多环境近场画面、CNN 参数更新及 checkpoint 重载推理一致性：

```bash
.venv/bin/python scripts/train/train_l515.py --smoke
```

确认输出的 `camera_rgb.png` 中能看到手与物体后，从随机参数训练：

```bash
.venv/bin/python scripts/train/train_l515.py --num-envs 64 --iterations 5000
```

脚本保存图像预览、完整配置、TensorBoard、定期 checkpoint 和 `final.pt`，默认路径为 `logs/rsl_rl/wuji_dodecahedron_l515_rgbd/<timestamp>/`。没有自动启动长训练。显存/吞吐先以 64 环境实测，不沿用状态策略的 4096 环境数。可以用 `--width 256 --height 192` 提高像素数，但网络形状与显存随之改变。

续训只接受同一 RGB-D 架构与分辨率的 checkpoint：

```bash
.venv/bin/python scripts/train/train_l515.py --num-envs 64 --iterations 1000 --resume logs/rsl_rl/wuji_dodecahedron_l515_rgbd/<run>/final.pt
```

旧 207 维状态模型不能直接加载到这个 CNN actor。当前仍只训练单个十二面体，未实现 RotateAnything 多物体泛化。

## 本次验证记录

- 十二面体、视觉观测、实际标记检测、丢失/跳变/重置、无物体真值依赖及并行检测测试：17 项通过。
- 12/12 正对渲染面解码成功；每面最大角点投影偏差小于 1 px，PnP 平移误差约 2.6 mm，默认标称几何的质量/惯量/碰撞一致性检查通过。该误差来自本次理想渲染及像素检测，不是 L515 实测精度。
- 4 环境、2 次 PPO 更新通过；CNN 权重发生更新，checkpoint 重载后的确定性动作一致。最新记录：`logs/rsl_rl/wuji_dodecahedron_l515_rgbd/20260912_233603_903347/result.json`。
- 打印 SVG 已渲染检查版面；实物打印比例仍需按校验尺核对。
- AprilTag 位姿链路：4 环境、2 次短 PPO 更新及重载检查通过。16 环境、32 步零动作探测在加入视觉历史跳变拒绝后，501/512 帧接受位姿，位置误差中位数/P95/最大值约 1.0/3.3/13.0 mm，旋转约 0.72/1.84/3.77°。这是单种子、短时、无真实成像噪声的仿真诊断，误差仅统计被接受的帧。
- 16 环境训练期完成 2 次更新（每轮 40 步），1252/1280 帧接受位姿；probe 吞吐约 56.2 个环境步/秒，两轮训练约 23.5 秒，PPO 观测缓冲 1,587,200 字节（约 1.51 MiB）。采样显卡总占用约 923 MiB 是一次运行中读数，不是严格峰值。结果：`logs/rsl_rl/wuji_dodecahedron_apriltag_pose/20260912_235457_091638/result.json`。
- 未启用视觉历史门限时，同一 16 环境诊断曾出现最大约 125° 的姿态异常；不能仅凭高识别比例判断追踪可靠。门限修复了这次回归，仍需测试高速转动、长遮挡与重获。

## 依据

- AprilRobotics [AprilTag](https://github.com/AprilRobotics/apriltag) 与[官方标记图](https://github.com/AprilRobotics/apriltag-imgs)：标记 family 与检测边界尺寸约定。
- Intel [L515 规格](https://www.intel.com/content/www/us/en/products/sku/201775/intel-realsense-lidar-camera-l515/specifications.html)：最小深度距离 0.25 m。近距实物有效像素与运动追踪需实测。

## 特权状态 teacher → RGB-D student 蒸馏

状态策略长训 checkpoint
`logs/rsl_rl/wuji_dodecahedron/2026-09-12_15-55-31_Dodecahedron/model_4999.pt`
已通过 128/128 首目标评估，可作为动作 teacher。这里的 teacher 是其 207 维状态 actor；
407 维特权 critic 只预测 value，不能直接提供动作标签。student 输入 186 维本体/目标状态和
5×96×128 RGB-D（RGB、深度、有效 mask），不读取物体真值位置或姿态。

先运行两轮端到端检查：

```bash
pixi run python scripts/train/train_l515_distillation.py --smoke
```

再启动正式训练：

```bash
pixi run python scripts/train/train_l515_distillation.py \
  --num-envs 64 \
  --iterations 5000 \
  --teacher logs/rsl_rl/wuji_dodecahedron/2026-09-12_15-55-31_Dodecahedron/model_4999.pt
```

训练前 1000 次更新将环境控制权从 teacher 线性退火到 student：开始时全部执行 teacher
动作以采集稳定轨迹，结束后全部执行 student 动作；每一步始终用 teacher 的确定性动作监督
student。断点会保存并恢复退火进度。`result.json` 中的 `teacher_unchanged` 和
`student_updated` 分别检查 teacher 未被训练以及 student 确实更新；
`convergence_validated` 仍为 false，正式训练后还必须单独评估成功率。
