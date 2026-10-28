"""Measure wrist tag tracking on a privileged teacher trajectory (not visual control)."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
import argparse, csv, json, time, xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import asdict
from functools import partial
import cv2, mujoco, numpy as np, torch, mediapy as media
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.models import MLPModel
from tensordict import TensorDict
from train.train_cube_downward import make_cfg
from wuji_mjlab.assets.objects.inhand_object.object_cfg import INHAND_OBJECT_XML
from wuji_mjlab.perception.apriltag import AprilTagBoardEstimator
from wuji_mjlab.tasks.reorient.config.wuji_hand.rsl_rl.ppo import (
  wuji_hand_reorient_ppo_runner_cfg,
)

ROOT = Path(__file__).resolve().parents[1]


def build_cube(out, family):
  cfg = json.loads((ROOT / "hardware/palm_down_cube_45mm/cube_tags45.json").read_text())
  root = ET.parse(INHAND_OBJECT_XML).getroot()
  root.find("compiler").set("assetdir", str(INHAND_OBJECT_XML.parent.parent))
  body = root.find(".//body[@name='cube']")
  body.remove(body.find("geom[@type='mesh']"))
  collision = body.find("geom")
  collision.set("size", "0.0225 0.0225 0.0225")
  collision.set("mass", "0.028192")
  asset = root.find("asset")
  board = {"family": family, "faces": []}
  dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_APRILTAG_36h11 if family == "tag36h11" else cv2.aruco.DICT_4X4_50
  )
  for face, layout in cfg["faces_config"].items():
    axes = cfg["face_axes"][face]
    n = np.array(axes["center"], float)
    u = np.array(axes["u"], float)
    v = np.array(axes["v"], float)
    half = 0.02251
    verts = np.array(
      [
        n * half + u * x * 0.0225 + v * y * 0.0225
        for x, y in [(-1, 1), (1, 1), (1, -1), (-1, -1)]
      ]
    )
    verts = np.vstack((verts, verts - n * 0.00001))
    mesh = out / (face + ".obj")
    mesh.write_text(
      "\n".join(["v " + " ".join(map(str, p)) for p in verts])
      + "\nvt 0 1\nvt 1 1\nvt 1 0\nvt 0 0\nf 1/1 4/4 3/3\nf 1/1 3/3 2/2\nf 5/1 6/2 7/3\nf 5/1 7/3 8/4\nf 1/1 2/2 6/2\nf 1/1 6/2 5/1\nf 2/2 3/3 7/3\nf 2/2 7/3 6/2\nf 3/3 4/4 8/4\nf 3/3 8/4 7/3\nf 4/4 1/1 5/1\nf 4/4 5/1 8/4\n"
    )
    tex = np.full((900, 900), 255, np.uint8)
    angle = cfg["face_rotations"][face] // 90
    for tid, slot in layout.items():
      tid = int(tid) + (100 if family == "tag36h11" else 0)
      x, y = {"T": (0, 0.015), "R": (0.015, 0), "B": (0, -0.015), "L": (-0.015, 0)}[
        slot
      ]
      px = round(450 + x / 0.045 * 900)
      py = round(450 - y / 0.045 * 900)
      marker = np.rot90(
        cv2.aruco.generateImageMarker(dictionary, tid, 216), angle
      ).copy()
      tex[py - 108 : py + 108, px - 108 : px + 108] = marker
      center = n * half + u * x + v * y
      h = 0.0108 / 2  # exact 216 pixels out of 900 on a 45mm face
      corners = np.array(
        [
          center - h * u + h * v,
          center + h * u + h * v,
          center + h * u - h * v,
          center - h * u - h * v,
        ]
      )
      corners = np.roll(corners, angle, axis=0)
      board["faces"].append(
        {
          "id": tid,
          "corners_object_m": corners.tolist(),
          "normal_object": n.tolist(),
          "center_object_m": center.tolist(),
        }
      )
    png = out / (face + ".png")
    cv2.imwrite(str(png), tex)
    ET.SubElement(asset, "mesh", name="wrist_tag_mesh_" + face, file=str(mesh))
    ET.SubElement(
      asset, "texture", name="wrist_tag_tex_" + face, type="2d", file=str(png)
    )
    ET.SubElement(
      asset,
      "material",
      name="wrist_tag_mat_" + face,
      texture="wrist_tag_tex_" + face,
      texuniform="false",
      texrepeat="1 1",
    )
    ET.SubElement(
      body,
      "geom",
      type="mesh",
      mesh="wrist_tag_mesh_" + face,
      material="wrist_tag_mat_" + face,
      group="2",
      contype="0",
      conaffinity="0",
      density="0",
    )
  xml = out / "tagged_cube.xml"
  ET.ElementTree(root).write(xml)
  boardpath = out / "board.json"
  boardpath.write_text(json.dumps(board, indent=2))
  return xml, boardpath


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--family", choices=["tag36h11", "aruco4x4_50"], default="tag36h11")
  p.add_argument("--seconds", type=float, default=12)
  p.add_argument("--seed", type=int, default=444)
  p.add_argument("--output", type=Path, required=True)
  args = p.parse_args()
  out = args.output.resolve()
  out.mkdir(parents=True, exist_ok=False)
  xml, board = build_cube(out, args.family)
  est = AprilTagBoardEstimator(board)
  configure_torch_backends()
  cfg = make_cfg(
    "rotate",
    num_envs=1,
    seed=args.seed,
    cube_edge_m=0.045,
    cube_mass_kg=0.028192,
    wrist_assembly=True,
  )
  cfg.scene.entities["object"].spec_fn = partial(mujoco.MjSpec.from_file, str(xml))
  cfg.viewer.width = 640
  cfg.viewer.height = 480
  cfg.commands["reorient_command"].debug_vis = False
  env = ManagerBasedRlEnv(cfg, device="cuda:0", render_mode="rgb_array")
  renderer = None
  try:
    agent = wuji_hand_reorient_ppo_runner_cfg()
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    obs, _ = wrapped.reset()
    ac = asdict(agent.actor)
    for k in ["class_name", "cnn_cfg", "rnn_type", "rnn_hidden_dim", "rnn_num_layers"]:
      ac.pop(k, None)
    policy = MLPModel(
      TensorDict({"policy": torch.zeros_like(obs["policy"])}, batch_size=[1]),
      {"policy": ["policy"]},
      "policy",
      wrapped.num_actions,
      **ac,
    ).to("cuda:0")
    ckpt = (
      ROOT
      / "logs/rsl_rl/wuji_cube_downward/anypose_rewardfix_20260923_v1/rotate_full_so3_8000/final.pt"
    )
    policy.load_state_dict(
      torch.load(ckpt, map_location="cuda:0", weights_only=False)["actor_state_dict"]
    )
    policy.eval()
    m = env.sim.mj_model
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "robot/d405_b")
    objid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object/cube")
    res = m.cam_resolution[cid]
    intr = m.cam_intrinsic[cid]
    ss = m.cam_sensorsize[cid]
    K = np.array(
      [
        [intr[0] / ss[0] * 640, 0, 319.5 - intr[2] / ss[0] * 640],
        [0, intr[1] / ss[1] * 480, 239.5 - intr[3] / ss[1] * 480],
        [0, 0, 1.0],
      ]
    )
    # Native MuJoCo renderer: x principal offset is reversed; y points down.
    # Pixel centers are integer coordinates, hence the half-pixel shift.
    # Independently audited against rendered marker corners, not used as pose input.
    renderer = mujoco.Renderer(m, height=480, width=640)
    opt = mujoco.MjvOption()
    opt.geomgroup[3:] = 0
    frames = []
    rows = []
    missingrun = 0
    longest = 0
    projection_samples = []
    with torch.inference_mode():
      for step in range(round(args.seconds / env.step_dt)):
        action = policy(TensorDict({"policy": obs["policy"]}, batch_size=[1]))
        obs, _, done, _ = wrapped.step(action)
        env._offline_renderer.update(env.sim.data)
        d = env._offline_renderer._data
        renderer.update_scene(d, camera=cid, scene_option=opt)
        rgb = renderer.render().copy()
        start = time.perf_counter()
        pose = est.estimate(rgb, K)
        latency = (time.perf_counter() - start) * 1000
        raw_corners, raw_ids, _ = est.detector.detectMarkers(
          cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        )
        raw_count = (
          0 if raw_ids is None else sum(int(i) in est.faces for i in raw_ids.ravel())
        )
        row = {
          "raw_tag_count": raw_count,
          "time_s": (step + 1) * env.step_dt,
          "detected": pose is not None,
          "detector_ms": latency,
          "tag_count": 0,
          "position_error_mm": None,
          "rotation_error_deg": None,
          "reprojection_px": None,
        }
        # Ground truth is used ONLY after estimation, for scoring. Teacher remains privileged.
        if pose:
          CR = d.cam_xmat[cid].reshape(3, 3)
          truthR = np.diag([1, -1, -1]) @ CR.T @ d.xmat[objid].reshape(3, 3)
          truthT = np.diag([1, -1, -1]) @ CR.T @ (d.xpos[objid] - d.cam_xpos[cid])
          row.update(
            tag_count=len(pose.ids),
            position_error_mm=float(np.linalg.norm(pose.translation - truthT) * 1000),
            rotation_error_deg=float(
              np.degrees(Rotation.from_matrix(pose.rotation @ truthR.T).magnitude())
            ),
            reprojection_px=pose.reprojection_error_px,
          )
          missingrun = 0
        else:
          missingrun += 1
          longest = max(longest, missingrun)
        if step < 20:
          dc, di, _ = est.detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
          if di is not None:
            CR = d.cam_xmat[cid].reshape(3, 3)
            truthR = np.diag([1, -1, -1]) @ CR.T @ d.xmat[objid].reshape(3, 3)
            truthT = np.diag([1, -1, -1]) @ CR.T @ (d.xpos[objid] - d.cam_xpos[cid])
            for marker, tagid in zip(dc, di.ravel()):
              if int(tagid) in est.faces:
                xyz = (
                  truthR @ np.array(est.faces[int(tagid)]["corners_object_m"]).T
                ).T + truthT
                projection_samples.extend(
                  np.column_stack((xyz[:, :2] / xyz[:, 2, None], marker[0])).tolist()
                )
        rows.append(row)
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, 640, 45), fill="black")
        draw.text(
          (10, 5),
          f"{args.family} | tags {row['tag_count']} | {latency:.1f} ms",
          fill="white",
        )
        draw.text(
          (10, 23),
          f"position {row['position_error_mm']} mm | rotation {row['rotation_error_deg']} deg"
          if pose
          else "NO FRESH VISUAL POSE",
          fill="white",
        )
        frames.append(np.asarray(img))
        if bool(done[0]):
          break
    stats = {
      "family": args.family,
      "seed": args.seed,
      "frames": len(rows),
      "duration_s": len(rows) * env.step_dt,
      "fresh_pose_fraction": sum(r["detected"] for r in rows) / len(rows),
      "longest_missing_s": longest * env.step_dt,
      "K": K.tolist(),
      "scope": "Vision evaluated on privileged teacher trajectory; not vision-driven closed-loop control. Synthetic lighting/no motion blur.",
      "no_decoded_markers_fraction": sum(r["raw_tag_count"] == 0 for r in rows)
      / len(rows),
      "markers_decoded_but_pose_rejected_fraction": sum(
        r["raw_tag_count"] > 0 and not r["detected"] for r in rows
      )
      / len(rows),
      "goals_reached": int(
        env.command_manager.get_term("reorient_command").goal_reach_count[0]
      ),
    }
    if projection_samples:
      samples = np.array(projection_samples)
      fits = []
      for axis in [0, 1]:
        design = np.column_stack((samples[:, axis], np.ones(len(samples))))
        fit = np.linalg.lstsq(design, samples[:, axis + 2], rcond=None)[0]
        fits.append(fit.tolist())
      stats["render_projection_audit_only"] = fits
    for key in [
      "detector_ms",
      "position_error_mm",
      "rotation_error_deg",
      "reprojection_px",
      "tag_count",
    ]:
      vals = [
        r[key]
        for r in rows
        if r[key] is not None and (key != "tag_count" or r["detected"])
      ]
      stats[key] = (
        {
          "median": float(np.median(vals)),
          "p95": float(np.percentile(vals, 95)),
          "max": float(np.max(vals)),
        }
        if vals
        else None
      )
    (out / "report.json").write_text(json.dumps(stats, indent=2))
    media.write_video(str(out / "tracking.mp4"), frames, fps=20)
    Image.fromarray(frames[len(frames) // 2]).save(out / "preview.jpg")
    with (out / "tracking.csv").open("w") as f:
      w = csv.DictWriter(f, fieldnames=rows[0])
      w.writeheader()
      w.writerows(rows)
    print(json.dumps(stats, indent=2))
  finally:
    if renderer:
      renderer.close()
    env.close()


if __name__ == "__main__":
  main()
