"""Read-only release verification. This script never sends motor commands."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--query-camera", action="store_true")
  parser.add_argument(
    "--camera-serial", default=None, help="RealSense serial to use (default: first device)"
  )
  a = parser.parse_args()
  root = Path(__file__).resolve().parents[2]
  manifest = json.loads((root / "RELEASE_MANIFEST.json").read_text())
  for item in manifest["files"]:
    path = root / item["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
      raise RuntimeError(f"Checksum mismatch: {item['path']}")
  torch.set_num_threads(1)
  model_dir = root / "release_models"
  policy = torch.jit.load(str(model_dir / "policy.jit"), map_location="cpu").eval()
  fixture = np.load(model_dir / "verification.npz", allow_pickle=False)
  with torch.inference_mode():
    actual = policy(torch.from_numpy(fixture["observations"])).numpy()
    np.testing.assert_allclose(actual, fixture["actions"], atol=2e-5, rtol=2e-5)
    single = torch.from_numpy(fixture["observations"][:1])
    for _ in range(10):
      policy(single)
    started = time.perf_counter()
    for _ in range(200):
      policy(single)
    latency_ms = (time.perf_counter() - started) * 1000 / 200
  report = {
    "checksums_verified": len(manifest["files"]),
    "input_dim": 207,
    "action_dim": 20,
    "reference_max_absolute_error": float(np.max(np.abs(actual - fixture["actions"]))),
    "cpu_inference_ms": latency_ms,
    "motor_commands_sent": False,
  }
  if a.query_camera:
    import pyrealsense2 as rs

    expected = json.loads(
      (
        root
        / "src/wuji_mjlab/assets/robots/wuji_hand/meshes/wrist_assembly/camera_calibration.json"
      ).read_text()
    )
    ctx = rs.context()
    device = next(
      (
        d
        for d in ctx.query_devices()
        if a.camera_serial is None
        or d.get_info(rs.camera_info.serial_number) == a.camera_serial
      ),
      None,
    )
    if device is None:
      raise RuntimeError("RealSense camera not found")
    matched = False
    for sensor in device.query_sensors():
      for profile in sensor.get_stream_profiles():
        if (
          profile.stream_type() != rs.stream.color
          or profile.format() != rs.format.rgb8
          or profile.fps() != 30
        ):
          continue
        video = profile.as_video_stream_profile()
        if (video.width(), video.height()) != (640, 480):
          continue
        intrinsic = video.get_intrinsics()
        np.testing.assert_allclose(
          [intrinsic.fx, intrinsic.fy, intrinsic.ppx, intrinsic.ppy],
          [expected[k] for k in ["fx", "fy", "cx", "cy"]],
          atol=1e-4,
        )
        report["camera"] = {
          "serial": device.get_info(rs.camera_info.serial_number),
          "resolution": [640, 480],
          "fov_deg": rs.rs2_fov(intrinsic),
        }
        matched = True
        break
    if not matched:
      raise RuntimeError("Expected RGB 640x480 30fps profile not available")
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
