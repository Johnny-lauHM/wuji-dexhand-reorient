"""Replay identical recorded RGB frames through serial/parallel tag estimators."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from wuji_mjlab.perception.batch_apriltag import BatchAprilTagEstimator


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--recording", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  cv2.setNumThreads(1)
  K = np.array(json.loads((args.recording / "calibration.json").read_text())["K"])
  frames = [
    cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    for p in sorted(args.recording.glob("camera_initial_*.png"))
  ]
  if not frames:
    raise ValueError("No recorded RGB frames")
  frames = (frames * 16)[:16]
  rows, reference = [], None
  for workers in (1, 2, 4, 8):
    estimator = BatchAprilTagEstimator(args.recording / "apriltag_board.json", workers)
    try:
      actual = estimator.estimate_batch(frames, K)  # warm up worker-local detectors
      if reference is None:
        reference = actual
      for expected, pose in zip(reference, actual, strict=True):
        assert (expected is None) == (pose is None)
        if pose is not None:
          assert expected.ids == pose.ids
          np.testing.assert_allclose(expected.translation, pose.translation, atol=1e-9)
          np.testing.assert_allclose(expected.rotation, pose.rotation, atol=1e-9)
      times = []
      for _ in range(5):
        tick = time.perf_counter()
        estimator.estimate_batch(frames, K)
        times.append(time.perf_counter() - tick)
      rows.append(
        {
          "workers": workers,
          "batch_size": len(frames),
          "batch_seconds_median": float(np.median(times)),
          "frames_per_second": len(frames) / float(np.median(times)),
          "same_poses": True,
        }
      )
    finally:
      estimator.close()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(rows, indent=2))
  print(json.dumps(rows, indent=2))


if __name__ == "__main__":
  main()
