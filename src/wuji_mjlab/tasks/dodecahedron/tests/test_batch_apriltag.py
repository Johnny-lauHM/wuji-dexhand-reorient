"""Parallel workers preserve image-to-environment ordering and pose results."""

import numpy as np
import pytest
from wuji_mjlab.assets.objects.dodecahedron.object_cfg import DODECAHEDRON_XML
from wuji_mjlab.perception.batch_apriltag import BatchAprilTagEstimator


def test_parallel_matches_serial_with_missing_frames():
  cv2 = pytest.importorskip("cv2")
  cv2.setNumThreads(1)
  board = DODECAHEDRON_XML.with_name("apriltag_board.json")
  K = np.array([[500.0, 0, 199.5], [0, 500.0, 199.5], [0, 0, 1.0]])
  frames = []
  dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
  for tag in (1, 0, 7, 12, 0, 3):
    rgb = np.full((400, 400, 3), 255, dtype=np.uint8)
    rgb[80:320, 80:320] = cv2.aruco.generateImageMarker(dictionary, tag, 240)[..., None]
    frames.append(rgb)
  serial, parallel = BatchAprilTagEstimator(board, 1), BatchAprilTagEstimator(board, 4)
  try:
    expected = serial.estimate_batch(frames, K)
    for _ in range(3):
      actual = parallel.estimate_batch(frames, K)
      for a, b in zip(expected, actual, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
          assert a.ids == b.ids
          np.testing.assert_allclose(a.translation, b.translation, atol=1e-10)
          np.testing.assert_allclose(a.rotation, b.rotation, atol=1e-10)
  finally:
    serial.close()
    parallel.close()
  with pytest.raises(RuntimeError, match="closed"):
    parallel.estimate_batch(frames, K)
