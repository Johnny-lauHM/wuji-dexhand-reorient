"""Bounded thread-parallel detection with a private OpenCV detector per worker."""

from concurrent.futures import ThreadPoolExecutor
from threading import local

from .apriltag import AprilTagBoardEstimator


class BatchAprilTagEstimator:
  def __init__(self, board_path, workers=4):
    if workers < 1:
      raise ValueError("workers must be positive")
    self.board_path = board_path
    self.local = local()
    self.pool = (
      ThreadPoolExecutor(max_workers=workers, thread_name_prefix="apriltag")
      if workers > 1
      else None
    )
    self.closed = False

  def _estimate(self, frame, K):
    if not hasattr(self.local, "estimator"):
      self.local.estimator = AprilTagBoardEstimator(self.board_path)
    # OpenCV RNG is thread-local. Seed each solve so worker scheduling cannot
    # change the RANSAC result for an identical image.
    self.local.estimator.cv2.setRNGSeed(42)
    return self.local.estimator.estimate(frame, K)

  def estimate_batch(self, frames, K):
    if self.closed:
      raise RuntimeError("Batch estimator is closed")
    if self.pool is None:
      return [self._estimate(frame, K) for frame in frames]
    # map preserves environment ordering; all work finishes before buffers are
    # reused by the next control step. No additional sensor delay is introduced.
    return list(self.pool.map(lambda frame: self._estimate(frame, K), frames))

  def close(self):
    if self.pool is not None:
      self.pool.shutdown(wait=True, cancel_futures=True)
    self.closed = True
