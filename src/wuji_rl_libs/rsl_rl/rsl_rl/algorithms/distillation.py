# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer


class Distillation:
  """Distillation algorithm for training a student model to mimic a teacher model."""

  student: MLPModel
  """The student model."""

  teacher: MLPModel
  """The teacher model."""

  teacher_loaded: bool = False
  """Indicates whether the teacher model parameters have been loaded."""

  def __init__(
    self,
    student: MLPModel,
    teacher: MLPModel,
    storage: RolloutStorage,
    num_learning_epochs: int = 1,
    gradient_length: int = 15,
    learning_rate: float = 1e-3,
    max_grad_norm: float | None = None,
    loss_type: str = "mse",
    optimizer: str = "adam",
    auxiliary_target_group: str | None = None,
    auxiliary_loss_coef: float = 0.0,
    auxiliary_loss_type: str = "huber",
    auxiliary_position_loss_coef: float = 1.0,
    auxiliary_rotation_loss_coef: float = 1.0,
    teacher_action_prob_start: float = 0.0,
    teacher_action_prob_end: float = 0.0,
    teacher_action_prob_warmup_updates: int = 0,
    teacher_action_prob_decay_updates: int = 1,
    device: str = "cpu",
    # Distributed training parameters
    multi_gpu_cfg: dict | None = None,
    **kwargs: dict,  # handle unused config parameters
  ) -> None:
    """Initialize the algorithm with models, storage, and optimization settings."""
    # Device-related parameters
    self.device = device
    self.is_multi_gpu = multi_gpu_cfg is not None

    # Multi-GPU parameters
    if multi_gpu_cfg is not None:
      self.gpu_global_rank = multi_gpu_cfg["global_rank"]
      self.gpu_world_size = multi_gpu_cfg["world_size"]
    else:
      self.gpu_global_rank = 0
      self.gpu_world_size = 1

    # Distillation components
    self.student = student.to(self.device)
    self.teacher = teacher.to(self.device)

    # Create the optimizer
    self.optimizer = resolve_optimizer(optimizer)(
      (p for p in self.student.parameters() if p.requires_grad), lr=learning_rate
    )  # type: ignore

    # Add storage
    self.storage = storage
    self.transition = RolloutStorage.Transition()
    self.last_hidden_states = (None, None)

    # Distillation parameters
    self.num_learning_epochs = num_learning_epochs
    self.gradient_length = gradient_length
    self.learning_rate = learning_rate
    self.max_grad_norm = max_grad_norm
    self.auxiliary_target_group = auxiliary_target_group
    self.auxiliary_loss_coef = auxiliary_loss_coef
    self.auxiliary_loss_type = auxiliary_loss_type
    self.auxiliary_position_loss_coef = auxiliary_position_loss_coef
    self.auxiliary_rotation_loss_coef = auxiliary_rotation_loss_coef
    if auxiliary_loss_type not in ("huber", "rotation_6d"):
      raise ValueError(f"Unknown auxiliary_loss_type: {auxiliary_loss_type}")
    if auxiliary_loss_coef and not hasattr(self.student, "predict_auxiliary"):
      raise ValueError(
        "auxiliary_loss_coef requires a student with predict_auxiliary()"
      )
    if not 0.0 <= teacher_action_prob_start <= 1.0:
      raise ValueError("teacher_action_prob_start must be in [0, 1]")
    if not 0.0 <= teacher_action_prob_end <= 1.0:
      raise ValueError("teacher_action_prob_end must be in [0, 1]")
    if teacher_action_prob_warmup_updates < 0:
      raise ValueError("teacher_action_prob_warmup_updates must be nonnegative")
    if teacher_action_prob_decay_updates <= 0:
      raise ValueError("teacher_action_prob_decay_updates must be positive")
    self.teacher_action_prob_start = teacher_action_prob_start
    self.teacher_action_prob_end = teacher_action_prob_end
    self.teacher_action_prob_warmup_updates = teacher_action_prob_warmup_updates
    self.teacher_action_prob_decay_updates = teacher_action_prob_decay_updates
    self.last_teacher_action_prob = teacher_action_prob_start

    # Initialize the loss function
    loss_fn_dict = {
      "mse": nn.functional.mse_loss,
      "huber": nn.functional.huber_loss,
    }
    if loss_type in loss_fn_dict:
      self.loss_fn = loss_fn_dict[loss_type]
    else:
      raise ValueError(
        f"Unknown loss type: {loss_type}. Supported types are: {list(loss_fn_dict.keys())}"
      )

    self.num_updates = 0

  def act(self, obs: TensorDict) -> torch.Tensor:
    """Sample actions and store transition data."""
    # Compute the actions
    student_actions = self.student(obs).detach()
    teacher_actions = self.teacher(obs).detach()
    decay_update = max(self.num_updates - self.teacher_action_prob_warmup_updates, 0)
    progress = min(decay_update / self.teacher_action_prob_decay_updates, 1.0)
    teacher_prob = self.teacher_action_prob_start + progress * (
      self.teacher_action_prob_end - self.teacher_action_prob_start
    )
    self.last_teacher_action_prob = teacher_prob
    use_teacher = (
      torch.rand(student_actions.shape[0], 1, device=self.device) < teacher_prob
    )
    self.transition.actions = torch.where(use_teacher, teacher_actions, student_actions)
    self.transition.privileged_actions = teacher_actions
    # Record the observations
    self.transition.observations = obs
    return self.transition.actions  # type: ignore

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, torch.Tensor],
  ) -> None:
    """Record one environment step and update the normalizers."""
    # Update the normalizers
    self.student.update_normalization(obs)
    # Record the rewards and dones
    self.transition.rewards = rewards
    self.transition.dones = dones
    # Record the transition
    self.storage.add_transition(self.transition)
    self.transition.clear()
    self.student.reset(dones)
    self.teacher.reset(dones)

  def compute_returns(self, obs: TensorDict) -> None:
    """No-op since distillation does not use return targets."""
    # Not needed for distillation
    pass

  def update(self) -> dict[str, float]:
    """Run optimization epochs over stored batches and return mean losses."""
    self.num_updates += 1
    mean_behavior_loss = 0
    mean_auxiliary_loss = 0
    loss = 0
    cnt = 0

    for _epoch in range(self.num_learning_epochs):
      self.student.reset(hidden_state=self.last_hidden_states[0])
      self.teacher.reset(hidden_state=self.last_hidden_states[1])
      self.student.detach_hidden_state()
      for batch in self.storage.generator():
        # Inference of the student for gradient computation
        actions = self.student(batch.observations)

        # Behavior cloning loss
        behavior_loss = self.loss_fn(actions, batch.privileged_actions)

        auxiliary_loss = torch.zeros((), device=self.device)
        if self.auxiliary_loss_coef:
          auxiliary_prediction = self.student.predict_auxiliary(batch.observations)
          auxiliary_target = batch.observations[self.auxiliary_target_group]
          if self.auxiliary_loss_type == "rotation_6d":
            position_loss = nn.functional.huber_loss(
              auxiliary_prediction[:, :9], auxiliary_target[:, :9]
            )
            prediction_matrix = self._rotation_6d_matrix(auxiliary_prediction[:, 9:])
            target_matrix = self._rotation_6d_matrix(auxiliary_target[:, 9:])
            relative = prediction_matrix @ target_matrix.transpose(-1, -2)
            cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(
              -1.0 + 1.0e-6, 1.0 - 1.0e-6
            )
            rotation_loss = torch.acos(cosine).mean()
            auxiliary_loss = (
              self.auxiliary_position_loss_coef * position_loss
              + self.auxiliary_rotation_loss_coef * rotation_loss
            )
          else:
            auxiliary_loss = nn.functional.huber_loss(
              auxiliary_prediction, auxiliary_target
            )

        # Total loss
        loss = loss + behavior_loss + self.auxiliary_loss_coef * auxiliary_loss
        mean_behavior_loss += behavior_loss.item()
        mean_auxiliary_loss += auxiliary_loss.item()
        cnt += 1

        # Gradient step
        if cnt % self.gradient_length == 0:
          self.optimizer.zero_grad()
          loss.backward()
          if self.is_multi_gpu:
            self.reduce_parameters()
          if self.max_grad_norm:
            nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
          self.optimizer.step()
          self.student.detach_hidden_state()
          loss = 0

        # Reset dones
        self.student.reset(batch.dones.view(-1))
        self.teacher.reset(batch.dones.view(-1))
        self.student.detach_hidden_state(batch.dones.view(-1))

    mean_behavior_loss /= cnt
    mean_auxiliary_loss /= cnt
    self.storage.clear()
    self.last_hidden_states = (
      self.student.get_hidden_state(),
      self.teacher.get_hidden_state(),
    )
    self.student.detach_hidden_state()

    # Construct the loss dictionary
    loss_dict = {
      "behavior": mean_behavior_loss,
      "auxiliary": mean_auxiliary_loss,
      "teacher_action_probability": self.last_teacher_action_prob,
    }

    return loss_dict

  @staticmethod
  def _rotation_6d_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Convert flattened last-two-rows encodings to valid rotation matrices."""
    rows = rotation_6d.reshape(rotation_6d.shape[0], -1, 6)
    row_1 = nn.functional.normalize(rows[..., :3], dim=-1, eps=1.0e-6)
    row_2_raw = rows[..., 3:]
    row_2 = nn.functional.normalize(
      row_2_raw - (row_2_raw * row_1).sum(-1, keepdim=True) * row_1,
      dim=-1,
      eps=1.0e-6,
    )
    row_0 = torch.linalg.cross(row_1, row_2, dim=-1)
    return torch.stack((row_0, row_1, row_2), dim=-2)

  def train_mode(self) -> None:
    """Set train mode for the student and keep the teacher in eval mode."""
    self.student.train()
    # Teacher is always in eval mode
    self.teacher.eval()

  def eval_mode(self) -> None:
    """Set evaluation mode for student and teacher models."""
    self.student.eval()
    self.teacher.eval()

  def save(self) -> dict:
    """Return a dict of all models for saving."""
    saved_dict = {
      "student_state_dict": self.student.state_dict(),
      "teacher_state_dict": self.teacher.state_dict(),
      "optimizer_state_dict": self.optimizer.state_dict(),
      "num_updates": self.num_updates,
    }
    return saved_dict

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    """Load specified models from a saved dict."""
    # If no load_cfg is provided, determine what to load automatically
    if load_cfg is None and any(
      "actor_state_dict" in key for key in loaded_dict
    ):  # Load from RL training
      load_cfg = {"teacher": True, "iteration": False}  # Only load teacher by default
    elif load_cfg is None:  # Load from distillation training
      load_cfg = {
        "student": True,
        "teacher": True,
        "optimizer": True,
        "iteration": True,
      }

    # Load the specified models
    if load_cfg.get("student"):
      self.student.load_state_dict(loaded_dict["student_state_dict"], strict=strict)
    if load_cfg.get("teacher"):
      self.teacher.load_state_dict(
        loaded_dict.get("teacher_state_dict") or loaded_dict["actor_state_dict"],
        strict=strict,
      )
      self.teacher_loaded = True
      if load_cfg.get("student") is not True and hasattr(
        self.student, "initialize_policy_from_teacher"
      ):
        self.student.initialize_policy_from_teacher(self.teacher)
    if load_cfg.get("optimizer"):
      self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
    if load_cfg.get("iteration"):
      self.num_updates = loaded_dict.get("num_updates", loaded_dict.get("iter", 0))
    return load_cfg.get("iteration", False)

  def get_policy(self) -> MLPModel:
    """Get the policy model."""
    return self.student

  @staticmethod
  def construct_algorithm(
    obs: TensorDict, env: VecEnv, cfg: dict, device: str
  ) -> Distillation:
    """Construct the distillation algorithm."""
    # Resolve class callables
    alg_class: type[Distillation] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
    student_class: type[MLPModel] = resolve_callable(cfg["student"].pop("class_name"))  # type: ignore
    teacher_class: type[MLPModel] = resolve_callable(cfg["teacher"].pop("class_name"))  # type: ignore

    # Resolve observation groups
    default_sets = ["student", "teacher"]
    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

    # Distillation is not compatible with RND and symmetry extensions
    if cfg["algorithm"].get("rnd_cfg") is not None:
      raise ValueError("The RND extension is not compatible with Distillation.")
    cfg["algorithm"]["rnd_cfg"] = None
    if cfg["algorithm"].get("symmetry_cfg") is not None:
      raise ValueError("The symmetry extension is not compatible with Distillation.")
    cfg["algorithm"]["symmetry_cfg"] = None

    # Initialize the policy
    student: MLPModel = student_class(
      obs, cfg["obs_groups"], "student", env.num_actions, **cfg["student"]
    ).to(device)
    print(f"Student Model: {student}")
    teacher: MLPModel = teacher_class(
      obs, cfg["obs_groups"], "teacher", env.num_actions, **cfg["teacher"]
    ).to(device)
    print(f"Teacher Model: {teacher}")

    # Initialize the storage
    storage = RolloutStorage(
      "distillation",
      env.num_envs,
      cfg["num_steps_per_env"],
      obs,
      [env.num_actions],
      device,
    )

    # Initialize the algorithm
    alg: Distillation = alg_class(
      student,
      teacher,
      storage,
      device=device,
      **cfg["algorithm"],
      multi_gpu_cfg=cfg["multi_gpu"],
    )

    return alg

  def broadcast_parameters(self) -> None:
    """Broadcast model parameters to all GPUs."""
    # Obtain the model parameters on current GPU
    model_params = [self.student.state_dict(), self.teacher.state_dict()]
    # Broadcast the model parameters
    torch.distributed.broadcast_object_list(model_params, src=0)
    # Load the model parameters on all GPUs from source GPU
    self.student.load_state_dict(model_params[0])
    self.teacher.load_state_dict(model_params[1])

  def reduce_parameters(self) -> None:
    """Collect gradients from all GPUs and average them.

    This function is called after the backward pass to synchronize the gradients across all GPUs.
    """
    # Create a tensor to store the gradients
    grads = [
      param.grad.view(-1)
      for param in self.student.parameters()
      if param.grad is not None
    ]
    all_grads = torch.cat(grads)
    # Average the gradients across all GPUs
    torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
    all_grads /= self.gpu_world_size
    # Update the gradients for all parameters with the reduced gradients
    offset = 0
    for param in self.student.parameters():
      if param.grad is not None:
        numel = param.numel()
        # Copy data back from shared buffer
        param.grad.data.copy_(
          all_grads[offset : offset + numel].view_as(param.grad.data)
        )
        # Update the offset for the next parameter
        offset += numel
