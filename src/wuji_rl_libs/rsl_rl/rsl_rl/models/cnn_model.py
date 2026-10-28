# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import CNN, MLP, EmpiricalNormalization, HiddenState


class CNNModel(MLPModel):
  """CNN-based neural model.

  This model uses one or more convolutional neural network (CNN) encoders to process one or more 2D observation groups
  before passing the resulting latent to an MLP. Any 1D observation groups are directly concatenated with the CNN
  latent and passed to the MLP. 1D observations can be normalized before being passed to the MLP. The output of the
  model can be either deterministic or stochastic, in which case a distribution module is used to sample the outputs.
  """

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    cnn_cfg: dict[str, dict] | dict[str, Any] | None = None,
    cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
  ) -> None:
    """Initialize the CNN-based model.

    Args:
        obs: Observation Dictionary.
        obs_groups: Dictionary mapping observation sets to lists of observation groups.
        obs_set: Observation set to use for this model (e.g., "actor" or "critic").
        output_dim: Dimension of the output.
        hidden_dims: Hidden dimensions of the MLP.
        activation: Activation function of the CNN and MLP.
        obs_normalization: Whether to normalize the observations before feeding them to the MLP.
        distribution_cfg: Configuration dictionary for the output distribution.
        cnn_cfg: Configuration of the CNN encoder(s).
        cnns: CNN modules to use, e.g., for sharing CNNs between actor and critic. If None, new CNNs are created.
    """
    # Resolve observation groups and dimensions
    self._get_obs_dim(obs, obs_groups, obs_set)

    # Create or validate CNN encoders
    if cnns is not None:
      # Check compatibility if CNNs are provided
      if set(cnns.keys()) != set(self.obs_groups_2d):
        raise ValueError(
          "The 2D observations must be identical for all models sharing CNN encoders."
        )
      print(
        "Sharing CNN encoders between models, the CNN configurations of the receiving model are ignored."
      )
    else:
      if cnn_cfg is None:
        raise ValueError("CNN configurations must be provided if CNNs are not shared.")
      # Create a cnn config for each 2D observation group in case only one is provided
      if not all(isinstance(v, dict) for v in cnn_cfg.values()):
        cnn_cfg = {group: cnn_cfg for group in self.obs_groups_2d}
      # Check that the number of configs matches the number of observation groups
      if len(cnn_cfg) != len(self.obs_groups_2d):
        raise ValueError(
          "The number of CNN configurations must match the number of 2D observation groups."
        )
      # Create CNNs for each 2D observation
      cnns = {}
      for idx, obs_group in enumerate(self.obs_groups_2d):
        cnns[obs_group] = CNN(
          input_dim=self.obs_dims_2d[idx],
          input_channels=self.obs_channels_2d[idx],
          **cnn_cfg[obs_group],
        )

    # Compute latent dimension of the CNNs
    self.cnn_latent_dim = 0
    for cnn in cnns.values():
      if cnn.output_channels is not None:
        raise ValueError(
          "The output of the CNN must be flattened before passing it to the MLP."
        )
      self.cnn_latent_dim += int(cnn.output_dim)  # type: ignore

    # Initialize the parent MLP model
    super().__init__(
      obs,
      obs_groups,
      obs_set,
      output_dim,
      hidden_dims,
      activation,
      obs_normalization,
      distribution_cfg,
    )

    # Register CNN encoders
    if isinstance(cnns, nn.ModuleDict):
      self.cnns = cnns
    else:
      self.cnns = nn.ModuleDict(cnns)

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    """Build the model latent by combining normalized 1D and CNN-encoded 2D observation groups."""
    # Concatenate 1D observation groups and normalize
    latent_1d = super().get_latent(obs)
    # Process 2D observation groups with CNNs
    latent_cnn_list = [
      self.cnns[obs_group](obs[obs_group]) for obs_group in self.obs_groups_2d
    ]
    latent_cnn = torch.cat(latent_cnn_list, dim=-1)
    # Concatenate 1D and CNN latents
    return torch.cat([latent_1d, latent_cnn], dim=-1)

  def as_jit(self) -> nn.Module:
    """Return a version of the model compatible with Torch JIT export."""
    return _TorchCNNModel(self)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    """Return a version of the model compatible with ONNX export."""
    return _OnnxCNNModel(self, verbose)

  def _get_obs_dim(
    self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
  ) -> tuple[list[str], int]:
    """Select active observation groups and compute observation dimension."""
    active_obs_groups = obs_groups[obs_set]
    obs_dim_1d = 0
    obs_groups_1d = []
    obs_dims_2d = []
    obs_channels_2d = []
    obs_groups_2d = []

    # Iterate through active observation groups and separate 1D and 2D observations
    for obs_group in active_obs_groups:
      if len(obs[obs_group].shape) == 4:  # B, C, H, W
        obs_groups_2d.append(obs_group)
        obs_dims_2d.append(obs[obs_group].shape[2:4])
        obs_channels_2d.append(obs[obs_group].shape[1])
      elif len(obs[obs_group].shape) == 2:  # B, C
        obs_groups_1d.append(obs_group)
        obs_dim_1d += obs[obs_group].shape[-1]
      else:
        raise ValueError(
          f"Invalid observation shape for {obs_group}: {obs[obs_group].shape}"
        )

    if not obs_groups_2d:
      raise ValueError(
        "No 2D observations are provided. If this is intentional, use the MLP model instead."
      )

    # Store active 2D observation groups and dimensions directly as attributes
    self.obs_dims_2d = obs_dims_2d
    self.obs_channels_2d = obs_channels_2d
    self.obs_groups_2d = obs_groups_2d
    # Return active 1D observation groups and dimension for parent class
    return obs_groups_1d, obs_dim_1d

  def _get_latent_dim(self) -> int:
    """Return the latent dimensionality consumed by the MLP head."""
    return self.obs_dim + self.cnn_latent_dim


class _TorchCNNModel(nn.Module):
  """Exportable CNN model for JIT."""

  def __init__(self, model: CNNModel) -> None:
    """Create a TorchScript-friendly copy of a CNNModel."""
    super().__init__()
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    # Convert ModuleDict to ModuleList for ordered iteration
    self.cnns = nn.ModuleList(
      [copy.deepcopy(model.cnns[g]) for g in model.obs_groups_2d]
    )
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()

  def forward(self, obs_1d: torch.Tensor, obs_2d: list[torch.Tensor]) -> torch.Tensor:
    """Run deterministic inference from separated 1D and 2D inputs."""
    latent_1d = self.obs_normalizer(obs_1d)

    latent_cnn_list = []
    for i, cnn in enumerate(
      self.cnns
    ):  # We assume obs_2d list matches the order of obs_groups_2d
      latent_cnn_list.append(cnn(obs_2d[i]))

    latent_cnn = torch.cat(latent_cnn_list, dim=-1)
    latent = torch.cat([latent_1d, latent_cnn], dim=-1)

    out = self.mlp(latent)
    return self.deterministic_output(out)

  @torch.jit.export
  def reset(self) -> None:
    """Reset recurrent export state (no-op for CNN exports)."""
    pass


class _OnnxCNNModel(nn.Module):
  """Exportable CNN model for ONNX."""

  def __init__(self, model: CNNModel, verbose: bool) -> None:
    """Create an ONNX-export wrapper around a CNNModel."""
    super().__init__()
    self.verbose = verbose
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    # Convert ModuleDict to ModuleList for ordered iteration
    self.cnns = nn.ModuleList(
      [copy.deepcopy(model.cnns[g]) for g in model.obs_groups_2d]
    )
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()

    self.obs_groups_2d = model.obs_groups_2d
    self.obs_dims_2d = model.obs_dims_2d
    self.obs_channels_2d = model.obs_channels_2d
    self.obs_dim_1d = model.obs_dim

  def forward(self, obs_1d: torch.Tensor, *obs_2d: torch.Tensor) -> torch.Tensor:
    """Run deterministic inference for ONNX export."""
    latent_1d = self.obs_normalizer(obs_1d)

    latent_cnn_list = []
    for i, cnn in enumerate(self.cnns):
      latent_cnn_list.append(cnn(obs_2d[i]))

    latent_cnn = torch.cat(latent_cnn_list, dim=-1)
    latent = torch.cat([latent_1d, latent_cnn], dim=-1)

    out = self.mlp(latent)
    return self.deterministic_output(out)

  def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
    """Return representative dummy inputs for ONNX tracing."""
    dummy_1d = torch.zeros(1, self.obs_dim_1d)
    dummy_2d = []
    for i in range(len(self.obs_groups_2d)):
      h, w = self.obs_dims_2d[i]
      c = self.obs_channels_2d[i]
      dummy_2d.append(torch.zeros(1, c, h, w))
    return (dummy_1d, *dummy_2d)

  @property
  def input_names(self) -> list[str]:
    """Return ONNX input tensor names."""
    return ["obs", *self.obs_groups_2d]

  @property
  def output_names(self) -> list[str]:
    """Return ONNX output tensor names."""
    return ["actions"]


class VisionStudentModel(CNNModel):
  """CNN policy with a training-only head for privileged visual targets."""

  def __init__(self, *args, auxiliary_output_dim: int = 27, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.auxiliary_head = nn.Sequential(
      nn.Linear(self._get_latent_dim(), 256),
      nn.ELU(),
      nn.Linear(256, 128),
      nn.ELU(),
      nn.Linear(128, auxiliary_output_dim),
    )

  def predict_auxiliary(self, obs: TensorDict) -> torch.Tensor:
    """Predict the privileged object position and rotation-error history."""
    return self.auxiliary_head(self.get_latent(obs))


def project_rotation_6d_rows(
  rotation_6d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Project the last-two-rows 6D encoding onto SO(3)."""
  shape = rotation_6d.shape
  rows = rotation_6d.reshape(*shape[:-1], -1, 6)
  row_1 = nn.functional.normalize(rows[..., :3], dim=-1, eps=1.0e-6)
  row_2_raw = rows[..., 3:]
  row_2 = nn.functional.normalize(
    row_2_raw - (row_2_raw * row_1).sum(dim=-1, keepdim=True) * row_1,
    dim=-1,
    eps=1.0e-6,
  )
  row_0 = torch.linalg.cross(row_1, row_2, dim=-1)
  matrices = torch.stack((row_0, row_1, row_2), dim=-2)
  projected = torch.cat((row_1, row_2), dim=-1).reshape(shape)
  return projected, matrices


class VisionStateTeacherModel(CNNModel):
  """Estimate privileged history, then execute a frozen copy of the teacher MLP."""

  def __init__(self, *args, auxiliary_output_dim: int = 27, **kwargs) -> None:
    hidden_dims = kwargs.get("hidden_dims", (256, 256, 256))
    activation = kwargs.get("activation", "elu")
    super().__init__(*args, **kwargs)
    if self.obs_dim != 186:
      raise ValueError(
        f"VisionStateTeacherModel expects 186 policy features, got {self.obs_dim}"
      )
    if auxiliary_output_dim != 27:
      raise ValueError(
        "VisionStateTeacherModel requires the teacher's 27 object-state features"
      )
    latent_dim = self._get_latent_dim()
    self.auxiliary_head = nn.Sequential(
      nn.Linear(latent_dim, 512),
      nn.ELU(),
      nn.Linear(512, 256),
      nn.ELU(),
      nn.Linear(256, auxiliary_output_dim),
    )
    # Replace the direct visual action head with the exact teacher topology.
    output_dim = (
      self.distribution.input_dim
      if self.distribution is not None
      else self.mlp[-1].out_features
    )
    self.mlp = MLP(207, output_dim, hidden_dims, activation)
    if self.distribution is not None:
      self.distribution.init_mlp_weights(self.mlp)
    self.teacher_obs_normalizer = EmpiricalNormalization(207)
    self.mlp.requires_grad_(False)
    self.teacher_obs_normalizer.requires_grad_(False)
    self.teacher_policy_initialized = False

  def predict_auxiliary(self, obs: TensorDict) -> torch.Tensor:
    return self.auxiliary_head(super().get_latent(obs))

  def reconstruct_teacher_observation(
    self, obs: TensorDict, estimated_object_state: torch.Tensor
  ) -> torch.Tensor:
    policy = obs["policy"]
    # Teacher order: joint(60), qpos error(60), object(27), actions(60).
    return torch.cat(
      (policy[:, :120], estimated_object_state, policy[:, 120:180]), dim=-1
    )

  def forward(
    self,
    obs: TensorDict,
    masks=None,
    hidden_state=None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    del masks, hidden_state
    estimated_state = self.predict_auxiliary(obs)
    projected_rotation, _ = project_rotation_6d_rows(estimated_state[:, 9:])
    estimated_state = torch.cat((estimated_state[:, :9], projected_rotation), dim=-1)
    teacher_obs = self.reconstruct_teacher_observation(obs, estimated_state)
    mlp_output = self.mlp(self.teacher_obs_normalizer(teacher_obs))
    if self.distribution is not None:
      self.distribution.update(mlp_output)
      if stochastic_output:
        return self.distribution.sample()
      return self.distribution.deterministic_output(mlp_output)
    return mlp_output

  def initialize_policy_from_teacher(self, teacher: MLPModel) -> None:
    """Copy and freeze the converged teacher action mapping and normalizer."""
    self.mlp.load_state_dict(teacher.mlp.state_dict())
    self.teacher_obs_normalizer.load_state_dict(teacher.obs_normalizer.state_dict())
    self.mlp.requires_grad_(False)
    self.teacher_obs_normalizer.requires_grad_(False)
    self.teacher_policy_initialized = True
