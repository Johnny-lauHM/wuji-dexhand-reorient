"""Export the current camera student for CPU inference without training imports."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from train.train_cube_downward import make_cfg  # noqa: E402
from train.train_cube_wrist_student import model  # noqa: E402
from wuji_mjlab.perception.cube_wrist import calibration  # noqa: E402


class MeanAction(torch.nn.Module):
  def forward(self, value):
    return value[:, 0, :]


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--checkpoint", type=Path, required=True)
  p.add_argument("--output", type=Path, required=True)
  a = p.parse_args()
  a.output.mkdir(parents=True, exist_ok=True)
  torch.set_num_threads(1)
  actor = model(207, 1, "cpu")
  actor.load_state_dict(
    torch.load(a.checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
  )
  actor.eval()
  network = torch.nn.Sequential(actor.obs_normalizer, actor.mlp, MeanAction()).eval()
  torch.manual_seed(9403)
  inputs = actor.obs_normalizer._mean + torch.randn(32, 207) * actor.obs_normalizer._std
  with torch.inference_mode():
    expected = actor(TensorDict({"policy": inputs}, batch_size=[32]))
    torch.testing.assert_close(network(inputs), expected)
    traced = torch.jit.trace(network, inputs[:1])
    traced.save(str(a.output / "policy.jit"))
    torch.testing.assert_close(traced(inputs), expected)
  torch.onnx.export(
    network,
    inputs[:1],
    str(a.output / "policy.onnx"),
    input_names=["observations"],
    output_names=["actions"],
    dynamic_axes={"observations": {0: "batch"}, "actions": {0: "batch"}},
    opset_version=17,
    dynamo=False,
  )
  session = ort.InferenceSession(
    str(a.output / "policy.onnx"), providers=["CPUExecutionProvider"]
  )
  actual = session.run(None, {"observations": inputs.numpy()})[0]
  np.testing.assert_allclose(actual, expected.numpy(), atol=2e-5, rtol=2e-5)
  np.savez(
    a.output / "verification.npz", observations=inputs.numpy(), actions=expected.numpy()
  )
  cfg = make_cfg(
    "rotate", 1, 7400, cube_edge_m=0.045, cube_mass_kg=0.028192, wrist_assembly=True
  )
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    robot = env.scene["robot"]
    action = cfg.actions["joint_pos"]
    R, position, tag_rotation, K = calibration()
    metadata = {
      "input_dim": 207,
      "action_dim": 20,
      "history_len": 3,
      "ctrl_dt": env.step_dt,
      "control_mode": "absolute",
      "action_semantics": "q_default + clip(raw_action,-1,1)*action_scale; joint-limit clamp; EMA; startup warmup",
      "action_scale": action.action_scale,
      "ema_alpha": action.ema_alpha,
      "warmup_time_s": action.warmup_time_s,
      "joint_names": list(robot.joint_names),
      "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
      "soft_joint_pos_limits": robot.data.soft_joint_pos_limits[0].cpu().tolist(),
      "camera_to_tag_rotation": R.tolist(),
      "camera_origin_tag_m": position.tolist(),
      "tag_in_palm_rotation": tag_rotation.tolist(),
      "camera_K": K.tolist(),
      "visual_timeout_s": 0.15,
      "cube_edge_m": 0.045,
      "cube_mass_kg": 0.028192,
      "checkpoint": str(
        a.checkpoint.relative_to(ROOT)
        if a.checkpoint.is_relative_to(ROOT)
        else a.checkpoint
      ),
      "normalization_embedded": True,
      "onnx_max_absolute_error": float(np.max(np.abs(actual - expected.numpy()))),
      "status": "Inference export verified. Teleoperation-to-policy motor handover is not implemented by this export.",
    }
    (a.output / "policy_config.json").write_text(json.dumps(metadata, indent=2))
    print(
      json.dumps(
        {
          "output": str(a.output),
          "onnx_max_absolute_error": metadata["onnx_max_absolute_error"],
        }
      )
    )
  finally:
    env.close()


if __name__ == "__main__":
  main()
