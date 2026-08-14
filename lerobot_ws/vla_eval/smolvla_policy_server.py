#!/usr/bin/env python3
"""Serves a fine-tuned SmolVLA checkpoint over ZMQ, for closed-loop rollouts.

This exists because the policy and the robot cannot share a process. LeRobot
needs Python 3.12 (the `lerobot` conda env); rclpy for ROS 2 Humble is built
against Python 3.10 and will not import there. So the policy runs here and the
robot side runs in smolvla_rollout.py, talking over a loopback socket. Raw
frames at 6Hz are ~5MB/s over loopback, which is not worth compressing.

The checkpoint is loaded exactly as training saved it -- no dataset is needed.
config.json carries the trained input/output features, and the preprocessor
pipeline carries the rename (observation.images.scene -> ...camera1) and the
normalization statistics that were computed from the relay dataset. Feed it the
ORIGINAL dataset keys; the rename is the pipeline's first step, not ours.

    conda activate lerobot
    python smolvla_policy_server.py \
        --checkpoint /mnt/hdd/relay_datasets/train_smolvla/checkpoints/020000/pretrained_model
"""

from __future__ import annotations

import argparse
import glob
import os
import socket
import sys
import time
import traceback

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipc import recv_msg, send_msg  # noqa: E402

from lerobot.common.control_utils import predict_action
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors


class PolicyServer:
    def __init__(self, checkpoint: str, device: str):
        self.device = torch.device(device)

        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        cfg.pretrained_path = checkpoint
        cfg.device = device
        self.policy = get_policy_class(cfg.type).from_pretrained(checkpoint, config=cfg)
        self.policy.to(self.device)
        self.policy.eval()

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=checkpoint
        )
        # An action chunk spans chunk_size control steps, so a rollout that
        # starts mid-chunk would replay the tail of the previous episode.
        self.policy.reset()
        self.action_range = self._demonstrated_action_range(checkpoint)
        print(f"loaded {cfg.type} from {checkpoint}", flush=True)
        print(f"  chunk_size={getattr(cfg, 'chunk_size', '?')} "
              f"n_action_steps={getattr(cfg, 'n_action_steps', '?')} device={device}", flush=True)
        if self.action_range is not None:
            low, high = self.action_range
            print(f"  demonstrated action range: {np.round(low, 3)} .. {np.round(high, 3)}", flush=True)

    @staticmethod
    def _demonstrated_action_range(checkpoint: str):
        """(min, max) of `action` over the training set, or None.

        Handed to the robot side so it can refuse to command a joint angle no
        demonstration ever reached. A VLA asked for an out-of-distribution
        observation can return anything, and on a position-controlled arm
        "anything" is a full-speed slam into the table.
        """
        matches = glob.glob(os.path.join(checkpoint, "*unnormalizer*.safetensors"))
        if not matches:
            return None
        stats = load_file(matches[0])
        if "action.min" not in stats or "action.max" not in stats:
            return None
        return (stats["action.min"].numpy().astype(np.float32),
                stats["action.max"].numpy().astype(np.float32))

    def act(self, request: dict) -> np.ndarray:
        observation = {
            "observation.state": np.asarray(request["state"], dtype=np.float32),
            "observation.images.scene": np.asarray(request["scene"], dtype=np.uint8),
            "observation.images.wrist": np.asarray(request["wrist"], dtype=np.uint8),
        }
        action = predict_action(
            observation,
            self.policy,
            self.device,
            self.preprocessor,
            self.postprocessor,
            use_amp=False,
            task=request["task"],
            robot_type="rb5_850e",
        )
        return action.squeeze(0).to("cpu").numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="…/checkpoints/NNNNNN/pretrained_model")
    parser.add_argument("--port", type=int, default=5599)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    server = PolicyServer(args.checkpoint, args.device)

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", args.port))
    listener.listen(1)
    print(f"listening on 127.0.0.1:{args.port}", flush=True)

    try:
        while True:
            connection, _ = listener.accept()
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("client connected", flush=True)
            calls = 0
            try:
                while True:
                    request = recv_msg(connection)
                    command = request.get("cmd")
                    try:
                        if command == "ping":
                            reply = {"ok": True}
                            if server.action_range is not None:
                                reply["action_min"], reply["action_max"] = server.action_range
                        elif command == "reset":
                            server.policy.reset()
                            calls = 0
                            reply = {"ok": True}
                            print("reset", flush=True)
                        elif command == "act":
                            start = time.perf_counter()
                            action = server.act(request)
                            reply = {"ok": True, "action": action,
                                     "infer_ms": (time.perf_counter() - start) * 1e3}
                            calls += 1
                            if calls % 30 == 1:
                                print(f"  step {calls}: {reply['infer_ms']:.0f}ms "
                                      f"action={np.round(action, 3)}", flush=True)
                        else:
                            reply = {"ok": False, "error": f"unknown cmd {command!r}"}
                    except Exception:
                        reply = {"ok": False, "error": traceback.format_exc()}
                        print(reply["error"], flush=True)
                    send_msg(connection, reply)
            except (ConnectionError, EOFError):
                print("client disconnected", flush=True)
            finally:
                connection.close()
    except KeyboardInterrupt:
        pass
    finally:
        listener.close()


if __name__ == "__main__":
    main()
