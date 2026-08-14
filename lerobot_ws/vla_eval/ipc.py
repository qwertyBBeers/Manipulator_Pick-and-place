"""Length-prefixed pickle over a loopback TCP socket.

Shared by smolvla_policy_server.py (Python 3.12, `lerobot` env) and
smolvla_rollout.py (Python 3.10, ROS 2 Humble). Those two interpreters cannot
import each other's packages, so this file must stay pure standard library --
which also happens to be why it is a hand-rolled framer rather than ZeroMQ:
pyzmq is installed in the conda env but not in the system Python that rclpy
needs, and adding it there would mean mutating the system interpreter to run an
experiment.

Arrays are sent as raw buffers plus dtype and shape, never as pickled ndarrays.
The two interpreters carry different major versions of numpy (2.x in the conda
env, 1.x with ROS), and a pickled ndarray from one names a module -- numpy._core
-- that does not exist in the other. Only builtins ever reach pickle here.

Pickle is safe here only because both ends are these two files talking over
127.0.0.1. Do not point this at a socket anything else can reach.
"""

from __future__ import annotations

import pickle
import socket
import struct

import numpy as np

_HEADER = struct.Struct("!Q")
_ARRAY_TAG = "__ndarray__"


def _encode(value):
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {_ARRAY_TAG: contiguous.tobytes(),
                "dtype": contiguous.dtype.str,
                "shape": tuple(contiguous.shape)}
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_encode(v) for v in value)
    return value


def _decode(value):
    if isinstance(value, dict):
        if _ARRAY_TAG in value:
            return np.frombuffer(value[_ARRAY_TAG],
                                 dtype=np.dtype(value["dtype"])).reshape(value["shape"])
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_decode(v) for v in value)
    return value


def send_msg(sock: socket.socket, payload: dict) -> None:
    body = pickle.dumps(_encode(payload), protocol=4)
    sock.sendall(_HEADER.pack(len(body)) + body)


def _recv_exactly(sock: socket.socket, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("peer closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_msg(sock: socket.socket) -> dict:
    (length,) = _HEADER.unpack(_recv_exactly(sock, _HEADER.size))
    return _decode(pickle.loads(_recv_exactly(sock, length)))
