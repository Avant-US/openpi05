import asyncio
import http
import logging
import socket
import time
import traceback

import serve_policy
import tyro

from openpi.policies import policy as _policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

import cv2
import numpy as _np

# ===== fang'anC: client uploads raw JPEG bytes; server decodes + center-crops =====
# Crop aspect MUST match client config.toml [openpi.crop_aspect] for pixel-identical model input.
_CROP_ASPECT = {
    "head_rgb": (16, 9),
    "left_wrist_rgb": (4, 3),
    "right_wrist_rgb": (4, 3),
}


def _center_crop_to_aspect(img, aspect_w, aspect_h):
    h, w = img.shape[:2]
    target = aspect_w / aspect_h
    cur = w / h
    if abs(cur - target) < 1e-3:
        return img
    if cur > target:
        new_w = int(h * target)
        off = (w - new_w) // 2
        return img[:, off:off + new_w]
    new_h = int(w / target)
    off = (h - new_h) // 2
    return img[off:off + new_h, :]


def _maybe_decode_jpeg_obs(obs):
    """If an image field is 1-D uint8 (JPEG), decode to RGB and center-crop; else passthrough (old client compatible)."""
    for key, aspect in _CROP_ASPECT.items():
        v = obs.get(key)
        if v is None:
            continue
        # msgpack may deliver JPEG payloads as raw bytes; normalize to a 1-D uint8 array.
        if isinstance(v, (bytes, bytearray, memoryview)):
            arr = _np.frombuffer(v, _np.uint8)
        else:
            arr = _np.asarray(v)
        if arr.ndim == 1:
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Failed to decode JPEG for {key}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            obs[key] = _center_crop_to_aspect(rgb, *aspect)
    return obs


logger = logging.getLogger(__name__)


class LatencyWebsocketPolicyServer:
    """Policy server variant that returns server-side timestamps for latency tests."""

    def __init__(
        self,
        policy: _policy.Policy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                t2_recv_ns = time.time_ns()
                # JPEG 解码归入服务端计算(infer 段)，不计入 upload(网络)段。
                obs = _maybe_decode_jpeg_obs(obs)

                infer_start = time.monotonic()
                action = self._policy.infer(obs)
                infer_ms = (time.monotonic() - infer_start) * 1000

                # The robot records t1 before sending and t6 after receiving.
                # These two server timestamps let it compute upload and download latency.
                t5_send_ns = time.time_ns()
                action["server_timing"] = {
                    "infer_ms": infer_ms,
                    "t2_recv_ns": t2_recv_ns,
                    "t5_send_ns": t5_send_ns,
                }
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def main(args: serve_policy.Args) -> None:
    policy = serve_policy.create_policy(args)
    policy_metadata = policy.metadata

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        local_ip = "127.0.0.1"
        logging.warning("Failed to resolve hostname %s, falling back to %s", hostname, local_ip)
    logging.info("Creating latency server (host: %s, ip: %s)", hostname, local_ip)

    server = LatencyWebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(serve_policy.Args))
