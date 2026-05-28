"""Serve a pi05 keyframe-chaining policy with server-side KSM.

Usage (matches `serve_policy.py` but with extra `--ksm.*` flags):

    cd /home/likaixin/kaizhe_ws/pi0.5/openpi

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/serve_keyframe_policy.py \\
        --port 8000 \\
        --ksm.model /home/likaixin/kaizhe_ws/pi0.5/checkpoints/ksm/stage2/best_model.pth \\
        --ksm.backbone /home/likaixin/kaizhe_ws/pi0.5/checkpoints/ksm/stage1/best_backbone.pth \\
        policy:checkpoint \\
        --policy.config pi05_r1pro_chassis_keyframe \\
        --policy.dir checkpoints/pi05_r1pro_chassis_keyframe/keyframe_v1/10000

The client side is unchanged: it only sends head_rgb / left_wrist_rgb /
right_wrist_rgb / state / prompt. KSM detection and keyframe injection
happen entirely on the server, with state held per WebSocket connection
(disconnect == new episode == fresh keyframe queue).
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_keyframe_server as _kf_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported default-environment modes (kept for symmetry with serve_policy.py)."""

    R1PRO_KEYFRAME = "r1pro_keyframe"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    config: str
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default keyframe policy."""


@dataclasses.dataclass
class KsmArgs:
    """KSM (Keyframe Selection Module) options."""

    # Stage-2 KSM checkpoint (full model). Set to None alongside `backbone`
    # to disable KSM (only the very first frame of each episode is used).
    model: str | None = None
    # Stage-1 backbone checkpoint (ResNet-18 contrastive).
    backbone: str | None = None
    # Number of historical keyframes to keep / inject (must match the value
    # used during training; pi05_r1pro_chassis_keyframe uses 3).
    num_keyframes: int = 3
    # Expected number of phases / keyframes per episode (excluding frame 0).
    # 2026-05: 删除"手碰把手"phase（视觉边界 <1.2s 无法区分），从 4 → 3。
    # 对应新 KSM（same-frame multi-phase BCE）+ 新 pi0.5 v2 训练。
    num_phases: int = 3
    # Probability threshold above which a frame is treated as a keyframe.
    threshold: float = 0.5
    # Minimum number of frames between two consecutive keyframes.
    min_gap: int = 10
    # Number of tasks the KSM was trained with (must match checkpoint).
    num_tasks: int = 1
    # Which task embedding to use at inference time.
    task_id: int = 0
    # Device for KSM inference. None -> auto-detect cuda/cpu.
    device: str | None = None
    # If true, do not load KSM at all. Useful as a sanity-check baseline.
    disable: bool = False
    # If true, advance phase only when prob[current_phase] > threshold AND
    # argmax over **current + future** phase queries equals current_phase.
    # This blocks premature jumps to a later phase, while ignoring high
    # probabilities on already-passed phases (which would otherwise
    # falsely block legitimate forward progression). Pass
    # --ksm.no-argmax-gate to disable entirely.
    argmax_gate: bool = True
    # Rollback awareness: if true, watch for sustained evidence that the
    # world has regressed to an earlier phase (e.g. a door that was
    # opened got closed again) and decrement `_current_phase` plus pop
    # the most recently captured keyframe(s). Off by default.
    enable_rollback: bool = False
    # Number of recent frames over which to evaluate rollback evidence.
    # At ~14 Hz, 30 ≈ 2 s of stable observation.
    rollback_window: int = 30
    # `prob[current_phase]` must stay BELOW this for at least
    # `rollback_sustain_ratio` of `rollback_window` to count as low.
    rollback_low_thr: float = 0.1
    # Some past phase k must stay ABOVE this for at least
    # `rollback_sustain_ratio` of `rollback_window` to count as high.
    rollback_high_thr: float = 0.7
    # Fraction of frames in the rollback window that must satisfy each
    # of the low/high conditions. 0.7 ≈ 21/30 frames.
    rollback_sustain_ratio: float = 0.7
    # Min frames since the last forward keyframe / rollback before a
    # new rollback can fire. Default equals `rollback_window`, which is
    # exactly the time needed for the evidence buffer to be fully
    # post-advance. Set higher only if you want extra hard cooldown.
    rollback_min_gap: int = 30


@dataclasses.dataclass
class Args:
    """Arguments for the serve_keyframe_policy script."""

    # Port to serve the policy on.
    port: int = 8000
    # Default prompt to inject if the client did not provide one.
    default_prompt: str | None = None
    # Record policy I/O for debugging.
    record: bool = False
    # KSM configuration.
    ksm: KsmArgs = dataclasses.field(default_factory=KsmArgs)
    # Where to load the policy from.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)
    # Optional default environment selector (kept for parity with serve_policy.py).
    env: EnvMode = EnvMode.R1PRO_KEYFRAME
    # If set, save detected keyframe images to this directory for visual debugging.
    debug_save_dir: str | None = None


# Default checkpoints by environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.R1PRO_KEYFRAME: Checkpoint(
        config="pi05_r1pro_chassis_keyframe",
        dir="checkpoints/pi05_r1pro_chassis_keyframe/keyframe_v1/10000",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        local_ip = "127.0.0.1"
        logging.warning("Failed to resolve hostname %s, falling back to %s", hostname, local_ip)
    logging.info("Creating keyframe server (host: %s, ip: %s)", hostname, local_ip)

    server = _kf_server.WebsocketKeyframePolicyServer(
        policy=policy,
        ksm_model_path=args.ksm.model,
        ksm_backbone_path=args.ksm.backbone,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
        num_keyframes=args.ksm.num_keyframes,
        num_phases=args.ksm.num_phases,
        threshold=args.ksm.threshold,
        min_gap=args.ksm.min_gap,
        num_tasks=args.ksm.num_tasks,
        task_id=args.ksm.task_id,
        ksm_device=args.ksm.device,
        disable_ksm=args.ksm.disable,
        debug_save_dir=args.debug_save_dir,
        argmax_gate=args.ksm.argmax_gate,
        enable_rollback=args.ksm.enable_rollback,
        rollback_window=args.ksm.rollback_window,
        rollback_low_thr=args.ksm.rollback_low_thr,
        rollback_high_thr=args.ksm.rollback_high_thr,
        rollback_sustain_ratio=args.ksm.rollback_sustain_ratio,
        rollback_min_gap=args.ksm.rollback_min_gap,
    )
    server.warmup()
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
