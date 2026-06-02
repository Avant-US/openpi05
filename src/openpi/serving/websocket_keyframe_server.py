"""WebSocket policy server with server-side keyframe management.

This is a thin wrapper around the standard `WebsocketPolicyServer` that
intercepts each incoming observation and injects historical keyframe images
(and their masks) before delegating to the underlying policy.

Design goals
============
1. **Zero client changes**: clients keep sending the same fields as for the
   non-keyframe pi05 server (`head_rgb`, `left_wrist_rgb`, `right_wrist_rgb`,
   `state`, optional `prompt`). The server is solely responsible for running
   KSM, maintaining the keyframe queue, and adding `keyframe_<i>_rgb` /
   `keyframe_<i>_mask` fields expected by the trained pi05 keyframe model.
2. **Stateful per-connection**: each WebSocket connection gets a fresh
   `_ServerKeyframeManager` instance. When the client disconnects (e.g. at
   the end of an episode) the manager is dropped automatically, so the next
   connection starts a brand new keyframe queue.
3. **Non-invasive**: this file does NOT modify the existing
   `websocket_policy_server.py`, `serve_policy.py`, training config, or
   policy adapters. The training-time behaviour (where keyframes come from
   `meta/keyframes.json` via `DoorKeyframeTransform`) is unchanged.
"""

from __future__ import annotations

import asyncio
from collections import deque
import http
import logging
from pathlib import Path
import threading
import time
import traceback

import numpy as np
import torch
import websockets
import websockets.asyncio.server as _server
import websockets.frames
from PIL import Image
from torchvision import models, transforms as _T

from openpi_client import base_policy as _base_policy
from openpi_client import image_tools
from openpi_client import msgpack_numpy

logger = logging.getLogger(__name__)


class _Pi05ImageTransform:
    """Match pi05 ResizeImages: resize_with_pad to 224x224, then ImageNet normalize."""

    def __init__(self):
        self._to_tensor = _T.ToTensor()
        self._normalize = _T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __call__(self, image):
        image = np.asarray(image)
        if np.issubdtype(image.dtype, np.floating):
            image = (255 * image).astype(np.uint8)
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        image = np.asarray(image_tools.resize_with_pad(image, 224, 224))
        return self._normalize(self._to_tensor(image))


# ---------------------------------------------------------------------------
# KSM (Keyframe Selection Module) - lightweight inference-only copy.
# Architecture must match `scripts/keyframes/train_ksm.py` and
# `scripts/keyframes/infer_ksm.py`.
# ---------------------------------------------------------------------------


class _TransformerKeyframeSelector(torch.nn.Module):
    """Inference-only KSM. Loads the same checkpoint format produced by
    `scripts/keyframes/train_ksm.py` (Stage 1 backbone + Stage 2 model)."""

    def __init__(
        self,
        pretrained_backbone_path: str,
        num_tasks: int = 1,
        window_size: int = 3,
        embed_dim: int = 256,
        num_heads: int = 4,
        max_phases: int = 20,
    ) -> None:
        super().__init__()
        self.window_size = window_size

        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = torch.nn.Identity()
        state = torch.load(pretrained_backbone_path, map_location="cpu")
        self.backbone.load_state_dict(state)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.feat_proj = torch.nn.Linear(512, embed_dim)
        self.time_embedding = torch.nn.Parameter(torch.randn(window_size, embed_dim) * 0.02)
        self.task_embedding = torch.nn.Embedding(num_tasks, embed_dim)
        self.phase_embedding = torch.nn.Embedding(max_phases, embed_dim)

        self.self_attn = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = torch.nn.LayerNorm(embed_dim)

        self.film_gamma = torch.nn.Linear(embed_dim, embed_dim)
        self.film_beta = torch.nn.Linear(embed_dim, embed_dim)

        self.cross_attn = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = torch.nn.LayerNorm(embed_dim)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 1),
        )

    def forward(self, imgs, phase_ids, task_ids):
        B, Tw, C, H, W = imgs.shape
        imgs_flat = imgs.view(B * Tw, C, H, W)
        with torch.no_grad():
            feats = self.backbone(imgs_flat)
        feats = feats.view(B, Tw, 512)
        feats = self.feat_proj(feats) + self.time_embedding[:Tw]

        attn_out, _ = self.self_attn(feats, feats, feats)
        feats = self.norm1(feats + attn_out)

        task_emb = self.task_embedding(task_ids)
        gamma = self.film_gamma(task_emb)
        beta = self.film_beta(task_emb)

        phase_emb = self.phase_embedding(phase_ids)
        query = phase_emb * (1 + gamma) + beta
        query = query.unsqueeze(1)

        cross_out, _ = self.cross_attn(query, feats, feats)
        out = self.norm2(query + cross_out).squeeze(1)
        return self.classifier(out)


# ---------------------------------------------------------------------------
# Per-connection state container
# ---------------------------------------------------------------------------


class _ServerKeyframeManager:
    """Maintains a per-connection keyframe queue and runs KSM on incoming
    head images. Equivalent to `scripts/keyframes/keyframe_manager.py` but
    operates on the server side. The KSM model is shared (passed in) across
    connections, while the queue/buffer/state are per instance."""

    def __init__(
        self,
        ksm_model: _TransformerKeyframeSelector | None,
        device: str,
        num_keyframes: int = 3,
        num_phases: int = 3,
        threshold: float = 0.5,
        min_gap: int = 10,
        task_id: int = 0,
        debug_save_dir: str | None = None,
        argmax_gate: bool = True,
        enable_rollback: bool = False,
        rollback_window: int = 30,
        rollback_low_thr: float = 0.1,
        rollback_high_thr: float = 0.7,
        rollback_sustain_ratio: float = 0.7,
        rollback_min_gap: int = 30,
    ) -> None:
        self._ksm = ksm_model
        self._device = device
        self.num_keyframes = num_keyframes
        self.num_phases = num_phases
        self.threshold = threshold
        self.min_gap = min_gap
        self.task_id = task_id
        # Future-only argmax gate: advance only when prob[current_phase] >
        # threshold AND argmax over current + future phases equals
        # current_phase. Past-phase probabilities are ignored so a still-
        # high "phase 0/1 ending" signal cannot block the legitimate
        # transition into a later phase.
        self.argmax_gate = argmax_gate
        # Rollback awareness: detect that the world has regressed to an
        # earlier phase (e.g. a door that was opened got closed again) and
        # decrement `_current_phase` plus pop the most recent keyframes.
        # Off by default; only triggers under sustained, conservative
        # evidence to avoid thrashing.
        self.enable_rollback = enable_rollback
        self.rollback_low_thr = rollback_low_thr
        self.rollback_high_thr = rollback_high_thr
        self.rollback_sustain_ratio = rollback_sustain_ratio
        self.rollback_min_gap = rollback_min_gap
        self._debug_save_dir: Path | None = None
        if debug_save_dir:
            self._debug_save_dir = Path(debug_save_dir)
            self._debug_save_dir.mkdir(parents=True, exist_ok=True)

        self._img_transform = _Pi05ImageTransform()

        self._keyframe_queue: deque[np.ndarray] = deque(maxlen=num_keyframes)
        self._frame_buffer: deque[torch.Tensor] = deque(maxlen=3)
        self._recent_probs: deque[np.ndarray] = deque(maxlen=max(1, rollback_window))
        self._step_count: int = 0
        self._current_phase: int = 0
        self._last_kf_step: int = -self.min_gap
        self._last_prob: float | None = None
        self._connection_id: str = ""
        self._debug_records: list[dict] = []
        # Protects the shared keyframe / phase / step state. Held by
        # `update_from_frame` (writer) and the snapshot section of
        # `inject_keyframes` (reader); released before heavy obs assembly
        # to keep the critical section short.
        self._lock = threading.Lock()

    def update_from_frame(self, head_rgb: np.ndarray, *, source: str = "stream") -> None:
        """Streaming entry point: run KSM and update the keyframe queue.

        Thread-safe: holds `_lock` while mutating shared state. No return
        value; consumers read the latest queue via `inject_keyframes`.
        Called by `_stream_worker` (and by `step` for the warmup path).
        """
        head_rgb = np.asarray(head_rgb)
        with self._lock:
            self._append_ksm_frame(head_rgb)

            if self._step_count == 0:
                self._keyframe_queue.append(head_rgb.copy())
                self._last_kf_step = 0
                logger.info(
                    "[KSM] step=0 → keyframe (first frame), phase=0, queue=%d",
                    len(self._keyframe_queue),
                )
                self._save_debug_image(head_rgb, step=0, prob=1.0, phase=0)
                self._record_debug_step(
                    step=0, prob=1.0, is_keyframe=True, phase=0,
                    reason="first_frame", source=source,
                )
            elif self._ksm is not None and (
                self._current_phase < self.num_phases or self.enable_rollback
            ):
                phase_probs = self._detect_all_phase_probs(head_rgb)
                if phase_probs is None:
                    self._record_debug_step(
                        step=self._step_count, prob=0.0, is_keyframe=False,
                        phase=self._current_phase, reason="buffer_warmup",
                        source=source, phase_probs=None, argmax_phase=None,
                    )
                else:
                    self._recent_probs.append(phase_probs)
                    argmax_phase = int(np.argmax(phase_probs))

                    if self.enable_rollback:
                        rollback_target = self._check_rollback()
                        if rollback_target is not None:
                            old_phase = self._current_phase
                            pops = old_phase - rollback_target
                            for _ in range(pops):
                                if self._keyframe_queue:
                                    self._keyframe_queue.pop()
                            self._current_phase = rollback_target
                            self._last_kf_step = self._step_count
                            self._recent_probs.clear()
                            self._last_prob = float(phase_probs[rollback_target]) \
                                if rollback_target < self.num_phases else None
                            logger.warning(
                                "[KSM] step=%d → ROLLBACK phase %d → %d "
                                "(popped %d keyframe(s), queue=%d, "
                                "phase_probs=%s)",
                                self._step_count, old_phase, rollback_target,
                                pops, len(self._keyframe_queue),
                                ["%.3f" % p for p in phase_probs],
                            )
                            self._record_debug_step(
                                step=self._step_count,
                                prob=self._last_prob if self._last_prob is not None else 0.0,
                                is_keyframe=False, phase=rollback_target,
                                reason="rollback", source=source,
                                phase_probs=phase_probs, argmax_phase=argmax_phase,
                                rollback_from=old_phase,
                            )
                            self._step_count += 1
                            return

                    if self._current_phase < self.num_phases:
                        prob = float(phase_probs[self._current_phase])
                        future_probs = phase_probs[self._current_phase:]
                        local_argmax = self._current_phase + int(np.argmax(future_probs))
                        self._last_prob = prob

                        gap_ok = self._step_count - self._last_kf_step >= self.min_gap
                        above_thr = prob > self.threshold
                        argmax_ok = (not self.argmax_gate) or (local_argmax == self._current_phase)

                        if above_thr and gap_ok and argmax_ok:
                            self._keyframe_queue.append(head_rgb.copy())
                            self._current_phase += 1
                            self._last_kf_step = self._step_count
                            logger.info(
                                "[KSM] step=%d → KEYFRAME DETECTED! prob=%.4f "
                                "argmax(global)=%d argmax(future)=%d "
                                "phase_probs=%s, phase=%d, queue=%d",
                                self._step_count, prob, argmax_phase, local_argmax,
                                ["%.3f" % p for p in phase_probs],
                                self._current_phase, len(self._keyframe_queue),
                            )
                            self._save_debug_image(
                                head_rgb, step=self._step_count, prob=prob,
                                phase=self._current_phase,
                            )
                            self._record_debug_step(
                                step=self._step_count, prob=prob, is_keyframe=True,
                                phase=self._current_phase, reason="ksm_detected",
                                source=source, phase_probs=phase_probs,
                                argmax_phase=argmax_phase,
                            )
                        else:
                            if above_thr and gap_ok and not argmax_ok:
                                reason = "argmax_suppressed"
                                logger.info(
                                    "[KSM] step=%d → ARGMAX SUPPRESSED "
                                    "prob_curr[%d]=%.4f, argmax(future)=phase%d "
                                    "(prob=%.4f), no advance",
                                    self._step_count, self._current_phase, prob,
                                    local_argmax, phase_probs[local_argmax],
                                )
                            elif above_thr and not gap_ok:
                                reason = "min_gap_blocked"
                            else:
                                reason = "below_threshold"

                            if self._step_count % 50 == 0:
                                logger.info(
                                    "[KSM] step=%d, prob_curr[%d]=%.4f "
                                    "argmax(global)=phase%d argmax(future)=phase%d "
                                    "phase_probs=%s, no advance (%s)",
                                    self._step_count, self._current_phase, prob,
                                    argmax_phase, local_argmax,
                                    ["%.3f" % p for p in phase_probs], reason,
                                )
                            self._record_debug_step(
                                step=self._step_count, prob=prob, is_keyframe=False,
                                phase=self._current_phase, reason=reason,
                                source=source, phase_probs=phase_probs,
                                argmax_phase=argmax_phase,
                            )
                    else:
                        self._record_debug_step(
                            step=self._step_count, prob=0.0, is_keyframe=False,
                            phase=self._current_phase,
                            reason="terminal_rollback_watch",
                            source=source, phase_probs=phase_probs,
                            argmax_phase=argmax_phase,
                        )
            else:
                self._record_debug_step(
                    step=self._step_count, prob=0.0, is_keyframe=False,
                    phase=self._current_phase,
                    reason="ksm_disabled_or_finished",
                    source=source, phase_probs=None, argmax_phase=None,
                )

            self._step_count += 1

    def inject_keyframes(self, obs: dict) -> dict:
        """Policy entry point: read-only injection of `keyframe_<i>_rgb` and
        `keyframe_<i>_mask` based on the current queue snapshot.

        Does not advance KSM state. Strips the protocol-level `type` field
        (if present) before returning so the augmented obs is suitable for
        passing straight to `policy.infer`.
        """
        if "head_rgb" not in obs:
            raise KeyError("Server-side keyframe injection requires 'head_rgb' in observation")
        head_rgb = np.asarray(obs["head_rgb"])

        # Snapshot under lock, release before heavy dict assembly.
        with self._lock:
            kf_snapshot = [kf.copy() for kf in self._keyframe_queue]

        out = dict(obs)
        out.pop("type", None)
        num_pad = self.num_keyframes - len(kf_snapshot)
        for i in range(num_pad):
            out[f"keyframe_{i}_rgb"] = np.zeros_like(head_rgb)
            out[f"keyframe_{i}_mask"] = False
        for j, kf_img in enumerate(kf_snapshot):
            slot = num_pad + j
            if kf_img.shape != head_rgb.shape:
                # Rare resolution mismatch (e.g. a stream thumbnail entered
                # the queue while policy_request still uses native res).
                import cv2
                kf_img = cv2.resize(kf_img, (head_rgb.shape[1], head_rgb.shape[0]))
            out[f"keyframe_{slot}_rgb"] = kf_img
            out[f"keyframe_{slot}_mask"] = True
        return out

    def step(self, obs: dict) -> dict:
        """Sequential composite: run KSM on `obs["head_rgb"]` then inject
        keyframes. Kept for backward compatibility (warmup, old non-stream
        clients). New code should use `update_from_frame` + `inject_keyframes`.
        """
        if "head_rgb" not in obs:
            raise KeyError("Server-side keyframe injection requires 'head_rgb' in observation")
        self.update_from_frame(obs["head_rgb"], source="policy")
        return self.inject_keyframes(obs)

    def _record_debug_step(
        self,
        *,
        step: int,
        prob: float,
        is_keyframe: bool,
        phase: int,
        reason: str,
        source: str = "stream",
        phase_probs: np.ndarray | None = None,
        argmax_phase: int | None = None,
        rollback_from: int | None = None,
    ) -> None:
        """Caller must hold `self._lock`."""
        if self._debug_save_dir is None:
            return
        self._debug_records.append(
            {
                "step": step,
                "prob": prob,
                "is_keyframe": is_keyframe,
                "phase": phase,
                "reason": reason,
                "source": source,
                "phase_probs": (
                    [float(p) for p in phase_probs] if phase_probs is not None else None
                ),
                "argmax_phase": argmax_phase,
                "rollback_from": rollback_from,
            }
        )

    def _check_rollback(self) -> int | None:
        """Decide whether to roll `_current_phase` back. Returns the
        target phase, or None if no rollback should be triggered.

        The signature of a real-world regression: for the last
        ``rollback_window`` frames, ``prob[current_phase]`` is sustained
        below ``rollback_low_thr`` AND some past phase ``k`` has
        ``prob[k]`` sustained above ``rollback_high_thr``. We pick the
        largest such ``k`` (smallest rollback distance) and return
        ``k + 1`` so that the next phase to detect is the one that
        comes after the now-recognised "end of phase k".

        Caller must hold ``self._lock``.
        """
        if self._current_phase <= 0:
            return None
        if self._step_count - self._last_kf_step < self.rollback_min_gap:
            return None
        max_w = self._recent_probs.maxlen or 0
        if max_w <= 0 or len(self._recent_probs) < max_w:
            return None

        window = np.stack(list(self._recent_probs))  # (W, num_phases)
        n = float(len(window))

        if self._current_phase < self.num_phases:
            curr_low_ratio = float((window[:, self._current_phase] < self.rollback_low_thr).sum()) / n
        else:
            # Terminal state: there is no `current_phase` column; treat
            # "current low" as automatically satisfied so rollback can
            # still fire when the world clearly regressed past finish.
            curr_low_ratio = 1.0
        if curr_low_ratio < self.rollback_sustain_ratio:
            return None

        past_window = window[:, : self._current_phase]  # (W, current_phase)
        past_high_ratio = (past_window > self.rollback_high_thr).sum(axis=0) / n
        candidates = np.where(past_high_ratio >= self.rollback_sustain_ratio)[0]
        if candidates.size == 0:
            return None

        last_high_phase = int(candidates.max())
        target_phase = last_high_phase + 1
        if target_phase >= self._current_phase:
            return None
        return target_phase

    def save_debug_summary(self) -> None:
        """Persist per-step KSM probabilities for this connection."""
        if self._debug_save_dir is None or not self._connection_id:
            return
        conn_dir = self._debug_save_dir / self._connection_id
        conn_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot debug records under lock (writer thread may still be
        # running if the connection died mid-stream).
        with self._lock:
            records = list(self._debug_records)
            total_steps = self._step_count
            final_phase = self._current_phase

        probs_path = conn_dir / "probs.txt"
        phase_cols = ",".join(f"prob_phase{i}" for i in range(self.num_phases))
        with probs_path.open("w") as f:
            f.write(
                "step,prob,is_keyframe,phase,argmax_phase,rollback_from,reason,"
                f"source,{phase_cols}\n"
            )
            for item in records:
                if item["is_keyframe"]:
                    marker = " *** KEYFRAME ***"
                elif item.get("reason") == "rollback":
                    marker = " *** ROLLBACK ***"
                else:
                    marker = ""
                argmax_str = (
                    str(item["argmax_phase"]) if item.get("argmax_phase") is not None else ""
                )
                rollback_str = (
                    str(item["rollback_from"]) if item.get("rollback_from") is not None else ""
                )
                pp = item.get("phase_probs")
                if pp is not None:
                    phase_probs_str = ",".join(f"{p:.6f}" for p in pp)
                else:
                    phase_probs_str = ",".join("" for _ in range(self.num_phases))
                f.write(
                    f"{item['step']},{item['prob']:.6f},{1 if item['is_keyframe'] else 0},"
                    f"{item['phase']},{argmax_str},{rollback_str},"
                    f"{item['reason']},{item.get('source', 'stream')},"
                    f"{phase_probs_str}{marker}\n"
                )

        summary_path = conn_dir / "summary.txt"
        keyframes = [item for item in records if item["is_keyframe"]]
        rollbacks = [item for item in records if item.get("reason") == "rollback"]
        valid_probs = [item["prob"] for item in records if item["reason"] == "below_threshold"]
        with summary_path.open("w") as f:
            f.write(f"Total frames: {total_steps}\n")
            f.write(f"Final phase: {final_phase}\n")
            f.write(f"Keyframes detected: {len(keyframes)}\n")
            f.write(f"Rollbacks: {len(rollbacks)}\n")
            if valid_probs:
                f.write(
                    f"Prob stats: min={min(valid_probs):.4f}, max={max(valid_probs):.4f}, "
                    f"last={valid_probs[-1]:.4f}, threshold={self.threshold:.4f}\n"
                )
            f.write("\nDetailed keyframes:\n")
            for item in keyframes:
                f.write(
                    f"  step={item['step']:5d}  prob={item['prob']:.4f}  "
                    f"phase={item['phase']}  ({item['reason']}, src={item.get('source', 'stream')})\n"
                )
            if rollbacks:
                f.write("\nDetailed rollbacks:\n")
                for item in rollbacks:
                    f.write(
                        f"  step={item['step']:5d}  phase {item.get('rollback_from')} "
                        f"-> {item['phase']}  (src={item.get('source', 'stream')})\n"
                    )
        logger.info("[KSM] Saved debug probabilities → %s", probs_path)

    def _save_debug_image(self, img: np.ndarray, step: int, prob: float, phase: int) -> None:
        """Save a keyframe image to the debug directory (if configured)."""
        if self._debug_save_dir is None:
            return
        fname = f"kf_step{step:05d}_phase{phase}_prob{prob:.3f}.jpg"
        path = self._debug_save_dir / self._connection_id / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        pil_img = Image.fromarray(img if img.dtype == np.uint8 else (img * 255).astype(np.uint8))
        pil_img.save(path, quality=90)
        logger.info("[KSM] Saved keyframe image → %s", path)

    def _append_ksm_frame(self, head_rgb: np.ndarray) -> None:
        if self._ksm is None:
            return
        img = Image.fromarray(head_rgb if head_rgb.dtype == np.uint8 else (head_rgb * 255).astype(np.uint8))
        self._frame_buffer.append(self._img_transform(img))

    def _detect_keyframe_prob(self, head_rgb: np.ndarray) -> float | None:
        """Return KSM probability for the current frame, or None if
        the frame buffer is not yet full (need 3 frames)."""
        if self._ksm is None:
            return None
        if len(self._frame_buffer) < 3:
            return None
        window = torch.stack(list(self._frame_buffer)).unsqueeze(0).to(self._device)
        phase_t = torch.tensor([self._current_phase], dtype=torch.long, device=self._device)
        task_t = torch.tensor([self.task_id], dtype=torch.long, device=self._device)
        with torch.no_grad():
            logit = self._ksm(window, phase_t, task_t)
            prob = torch.sigmoid(logit).item()
        self._last_prob = prob
        return prob

    def _detect_all_phase_probs(self, head_rgb: np.ndarray) -> np.ndarray | None:
        """Run KSM with phase_id = 0..num_phases-1 in a single batched
        forward pass. Returns shape (num_phases,) or None if the frame
        buffer is not yet full.
        """
        if self._ksm is None:
            return None
        if len(self._frame_buffer) < 3:
            return None
        window = torch.stack(list(self._frame_buffer)).unsqueeze(0).to(self._device)
        # Repeat the same window num_phases times along batch dim, vary phase_id.
        # `.expand` returns a non-contiguous view; `.contiguous()` is required
        # because the model's forward calls `imgs.view(...)` which rejects
        # non-contiguous strides.
        window_rep = window.expand(self.num_phases, -1, -1, -1, -1).contiguous()
        phase_t = torch.arange(self.num_phases, dtype=torch.long, device=self._device)
        task_t = torch.full((self.num_phases,), self.task_id, dtype=torch.long, device=self._device)
        with torch.no_grad():
            logits = self._ksm(window_rep, phase_t, task_t)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        return probs

    def _detect_keyframe(self, head_rgb: np.ndarray) -> bool:
        prob = self._detect_keyframe_prob(head_rgb)
        if prob is None:
            return False
        return prob > self.threshold

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "step": self._step_count,
                "phase": self._current_phase,
                "keyframes_in_queue": len(self._keyframe_queue),
                "last_prob": getattr(self, "_last_prob", None),
                "last_kf_step": self._last_kf_step if self._last_kf_step >= 0 else None,
            }


# ---------------------------------------------------------------------------
# WebSocket server
# ---------------------------------------------------------------------------


class WebsocketKeyframePolicyServer:
    """A WebSocket policy server that performs server-side keyframe
    detection. Drop-in replacement for
    `openpi.serving.websocket_policy_server.WebsocketPolicyServer` when the
    underlying policy was trained with the keyframe chaining configuration
    (e.g. `pi05_r1pro_chassis_keyframe`).

    Each new WebSocket connection receives its own `_ServerKeyframeManager`
    instance. KSM weights are loaded once at construction time and shared.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        *,
        ksm_model_path: str | None,
        ksm_backbone_path: str | None,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        num_keyframes: int = 3,
        num_phases: int = 3,
        threshold: float = 0.5,
        min_gap: int = 10,
        num_tasks: int = 1,
        task_id: int = 0,
        ksm_device: str | None = None,
        disable_ksm: bool = False,
        debug_save_dir: str | None = None,
        argmax_gate: bool = True,
        enable_rollback: bool = False,
        rollback_window: int = 30,
        rollback_low_thr: float = 0.1,
        rollback_high_thr: float = 0.7,
        rollback_sustain_ratio: float = 0.7,
        rollback_min_gap: int = 50,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._num_keyframes = num_keyframes
        self._num_phases = num_phases
        self._threshold = threshold
        self._min_gap = min_gap
        self._task_id = task_id
        self._debug_save_dir = debug_save_dir
        self._argmax_gate = argmax_gate
        self._enable_rollback = enable_rollback
        self._rollback_window = rollback_window
        self._rollback_low_thr = rollback_low_thr
        self._rollback_high_thr = rollback_high_thr
        self._rollback_sustain_ratio = rollback_sustain_ratio
        self._rollback_min_gap = rollback_min_gap

        if ksm_device is None:
            ksm_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._ksm_device = ksm_device

        self._ksm_model: _TransformerKeyframeSelector | None = None
        if not disable_ksm and ksm_model_path and ksm_backbone_path:
            logger.info(
                "Loading KSM (backbone=%s, model=%s, device=%s, num_tasks=%d)",
                ksm_backbone_path, ksm_model_path, ksm_device, num_tasks,
            )
            ksm = _TransformerKeyframeSelector(
                pretrained_backbone_path=ksm_backbone_path,
                num_tasks=num_tasks,
                max_phases=20,
            ).to(ksm_device)
            state = torch.load(ksm_model_path, map_location=ksm_device)
            ksm.load_state_dict(state)
            ksm.eval()
            self._ksm_model = ksm
        else:
            logger.warning(
                "KSM disabled (disable_ksm=%s, model=%s, backbone=%s). "
                "Only the very first frame of each episode will be used as keyframe; "
                "remaining slots are zero-padded with mask=False.",
                disable_ksm, ksm_model_path, ksm_backbone_path,
            )

        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def warmup(self) -> None:
        """Run a dummy inference to trigger JAX JIT compilation before
        accepting real connections. This avoids the first request being
        extremely slow (10-30s) due to compilation."""
        logger.info("Running warmup inference (JIT compilation)...")
        dummy_obs: dict = {
            "head_rgb": np.zeros((360, 640, 3), dtype=np.uint8),
            "left_wrist_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
            "right_wrist_rgb": np.zeros((480, 640, 3), dtype=np.uint8),
            "state": np.zeros(23, dtype=np.float32),
            "prompt": "warmup",
        }
        # Simulate keyframe injection (same path as real requests).
        dummy_manager = _ServerKeyframeManager(
            ksm_model=None,  # skip KSM for warmup
            device=self._ksm_device,
            num_keyframes=self._num_keyframes,
        )
        augmented = dummy_manager.step(dummy_obs)
        t0 = time.monotonic()
        _ = self._policy.infer(augmented)
        elapsed = time.monotonic() - t0
        logger.info("Warmup inference done in %.1fs (JIT compiled)", elapsed)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    def _log_keyframe_inputs(self, augmented: dict, step_idx: int) -> None:
        """Log keyframe image sanity checks before handing data to pi05."""
        for i in range(self._num_keyframes):
            img = augmented.get(f"keyframe_{i}_rgb")
            mask = bool(augmented.get(f"keyframe_{i}_mask", False))
            if img is None:
                logger.info("[KSM->pi05] step=%d keyframe_%d mask=%s image=None", step_idx, i, mask)
                continue

            img_arr = np.asarray(img)
            nonzero_ratio = float(np.count_nonzero(img_arr)) / img_arr.size if img_arr.size else 0.0
            logger.info(
                "[KSM->pi05] step=%d keyframe_%d mask=%s shape=%s nonzero=%.4f",
                step_idx,
                i,
                mask,
                img_arr.shape,
                nonzero_ratio,
            )

    async def run(self) -> None:
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        """Concurrent per-connection handler.

        Three coroutines run in parallel:
        - `_recv_loop` reads frames off the WebSocket and dispatches to
          `policy_q` or `ksm_q` based on the message `type` field.
        - `_stream_worker` drains `ksm_q` and runs `update_from_frame` in
          a thread executor, never blocking the event loop.
        - `_policy_worker` drains `policy_q`, calls `inject_keyframes`,
          runs `policy.infer` in a thread executor, then sends back the
          action under `send_lock`.

        The KSM queue allows old frames to be dropped on overflow so the
        latest visual signal always reaches the model.
        """
        logger.info(f"Connection from {websocket.remote_address} opened — creating new KeyframeManager")
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        conn_id = f"{websocket.remote_address[0]}_{websocket.remote_address[1]}_{int(time.time())}"
        manager = _ServerKeyframeManager(
            ksm_model=self._ksm_model,
            device=self._ksm_device,
            num_keyframes=self._num_keyframes,
            num_phases=self._num_phases,
            threshold=self._threshold,
            min_gap=self._min_gap,
            task_id=self._task_id,
            debug_save_dir=self._debug_save_dir,
            argmax_gate=self._argmax_gate,
            enable_rollback=self._enable_rollback,
            rollback_window=self._rollback_window,
            rollback_low_thr=self._rollback_low_thr,
            rollback_high_thr=self._rollback_high_thr,
            rollback_sustain_ratio=self._rollback_sustain_ratio,
            rollback_min_gap=self._rollback_min_gap,
        )
        manager._connection_id = conn_id

        ksm_q: asyncio.Queue = asyncio.Queue(maxsize=64)
        policy_q: asyncio.Queue = asyncio.Queue(maxsize=4)
        send_lock = asyncio.Lock()
        loop = asyncio.get_running_loop()
        counters = {"recv": 0, "stream": 0, "policy": 0, "dropped": 0}
        prev_total_time: float | None = None

        async def _recv_loop() -> None:
            async for raw in websocket:
                counters["recv"] += 1
                try:
                    msg = msgpack_numpy.unpackb(raw)
                except Exception as exc:
                    logger.error("[Recv] failed to unpack message #%d: %s", counters["recv"], exc)
                    continue
                msg_type = msg.get("type", "policy_request") if isinstance(msg, dict) else "policy_request"

                if msg_type == "frame_update":
                    if ksm_q.full():
                        try:
                            ksm_q.get_nowait()
                            counters["dropped"] += 1
                        except asyncio.QueueEmpty:
                            pass
                    ksm_q.put_nowait(msg)
                else:
                    await policy_q.put(msg)

        async def _stream_worker() -> None:
            while True:
                msg = await ksm_q.get()
                head = msg.get("head_rgb") if isinstance(msg, dict) else None
                if head is None:
                    continue
                try:
                    await loop.run_in_executor(None, manager.update_from_frame, head)
                    counters["stream"] += 1
                    if counters["stream"] % 50 == 0:
                        logger.info(
                            "[KSM stream] processed=%d, dropped=%d, q=%d",
                            counters["stream"], counters["dropped"], ksm_q.qsize(),
                        )
                except Exception:
                    logger.exception("[KSM stream] update_from_frame failed")

        async def _policy_worker() -> None:
            nonlocal prev_total_time
            while True:
                msg = await policy_q.get()
                step_idx = counters["policy"]
                start_time = time.monotonic()
                try:
                    kf_start = time.monotonic()
                    augmented = manager.inject_keyframes(msg)
                    kf_time = time.monotonic() - kf_start
                    self._log_keyframe_inputs(augmented, step_idx)

                    infer_start = time.monotonic()
                    action = await loop.run_in_executor(None, self._policy.infer, augmented)
                    infer_time = time.monotonic() - infer_start

                    total_time = time.monotonic() - start_time
                    logger.info(
                        "[Timing] step=%d, total=%.0fms (ksm=%.0fms, infer=%.0fms)",
                        step_idx, total_time * 1000, kf_time * 1000, infer_time * 1000,
                    )

                    action["server_timing"] = {
                        "infer_ms": infer_time * 1000,
                        "keyframe_ms": kf_time * 1000,
                        "total_ms": total_time * 1000,
                    }
                    action["keyframe_stats"] = manager.stats
                    if prev_total_time is not None:
                        action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                    async with send_lock:
                        await websocket.send(packer.pack(action))
                    prev_total_time = time.monotonic() - start_time
                    counters["policy"] += 1
                except websockets.ConnectionClosed:
                    raise
                except Exception:
                    logger.exception("[Policy] inference failed at step %d", step_idx)
                    try:
                        async with send_lock:
                            await websocket.send(traceback.format_exc())
                            await websocket.close(
                                code=websockets.frames.CloseCode.INTERNAL_ERROR,
                                reason="Internal server error. Traceback included in previous frame.",
                            )
                    except Exception:
                        pass
                    raise

        recv_task = asyncio.create_task(_recv_loop(), name=f"recv-{conn_id}")
        stream_task = asyncio.create_task(_stream_worker(), name=f"stream-{conn_id}")
        policy_task = asyncio.create_task(_policy_worker(), name=f"policy-{conn_id}")
        tasks = [recv_task, stream_task, policy_task]

        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                exc = t.exception()
                if exc and not isinstance(exc, websockets.ConnectionClosed):
                    logger.error("[Handler] task %s raised: %s", t.get_name(), exc)
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            for t in tasks:
                try:
                    await t
                except (asyncio.CancelledError, websockets.ConnectionClosed):
                    pass
                except Exception:
                    logger.exception("[Handler] task cleanup error")
            try:
                manager.save_debug_summary()
            except Exception:
                logger.exception("[Handler] save_debug_summary failed")
            logger.info(
                "Connection from %s closed — recv=%d, stream=%d, policy=%d, dropped=%d",
                websocket.remote_address,
                counters["recv"], counters["stream"], counters["policy"], counters["dropped"],
            )


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
