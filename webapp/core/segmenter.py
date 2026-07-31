# SAM2 prompting / propagation for the SAMURAI web portal.
#
# The desktop scripts drive SAM2's video predictor directly:
#
#     video_predictor = build_sam2_video_predictor(model_cfg, ckpt, device)
#     state = video_predictor.init_state(video_path=jpeg_stack_dir)
#     _, obj_ids, logits = video_predictor.add_new_points_or_box(...)
#     for idx, obj_ids, logits in video_predictor.propagate_in_video(state): ...
#     for idx, obj_ids, logits in video_predictor.propagate_in_video(state, reverse=True): ...
#
# This module wraps exactly those four calls behind a small interface so the
# Flask layer never touches torch, and so the pipeline can be exercised without
# a GPU-sized install (see StubBackend).

import os

import numpy as np

DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_CHECKPOINT = os.path.join("sam2", "checkpoints", "sam2.1_hiera_large.pt")


class BackendUnavailable(RuntimeError):
    """Raised when SAM2 (or its checkpoint) is not installed."""


class Sam2Backend:
    """Thin adapter over ``SAM2VideoPredictor``."""

    name = "sam2"
    is_real = True

    def __init__(self, repo_root, checkpoint=None, model_cfg=None, device=None):
        self.repo_root = str(repo_root)
        self.checkpoint = checkpoint or os.environ.get(
            "SAMURAI_SAM2_CHECKPOINT", os.path.join(self.repo_root, DEFAULT_CHECKPOINT)
        )
        self.model_cfg = model_cfg or os.environ.get("SAMURAI_SAM2_CONFIG", DEFAULT_MODEL_CFG)
        self._device_pref = device
        self._predictor = None
        self._device = None

    # -- availability -----------------------------------------------------
    def status(self):
        """Report what is and isn't in place, for the UI banner."""
        info = {
            "backend": self.name,
            "checkpoint": self.checkpoint,
            "checkpoint_present": os.path.isfile(self.checkpoint),
            "model_cfg": self.model_cfg,
            "torch": False,
            "sam2": False,
            "device": None,
            "loaded": self._predictor is not None,
        }
        try:
            import torch  # noqa: F401

            info["torch"] = True
            info["device"] = self._pick_device_name()
        except ImportError:
            pass
        try:
            import sam2  # noqa: F401

            info["sam2"] = True
        except ImportError:
            pass
        info["ready"] = info["torch"] and info["sam2"] and info["checkpoint_present"]
        return info

    def _pick_device_name(self):
        import torch

        if self._device_pref:
            return self._device_pref
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _ensure_predictor(self):
        if self._predictor is not None:
            return self._predictor
        try:
            import torch
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as exc:
            raise BackendUnavailable(
                "SAM2 is not importable ({0}). Run install_SAMURAI.sh / .ps1 in the "
                "repository root, or start the portal with SAMURAI_BACKEND=stub to "
                "use the offline test backend.".format(exc)
            )
        if not os.path.isfile(self.checkpoint):
            raise BackendUnavailable(
                "SAM2 checkpoint not found at {0}. Download it with "
                "sam2/checkpoints/download_ckpts.sh or set "
                "SAMURAI_SAM2_CHECKPOINT.".format(self.checkpoint)
            )
        self._device = torch.device(self._pick_device_name())
        self._predictor = build_sam2_video_predictor(
            self.model_cfg, self.checkpoint, device=self._device
        )
        return self._predictor

    # -- predictor lifecycle ---------------------------------------------
    def init_state(self, frames_dir):
        predictor = self._ensure_predictor()
        return {"inference_state": predictor.init_state(video_path=frames_dir)}

    def reset(self, state):
        if self._predictor is not None and state and state.get("inference_state") is not None:
            self._predictor.reset_state(state["inference_state"])

    def add_prompt(self, state, frame_idx, obj_id, points, labels, box=None):
        """Add point/box prompts for one object; returns {obj_id: logits2d}."""
        predictor = self._ensure_predictor()
        kwargs = dict(
            inference_state=state["inference_state"],
            frame_idx=int(frame_idx),
            obj_id=int(obj_id),
        )
        if points is not None and len(points):
            kwargs["points"] = np.asarray(points, dtype=np.float32)
            kwargs["labels"] = np.asarray(labels, dtype=np.int32)
        if box is not None:
            kwargs["box"] = np.asarray(box, dtype=np.float32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(**kwargs)
        return {
            int(oid): out_mask_logits[i][0].detach().cpu().numpy()
            for i, oid in enumerate(out_obj_ids)
        }

    def propagate(self, state, progress=None, should_stop=None):
        """Forward then reverse propagation, as the desktop scripts do."""
        predictor = self._ensure_predictor()
        inference_state = state["inference_state"]
        total = 2 * int(inference_state["num_frames"])
        done = 0
        segments = {}
        for reverse in (False, True):
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state, reverse=reverse
            ):
                segments[int(out_frame_idx)] = {
                    int(oid): out_mask_logits[i][0].detach().cpu().numpy()
                    for i, oid in enumerate(out_obj_ids)
                }
                done += 1
                if progress is not None:
                    progress(done, total)
                if should_stop is not None and should_stop():
                    return segments
        return segments


class StubBackend:
    """Offline stand-in for SAM2 used to exercise the portal without torch.

    It is NOT a segmentation model: it grows an intensity-similar blob around
    the foreground prompts and fades it across frames. Enable it deliberately
    with ``SAMURAI_BACKEND=stub``; the UI shows a warning banner while it is
    active so stub output is never mistaken for SAMURAI output.
    """

    name = "stub"
    is_real = False

    def __init__(self, repo_root=None, **_kwargs):
        self.repo_root = repo_root

    def status(self):
        return {
            "backend": self.name,
            "ready": True,
            "torch": False,
            "sam2": False,
            "checkpoint_present": False,
            "device": "cpu (stub)",
            "loaded": True,
            "warning": "Development stub backend - output is not SAM2 segmentation.",
        }

    def init_state(self, frames_dir):
        from PIL import Image

        names = sorted(n for n in os.listdir(frames_dir) if n.lower().endswith(".jpg"))
        frames = np.stack(
            [np.asarray(Image.open(os.path.join(frames_dir, n)).convert("L"), dtype=np.float32)
             for n in names]
        )
        return {"frames": frames, "prompts": {}}

    def reset(self, state):
        if state is not None:
            state["prompts"] = {}

    # -- pseudo-inference -------------------------------------------------
    @staticmethod
    def _logits_for_frame(frames, prompt, frame_idx):
        image = frames[frame_idx]
        h, w = image.shape
        yy, xx = np.mgrid[0:h, 0:w]
        fg = prompt["fg"]
        bg = prompt["bg"]
        ref = prompt["ref"]

        if len(fg):
            d_fg = np.min(
                np.stack([np.hypot(xx - px, yy - py) for px, py in fg]), axis=0
            )
        else:
            d_fg = np.full((h, w), 1e3)

        logits = 6.0 - 0.25 * d_fg - 0.08 * np.abs(image - ref)
        if len(bg):
            d_bg = np.min(
                np.stack([np.hypot(xx - px, yy - py) for px, py in bg]), axis=0
            )
            logits -= 8.0 * np.exp(-(d_bg ** 2) / (2 * 12.0 ** 2))
        box = prompt.get("box")
        if box is not None:
            x0, y0, x1, y1 = box
            outside = (xx < min(x0, x1)) | (xx > max(x0, x1)) | (yy < min(y0, y1)) | (yy > max(y0, y1))
            logits[outside] -= 25.0
        logits -= 0.6 * abs(frame_idx - prompt["frame_idx"])
        return logits.astype(np.float32)

    def add_prompt(self, state, frame_idx, obj_id, points, labels, box=None):
        frames = state["frames"]
        frame_idx = int(np.clip(int(frame_idx), 0, len(frames) - 1))
        points = np.asarray(points, dtype=np.float32) if points is not None else np.zeros((0, 2))
        labels = np.asarray(labels, dtype=np.int32) if labels is not None else np.zeros((0,))
        fg = [tuple(p) for p, l in zip(points, labels) if l == 1]
        bg = [tuple(p) for p, l in zip(points, labels) if l == 0]

        image = frames[frame_idx]
        samples = []
        for px, py in fg:
            x, y = int(round(px)), int(round(py))
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                samples.append(image[max(0, y - 1):y + 2, max(0, x - 1):x + 2].mean())
        ref = float(np.mean(samples)) if samples else float(image.mean())

        state["prompts"][int(obj_id)] = {
            "frame_idx": frame_idx, "fg": fg, "bg": bg, "ref": ref,
            "box": tuple(box) if box is not None else None,
        }
        return {
            oid: self._logits_for_frame(frames, prompt, frame_idx)
            for oid, prompt in state["prompts"].items()
        }

    def propagate(self, state, progress=None, should_stop=None):
        frames = state["frames"]
        prompts = state["prompts"]
        if not prompts:
            raise RuntimeError("No prompts have been added yet.")
        total = len(frames)
        segments = {}
        for i in range(total):
            segments[i] = {
                oid: self._logits_for_frame(frames, prompt, i) for oid, prompt in prompts.items()
            }
            if progress is not None:
                progress(i + 1, total)
            if should_stop is not None and should_stop():
                break
        return segments


def get_backend(repo_root):
    """Pick a backend from ``SAMURAI_BACKEND`` (``sam2`` by default)."""
    choice = os.environ.get("SAMURAI_BACKEND", "sam2").strip().lower()
    if choice == "stub":
        return StubBackend(repo_root)
    return Sam2Backend(repo_root)


def assemble_volume(segments, shape, axis, obj_ids=None):
    """Fold per-frame SAM2 logits back into a 3-D volume.

    Same slot assignment as the desktop scripts: frame index indexes ``axis``,
    and where several objects exist their logits are combined with a maximum
    (the multi-slice SAMURAI-BB script unions its objects).
    """
    out = np.full(shape, -1e4, dtype=np.float32)
    for frame_idx, per_obj in segments.items():
        if not per_obj:
            continue
        chosen = [v for k, v in per_obj.items() if obj_ids is None or k in obj_ids]
        if not chosen:
            continue
        merged = chosen[0] if len(chosen) == 1 else np.maximum.reduce(chosen)
        if axis == 0:
            out[frame_idx, :, :] = merged
        elif axis == 1:
            out[:, frame_idx, :] = merged
        else:
            out[:, :, frame_idx] = merged
    return out
