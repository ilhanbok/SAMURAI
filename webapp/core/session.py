# Per-browser-session state for the SAMURAI web portal.
#
# The desktop scripts keep their state in module globals and a working
# directory full of temp files; a web front-end needs the same state scoped to
# one user's segmentation run, so each session gets its own workspace directory
# (the equivalent of `scripts/`) and its own SAM2 inference state.

import os
import shutil
import threading
import time
import uuid

import numpy as np

SESSION_TTL_SECONDS = int(os.environ.get("SAMURAI_SESSION_TTL", 6 * 60 * 60))


class Session:
    """One segmentation run: volume, prompts, SAM2 state, results, timings."""

    def __init__(self, session_id, workspace):
        self.id = session_id
        self.workspace = workspace
        self.created = time.time()
        self.touched = self.created
        self.lock = threading.RLock()

        # source volume
        self.volume = None
        self.source_name = None
        self.source_path = None

        # display window + slice cursor, mirroring MultiViewMaskEditor
        self.window = None            # [vmin, vmax]
        self.slices = [0, 0, 0]
        self.axis = None

        # SAM2 plumbing
        self.frames_dir = None          # display-orientation stack, mode (i)
        self.native_frames_dir = None   # native axial stack, mode (ii)
        self.frame_names = []
        self.backend_state = None
        self.objects = []             # [{obj_id, frame_idx, points, box}]
        self.next_obj_id = 1
        self.redo_stack = []
        self.preview = None           # {obj_id: logits2d} for the anchor frame

        # results
        self.result = None            # 3-D logits in display orientation
        self.mask_threshold = 0.0
        self.edits = {}               # display-space voxel index -> 0/1 override
        self.edit_version = 0         # bumped whenever edits/result change
        self._mask_cache = None
        self.times = []               # the five splits the desktop script records
        self.timer_start = time.time()
        self.last_point_activity = None
        self.saved_files = []

        # mode (ii)
        self.threshold_run = None
        self.catalog = None
        self.lesion_index = 0
        self.lesion_results = []
        self.lesion_union = None

        # background job
        self.job = {"status": "idle", "progress": 0.0, "message": "", "error": None}
        self.cancel_requested = False

    # -- helpers ----------------------------------------------------------
    def touch(self):
        self.touched = time.time()

    def path(self, *parts):
        return os.path.join(self.workspace, *parts)

    def split_time(self, label=None, until=None):
        """Record a timing split, like the desktop `times_taken` array."""
        now = until if until is not None else time.time()
        elapsed = max(0.0, now - self.timer_start)
        self.timer_start = now
        self.times.append((label, elapsed))
        return elapsed

    def split_point_and_confirm(self):
        """Split the pre-propagation time the way the desktop script does:
        point selection up to the last click, then the approve/redo decision."""
        boundary = self.last_point_activity
        if boundary is None or boundary < self.timer_start:
            self.split_time("points")
            self.split_time("confirm")
        else:
            self.split_time("points", until=boundary)
            self.split_time("confirm")

    def reset_prompts(self):
        self.objects = []
        self.next_obj_id = 1
        self.redo_stack = []
        self.preview = None

    def current_object(self):
        if not self.objects:
            return None
        return self.objects[-1]

    def register_file(self, path):
        name = os.path.basename(path)
        if name not in self.saved_files:
            self.saved_files.append(name)
        return name

    def invalidate_mask(self):
        """Call after changing the threshold, the edits or the result."""
        self.edit_version += 1
        self._mask_cache = None

    def binary_mask(self):
        """Threshold the propagated logits and apply any manual voxel edits."""
        if self.result is None:
            return None
        key = (self.mask_threshold, self.edit_version)
        if self._mask_cache is not None and self._mask_cache[0] == key:
            return self._mask_cache[1]
        mask = (self.result > self.mask_threshold).astype(np.uint8)
        for (i, j, k), value in self.edits.items():
            mask[i, j, k] = value
        self._mask_cache = (key, mask)
        return mask

    def summary(self):
        volume_info = None
        if self.volume is not None:
            volume_info = {
                "name": self.source_name,
                "display_shape": [int(s) for s in self.volume.display_shape],
                "native_shape": [int(s) for s in self.volume.native_shape],
                "zooms": [float(z) for z in self.volume.zooms],
                "vmin": self.volume.vmin,
                "vmax": self.volume.vmax,
            }
        return {
            "id": self.id,
            "volume": volume_info,
            "window": self.window,
            "slices": self.slices,
            "axis": self.axis,
            "objects": [
                {
                    "obj_id": o["obj_id"],
                    "frame_idx": o["frame_idx"],
                    "points": o["points"],
                    "box": o["box"],
                }
                for o in self.objects
            ],
            "has_result": self.result is not None,
            "mask_threshold": self.mask_threshold,
            "edits": len(self.edits),
            "files": self.saved_files,
            "job": self.job,
            "times": [{"label": lbl, "seconds": secs} for lbl, secs in self.times],
        }


class SessionStore:
    def __init__(self, root):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self._sessions = {}
        self._lock = threading.Lock()

    def create(self):
        session_id = uuid.uuid4().hex[:16]
        workspace = os.path.join(self.root, session_id)
        os.makedirs(workspace, exist_ok=True)
        session = Session(session_id, workspace)
        with self._lock:
            self._sessions[session_id] = session
        self.sweep()
        return session

    def get(self, session_id):
        with self._lock:
            session = self._sessions.get(session_id)
        if session is not None:
            session.touch()
        return session

    def drop(self, session_id, remove_files=True):
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None and remove_files:
            shutil.rmtree(session.workspace, ignore_errors=True)
        return session is not None

    def sweep(self):
        """Expire idle sessions so uploaded scans don't accumulate."""
        cutoff = time.time() - SESSION_TTL_SECONDS
        with self._lock:
            stale = [sid for sid, s in self._sessions.items() if s.touched < cutoff]
            for sid in stale:
                session = self._sessions.pop(sid, None)
                if session is not None:
                    shutil.rmtree(session.workspace, ignore_errors=True)
        return len(stale)

    def all(self):
        with self._lock:
            return list(self._sessions.values())
