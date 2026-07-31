# SAMURAI web portal - Flask front-end.
#
# Every route here is a thin controller: the actual work lives in webapp/core,
# which is a headless port of the scripts/ pipeline. The three desktop entry
# points map onto three pages:
#
#   scripts/Adjunct_SAM_..._01232025.py (+ BoundingBox variants) -> /manual
#   scripts/thresholdSAM_01062025.py                             -> /threshold
#   scripts/Utilities.py                                         -> /utilities

import json
import os
import threading
import time
import traceback

import numpy as np
from flask import (
    Flask, abort, jsonify, render_template, request, send_from_directory, url_for,
)
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

from .core import render, segmenter, threshold as threshold_mod, utilities, volume as volume_mod
from .core.session import SessionStore

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WORKSPACE = os.path.join(REPO_ROOT, "webapp", "workspaces")
NIFTI_SUFFIXES = (".nii", ".nii.gz")

# Binarisation threshold the threshold-mode wrapper applies to SAM2 logits.
LESION_LOGIT_THRESHOLD = 2.0
# Sparsity index used to turn a candidate lesion into foreground prompts.
LESION_PROMPT_ETA = 0.2

TIME_LABELS = [
    ("slice", "Slice selection time"),
    ("points", "Point selection time"),
    ("confirm", "Mask confirmation time"),
    ("propagate", "Mask propagation time"),
    ("validate", "Output validation time"),
]


def load_pipeline_config():
    """Read scripts/config.json, the same knobs the desktop scripts use."""
    path = os.path.join(REPO_ROOT, "scripts", "config.json")
    defaults = {"threshold": 0, "color_markers": 0, "debug": 1, "dwi": 0}
    try:
        with open(path) as handle:
            defaults.update(json.load(handle))
    except (OSError, ValueError):
        pass
    return defaults


def create_app():
    app = Flask(__name__)
    app.config["REPO_ROOT"] = REPO_ROOT
    app.config["WORKSPACE_ROOT"] = os.environ.get("SAMURAI_WORKSPACE", DEFAULT_WORKSPACE)
    app.config["MAX_CONTENT_LENGTH"] = int(
        os.environ.get("SAMURAI_MAX_UPLOAD_MB", 2048)
    ) * 1024 * 1024
    app.config["PIPELINE_CONFIG"] = load_pipeline_config()
    # The desktop script prompts SAM2 at `slice - 1` to line up with ITK-SNAP's
    # 1-based numbering; the web viewer prompts the slice it is showing. Set
    # SAMURAI_ITKSNAP_OFFSET=1 to restore the old behaviour.
    app.config["ITKSNAP_OFFSET"] = os.environ.get("SAMURAI_ITKSNAP_OFFSET", "0") == "1"

    app.store = SessionStore(app.config["WORKSPACE_ROOT"])
    app.backend = segmenter.get_backend(REPO_ROOT)
    app.data_roots = _data_roots()

    _register_pages(app)
    _register_session_api(app)
    _register_manual_api(app)
    _register_threshold_api(app)
    _register_utilities_api(app)

    @app.errorhandler(413)
    def too_large(_error):
        limit = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        return jsonify({"error": "Upload exceeds the {0} MB limit.".format(limit)}), 413

    @app.errorhandler(HTTPException)
    def json_errors(error):
        """API calls get JSON errors; pages keep Flask's HTML ones."""
        if request.path.startswith("/api/"):
            return jsonify({"error": error.description, "status": error.code}), error.code
        return error

    @app.errorhandler(Exception)
    def unhandled(error):
        traceback.print_exc()
        if request.path.startswith("/api/"):
            return jsonify({"error": "{0}: {1}".format(type(error).__name__, error),
                            "status": 500}), 500
        raise error

    return app


def _data_roots():
    """Directories the portal may read scans from without an upload."""
    roots = [REPO_ROOT, os.path.abspath(os.path.join(REPO_ROOT, "..")), os.getcwd()]
    extra = os.environ.get("SAMURAI_DATA_ROOTS", "")
    roots.extend(p for p in extra.split(os.pathsep) if p)
    seen, out = set(), []
    for root in roots:
        real = os.path.realpath(root)
        if real not in seen and os.path.isdir(real):
            seen.add(real)
            out.append(real)
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _session_or_404(app, session_id):
    session = app.store.get(session_id)
    if session is None:
        abort(404, description="Unknown or expired session. Reload the page and start again.")
    return session


def _safe_server_path(app, raw_path):
    """Resolve a server-side scan path, confined to the configured roots."""
    candidate = os.path.realpath(os.path.expanduser(raw_path))
    for root in app.data_roots:
        if candidate == root or candidate.startswith(root + os.sep):
            if not os.path.isfile(candidate):
                abort(400, description="No such file: {0}".format(raw_path))
            return candidate
    abort(403, description="Path is outside the allowed data roots: {0}".format(raw_path))


def _float_arg(name, default=None):
    value = request.args.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _int_arg(name, default=0):
    value = request.args.get(name)
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def _png(data):
    from flask import Response

    response = Response(data, mimetype="image/png")
    response.headers["Cache-Control"] = "no-store"
    return response


def _start_job(session, kind, target, *args, **kwargs):
    """Run a long pipeline step on a worker thread; the page polls /job."""
    with session.lock:
        if session.job.get("status") == "running":
            abort(409, description="A {0} job is already running.".format(session.job.get("kind")))
        session.job = {
            "status": "running", "kind": kind, "progress": 0.0,
            "message": "Starting...", "error": None, "result": None,
        }
        session.cancel_requested = False

    def runner():
        try:
            result = target(*args, **kwargs)
            with session.lock:
                session.job.update(
                    {"status": "done", "progress": 1.0, "message": "Complete",
                     "result": result}
                )
        except Exception as exc:  # surfaced verbatim in the page's error panel
            traceback.print_exc()
            with session.lock:
                session.job.update(
                    {"status": "error", "message": str(exc), "error": str(exc)}
                )

    threading.Thread(target=runner, daemon=True, name="samurai-" + kind).start()
    return session.job


def _progress(session, fraction, message=None):
    with session.lock:
        session.job["progress"] = float(max(0.0, min(1.0, fraction)))
        if message:
            session.job["message"] = message


def parse_points_file(text):
    """Read a ``points_SAMURAI_*.txt`` file written by either front-end."""
    header = None
    points, labels = [], []
    slice_idx, axis = None, None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            body = stripped.lstrip("#").strip()
            if body.startswith("slice:"):
                slice_idx = int(float(body.split(":", 1)[1]))
            elif body.startswith("axis:"):
                axis = int(float(body.split(":", 1)[1]))
            continue
        if header is None and ":" not in stripped:
            parts = [p.strip() for p in stripped.split(",")]
            header = (int(float(parts[0])), int(float(parts[1])))
            continue
        head, rest = stripped.split(":", 1)
        coords = [c.strip() for c in rest.split(",")]
        points.append({"x": float(coords[0]), "y": float(coords[1]),
                       "label": int(float(head))})
        labels.append(int(float(head)))
    if header is not None and len(points) != sum(header):
        # same warning the desktop preload path prints
        pass
    return {"points": points, "slice": slice_idx, "axis": axis, "header": header}


def format_points_file(points, slice_idx, axis):
    lines = []
    n_fg = sum(1 for p in points if int(p["label"]) == 1)
    lines.append("{0}, {1}".format(n_fg, len(points) - n_fg))
    for point in points:
        lines.append("{0}: {1}, {2}".format(float(point["label"]), point["x"], point["y"]))
    lines.append("#slice: {0}".format(slice_idx))
    lines.append("#axis: {0}".format(axis))
    return "\n".join(lines) + "\n"


def format_times_file(times):
    by_label = {}
    for label, seconds in times:
        by_label[label] = by_label.get(label, 0.0) + seconds
    lines = [
        "{0}: {1}".format(title, by_label.get(key, 0.0)) for key, title in TIME_LABELS
    ]
    lines.append(" (in seconds)")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------

def _register_pages(app):
    @app.context_processor
    def inject_globals():
        return {
            "backend_status": app.backend.status(),
            "pipeline_config": app.config["PIPELINE_CONFIG"],
            "app_version": "1.1.0-web",
        }

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/manual")
    def manual():
        return render_template(
            "manual.html",
            colorbar=render.colorbar_stops(),
            default_dwi=bool(app.config["PIPELINE_CONFIG"].get("dwi", 0)),
            default_threshold=float(app.config["PIPELINE_CONFIG"].get("threshold", 0)),
        )

    @app.route("/threshold")
    def threshold_page():
        return render_template(
            "threshold.html",
            defaults=threshold_mod.MODE_DEFAULTS,
            volume_min=threshold_mod.VOLUME_THRESHOLD_MIN,
            volume_max=threshold_mod.VOLUME_THRESHOLD_MAX,
        )

    @app.route("/utilities")
    def utilities_page():
        return render_template("utilities.html")

    @app.route("/docs")
    def docs_page():
        return render_template("docs.html")

    @app.route("/api/status")
    def api_status():
        status = app.backend.status()
        status["data_roots"] = app.data_roots
        status["workspace"] = app.config["WORKSPACE_ROOT"]
        status["sessions"] = len(app.store.all())
        return jsonify(status)


# ---------------------------------------------------------------------------
# session lifecycle + slice rendering (shared by both modes)
# ---------------------------------------------------------------------------

def _register_session_api(app):
    @app.route("/api/sessions", methods=["POST"])
    def create_session():
        session = app.store.create()
        use_dwi = False
        source_path = None
        source_name = None

        upload = request.files.get("file")
        if upload is not None and upload.filename:
            use_dwi = request.form.get("dwi") in ("1", "true", "on")
            source_name = secure_filename(upload.filename)
            if not source_name.endswith(NIFTI_SUFFIXES):
                app.store.drop(session.id)
                abort(400, description="Expected a .nii or .nii.gz file.")
            source_path = session.path(source_name)
            upload.save(source_path)
        else:
            payload = request.get_json(silent=True) or request.form
            raw_path = (payload or {}).get("path")
            if not raw_path:
                app.store.drop(session.id)
                abort(400, description="Provide an uploaded file or a server path.")
            use_dwi = str((payload or {}).get("dwi", "")).lower() in ("1", "true", "on", "yes")
            source_path = _safe_server_path(app, raw_path)
            source_name = os.path.basename(source_path)

        try:
            vol = volume_mod.Volume(source_path, use_dwi=use_dwi)
        except Exception as exc:
            app.store.drop(session.id)
            abort(400, description="Could not read {0}: {1}".format(source_name, exc))

        with session.lock:
            session.volume = vol
            session.source_name = source_name
            session.source_path = source_path
            # MultiViewMaskEditor draws its panes at vmax/2; the sliders start
            # where the image does so the two agree.
            session.window = [vol.vmin, vol.vmax / 2.0]
            shape = vol.display_shape
            # MultiViewMaskEditor's opening cursor.
            session.slices = [min(32, shape[0] - 1), shape[1] // 2, min(43, shape[2] - 1)]
            session.mask_threshold = float(app.config["PIPELINE_CONFIG"].get("threshold", 0))
            session.timer_start = time.time()
        return jsonify(session.summary())

    @app.route("/api/sessions/<session_id>")
    def session_summary(session_id):
        return jsonify(_session_or_404(app, session_id).summary())

    @app.route("/api/sessions/<session_id>", methods=["DELETE"])
    def delete_session(session_id):
        return jsonify({"dropped": app.store.drop(session_id)})

    @app.route("/api/sessions/<session_id>/job")
    def job_status(session_id):
        session = _session_or_404(app, session_id)
        with session.lock:
            return jsonify(dict(session.job))

    @app.route("/api/sessions/<session_id>/job/cancel", methods=["POST"])
    def cancel_job(session_id):
        session = _session_or_404(app, session_id)
        session.cancel_requested = True
        return jsonify({"cancelling": True})

    @app.route("/api/sessions/<session_id>/slice.png")
    def slice_png(session_id):
        session = _session_or_404(app, session_id)
        if session.volume is None:
            abort(400, description="No volume loaded.")
        axis = _int_arg("axis", 0)
        index = _int_arg("index", 0)
        space = request.args.get("space", "display")
        if space == "native":
            data = session.volume.native
            index = int(np.clip(index, 0, data.shape[2] - 1))
            slice2d = data[:, :, index]
        else:
            slice2d = session.volume.slice(axis, index)
        vmin = _float_arg("vmin", session.window[0] if session.window else None)
        vmax = _float_arg("vmax", session.window[1] if session.window else None)
        return _png(render.grayscale_png(slice2d, vmin, vmax))

    @app.route("/api/sessions/<session_id>/mask.png")
    def mask_png(session_id):
        session = _session_or_404(app, session_id)
        if session.result is None:
            abort(404, description="No propagated mask yet.")
        axis = _int_arg("axis", 0)
        mask = session.binary_mask()
        index = int(np.clip(_int_arg("index", 0), 0, mask.shape[axis] - 1))
        if axis == 0:
            slice2d = mask[index, :, :]
        elif axis == 1:
            slice2d = mask[:, index, :]
        else:
            slice2d = mask[:, :, index]
        return _png(render.mask_overlay_png(slice2d, threshold=0.5))

    @app.route("/api/sessions/<session_id>/files")
    def session_files(session_id):
        session = _session_or_404(app, session_id)
        entries = []
        for name in sorted(os.listdir(session.workspace)):
            full = session.path(name)
            if os.path.isfile(full):
                entries.append(
                    {
                        "name": name,
                        "size": os.path.getsize(full),
                        "url": url_for("download_file", session_id=session.id, name=name),
                    }
                )
        return jsonify({"files": entries})

    @app.route("/api/sessions/<session_id>/files/<path:name>")
    def download_file(session_id, name):
        session = _session_or_404(app, session_id)
        return send_from_directory(session.workspace, name, as_attachment=True)

    @app.route("/api/browse")
    def browse():
        raw = request.args.get("dir") or app.data_roots[0]
        directory = os.path.realpath(os.path.expanduser(raw))
        if not any(directory == r or directory.startswith(r + os.sep) for r in app.data_roots):
            abort(403, description="Directory is outside the allowed data roots.")
        if not os.path.isdir(directory):
            abort(400, description="Not a directory: {0}".format(raw))
        dirs, files = [], []
        for entry in sorted(os.listdir(directory)):
            full = os.path.join(directory, entry)
            if os.path.isdir(full):
                dirs.append({"name": entry, "path": full})
            elif entry.endswith(NIFTI_SUFFIXES):
                files.append({"name": entry, "path": full,
                              "size": os.path.getsize(full)})
        parent = os.path.dirname(directory)
        return jsonify(
            {
                "dir": directory,
                "parent": parent if any(
                    parent == r or parent.startswith(r + os.sep) for r in app.data_roots
                ) else None,
                "dirs": dirs,
                "files": files,
            }
        )


# ---------------------------------------------------------------------------
# mode (i): manual point / bounding-box segmentation
# ---------------------------------------------------------------------------

def _register_manual_api(app):
    def _prepare_plane(session, axis, slice_index):
        """Write the JPEG stack for one plane and open a SAM2 inference state."""
        if session.backend_state is not None:
            # release the previous plane's SAM2 memory before opening another
            try:
                app.backend.reset(session.backend_state)
            except Exception:  # a stale state must never block a re-selection
                traceback.print_exc()
        _progress(session, 0.05, "Writing JPEG stack for the {0} plane".format(
            volume_mod.AXIS_NAMES[axis]))
        frames_dir = session.path("temp_jpeg_stack")
        frame_names = session.volume.write_jpeg_stack(axis, frames_dir)
        _progress(session, 0.6, "Loading SAM2 inference state ({0} frames)".format(
            len(frame_names)))
        state = app.backend.init_state(frames_dir)
        with session.lock:
            session.axis = int(axis)
            session.slices[axis] = int(slice_index)
            session.frames_dir = frames_dir
            session.frame_names = frame_names
            session.backend_state = state
            session.reset_prompts()
            session.result = None
            session.edits = {}
            session.invalidate_mask()
            session.split_time("slice")
        return {"axis": int(axis), "frames": len(frame_names), "slice": int(slice_index)}

    def _anchor_frame(session, slice_index):
        if app.config["ITKSNAP_OFFSET"]:
            return max(0, int(slice_index) - 1)
        return int(slice_index)

    def _apply_prompts(session, obj):
        """Push one object's points/box to the backend and cache the preview."""
        points = [(p["x"], p["y"]) for p in obj["points"]]
        labels = [int(p["label"]) for p in obj["points"]]
        if not points and obj["box"] is None:
            with session.lock:
                session.preview = None
            return None
        logits = app.backend.add_prompt(
            session.backend_state,
            frame_idx=obj["frame_idx"],
            obj_id=obj["obj_id"],
            points=points,
            labels=labels,
            box=obj["box"],
        )
        with session.lock:
            session.preview = logits
        return logits

    def _propagate(session):
        def progress(done, total):
            _progress(session, done / max(total, 1),
                      "Propagating through volume ({0}/{1} frames)".format(done, total))

        segments = app.backend.propagate(
            session.backend_state, progress=progress,
            should_stop=lambda: session.cancel_requested,
        )
        _progress(session, 0.98, "Assembling volume")
        result = segmenter.assemble_volume(
            segments, session.volume.display_shape, session.axis
        )
        with session.lock:
            session.result = result
            session.edits = {}
            session.invalidate_mask()
            session.split_time("propagate")
            lo, hi = float(np.min(result)), float(np.max(result))
            configured = float(app.config["PIPELINE_CONFIG"].get("threshold", 0))
            session.mask_threshold = configured if lo <= configured <= hi else 0.0
        return {
            "frames": len(segments),
            "logit_min": float(np.min(result)),
            "logit_max": float(np.max(result)),
            "threshold": session.mask_threshold,
            "cancelled": bool(session.cancel_requested),
        }

    @app.route("/api/sessions/<session_id>/window", methods=["POST"])
    def set_window(session_id):
        session = _session_or_404(app, session_id)
        payload = request.get_json(force=True)
        with session.lock:
            session.window = [float(payload["vmin"]), float(payload["vmax"])]
            if "slices" in payload:
                session.slices = [int(s) for s in payload["slices"]]
        return jsonify(session.summary())

    @app.route("/api/sessions/<session_id>/plane", methods=["POST"])
    def select_plane(session_id):
        session = _session_or_404(app, session_id)
        if session.volume is None:
            abort(400, description="No volume loaded.")
        payload = request.get_json(force=True)
        axis = int(payload["axis"])
        if axis not in (0, 1, 2):
            abort(400, description="axis must be 0, 1 or 2")
        slice_index = int(payload.get("slice", session.slices[axis]))
        slice_index = int(np.clip(slice_index, 0, session.volume.n_slices(axis) - 1))
        _start_job(session, "plane", _prepare_plane, session, axis, slice_index)
        return jsonify({"started": True})

    @app.route("/api/sessions/<session_id>/points", methods=["POST"])
    def set_points(session_id):
        session = _session_or_404(app, session_id)
        if session.backend_state is None:
            abort(400, description="Select a plane and slice first.")
        payload = request.get_json(force=True)
        points = payload.get("points", [])
        box = payload.get("box")
        frame = _anchor_frame(session, payload.get("slice", session.slices[session.axis]))

        with session.lock:
            obj = session.current_object()
            if obj is None:
                obj = {"obj_id": session.next_obj_id, "frame_idx": frame,
                       "points": [], "box": None}
                session.objects.append(obj)
                session.next_obj_id += 1
            obj["frame_idx"] = frame
            obj["points"] = [
                {"x": float(p["x"]), "y": float(p["y"]), "label": int(p["label"])}
                for p in points
            ]
            obj["box"] = [float(v) for v in box] if box else None
            session.last_point_activity = time.time()
            if payload.get("clear_redo", True):
                session.redo_stack = []

        try:
            logits = _apply_prompts(session, obj)
        except segmenter.BackendUnavailable as exc:
            abort(503, description=str(exc))
        return jsonify(
            {
                "object": obj,
                "preview": logits is not None,
                "objects": len(session.objects),
            }
        )

    @app.route("/api/sessions/<session_id>/points/undo", methods=["POST"])
    def undo_point(session_id):
        session = _session_or_404(app, session_id)
        with session.lock:
            obj = session.current_object()
            if not obj or not obj["points"]:
                return jsonify({"object": obj, "undone": False})
            session.redo_stack.append(obj["points"].pop())
        _apply_prompts(session, obj)
        return jsonify({"object": obj, "undone": True})

    @app.route("/api/sessions/<session_id>/points/redo", methods=["POST"])
    def redo_point(session_id):
        session = _session_or_404(app, session_id)
        with session.lock:
            obj = session.current_object()
            if not obj or not session.redo_stack:
                return jsonify({"object": obj, "redone": False})
            obj["points"].append(session.redo_stack.pop())
        _apply_prompts(session, obj)
        return jsonify({"object": obj, "redone": True})

    @app.route("/api/sessions/<session_id>/points/import", methods=["POST"])
    def import_points(session_id):
        session = _session_or_404(app, session_id)
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            abort(400, description="Attach a points_SAMURAI_*.txt file.")
        parsed = parse_points_file(upload.read().decode("utf-8", "replace"))
        parsed["session"] = session.id
        return jsonify(parsed)

    @app.route("/api/sessions/<session_id>/objects", methods=["POST"])
    def new_object(session_id):
        """Start another object on another slice - SAMURAI-BB multi-slice mode."""
        session = _session_or_404(app, session_id)
        if session.backend_state is None:
            abort(400, description="Select a plane and slice first.")
        payload = request.get_json(silent=True) or {}
        slice_index = int(payload.get("slice", session.slices[session.axis]))
        with session.lock:
            session.slices[session.axis] = slice_index
            obj = {
                "obj_id": session.next_obj_id,
                "frame_idx": _anchor_frame(session, slice_index),
                "points": [], "box": None,
            }
            session.objects.append(obj)
            session.next_obj_id += 1
            session.redo_stack = []
            session.preview = None
        return jsonify({"object": obj, "objects": len(session.objects)})

    @app.route("/api/sessions/<session_id>/preview.png")
    def preview_png(session_id):
        session = _session_or_404(app, session_id)
        if not session.preview:
            abort(404, description="No prediction yet.")
        obj_id = _int_arg("obj", 0)
        preview = session.preview
        logits = preview.get(obj_id) if obj_id else list(preview.values())[-1]
        if logits is None:
            logits = list(preview.values())[-1]
        png, lo, hi = render.logits_overlay_png(logits)
        response = _png(png)
        response.headers["X-Logit-Min"] = "{0:.4f}".format(lo)
        response.headers["X-Logit-Max"] = "{0:.4f}".format(hi)
        return response

    @app.route("/api/sessions/<session_id>/propagate", methods=["POST"])
    def propagate(session_id):
        session = _session_or_404(app, session_id)
        if session.backend_state is None or not session.objects:
            abort(400, description="Add at least one foreground point first.")
        session.split_point_and_confirm()
        _start_job(session, "propagate", _propagate, session)
        return jsonify({"started": True})

    @app.route("/api/sessions/<session_id>/mask/threshold", methods=["POST"])
    def set_mask_threshold(session_id):
        session = _session_or_404(app, session_id)
        if session.result is None:
            abort(400, description="Nothing propagated yet.")
        payload = request.get_json(force=True)
        with session.lock:
            session.mask_threshold = float(payload["threshold"])
            session.invalidate_mask()
        mask = session.binary_mask()
        return jsonify(
            {"threshold": session.mask_threshold, "voxels": int(np.count_nonzero(mask))}
        )

    @app.route("/api/sessions/<session_id>/mask/toggle", methods=["POST"])
    def toggle_voxel(session_id):
        """Click-to-toggle voxel editing, as in the desktop MaskEditor."""
        session = _session_or_404(app, session_id)
        if session.result is None:
            abort(400, description="Nothing propagated yet.")
        payload = request.get_json(force=True)
        axis = int(payload["axis"])
        index = int(payload["index"])
        x, y = int(round(payload["x"])), int(round(payload["y"]))
        if axis == 0:
            key = (index, y, x)
        elif axis == 1:
            key = (y, index, x)
        else:
            key = (y, x, index)
        shape = session.result.shape
        if not all(0 <= key[i] < shape[i] for i in range(3)):
            abort(400, description="Voxel outside the volume.")
        mask = session.binary_mask()
        with session.lock:
            session.edits[key] = int(1 - mask[key])
            value = session.edits[key]
            session.invalidate_mask()
        return jsonify({"voxel": list(key), "value": value, "edits": len(session.edits)})

    @app.route("/api/sessions/<session_id>/mask/reset-edits", methods=["POST"])
    def reset_edits(session_id):
        session = _session_or_404(app, session_id)
        with session.lock:
            session.edits = {}
            session.invalidate_mask()
        return jsonify({"edits": 0})

    @app.route("/api/sessions/<session_id>/save", methods=["POST"])
    def save_outputs(session_id):
        session = _session_or_404(app, session_id)
        if session.result is None:
            abort(400, description="Nothing to save yet.")
        session.split_time("validate")
        timestr = time.strftime("%Y%m%d-%H%M%S")
        mask = session.binary_mask()

        mask_path = session.path("output_SAMURAI_{0}.nii.gz".format(timestr))
        volume_mod.save_mask(mask, session.volume.affine, mask_path, threshold=0.5)
        session.register_file(mask_path)

        times_path = session.path("times_SAMURAI_{0}.txt".format(timestr))
        with open(times_path, "w") as handle:
            handle.write(format_times_file(session.times))
        session.register_file(times_path)

        written = [os.path.basename(mask_path), os.path.basename(times_path)]
        point_files = []
        for obj in session.objects:
            if not obj["points"]:
                continue
            suffix = timestr if len(session.objects) == 1 else "{0}_obj{1}".format(
                timestr, obj["obj_id"])
            points_path = session.path("points_SAMURAI_{0}.txt".format(suffix))
            with open(points_path, "w") as handle:
                handle.write(format_points_file(obj["points"], obj["frame_idx"], session.axis))
            session.register_file(points_path)
            point_files.append(os.path.basename(points_path))
        written.extend(point_files)
        if point_files:
            # the pointer file the desktop preload path reads
            with open(session.path("last_preloaded.txt"), "w") as handle:
                handle.write(point_files[-1])

        return jsonify(
            {
                "files": written,
                "voxels": int(np.count_nonzero(mask)),
                "times": [{"label": lbl, "seconds": secs} for lbl, secs in session.times],
                "timestamp": timestr,
            }
        )


# ---------------------------------------------------------------------------
# mode (ii): threshold-based lesion identification
# ---------------------------------------------------------------------------

def _register_threshold_api(app):
    def _run_threshold(session, params, volume_min, volume_max):
        def progress(message):
            _progress(session, min(0.9, session.job["progress"] + 0.12), message)

        run = threshold_mod.run_threshold_pipeline(
            session.source_path, session.workspace, progress=progress, **params
        )
        run["csv_name"] = params["output_csv"]
        catalog = threshold_mod.LesionCatalog(
            run["labeled_array"], run["rows"], axis=2, vmin=volume_min, vmax=volume_max
        )
        with session.lock:
            session.threshold_run = run
            session.catalog = catalog
            session.lesion_index = 0
            session.lesion_results = []
            session.lesion_union = np.zeros(session.volume.native_shape, dtype=bool)
            for path in run["outputs"].values():
                session.register_file(path)
        return {
            "num_features": run["num_features"],
            "candidates": len(catalog),
            "csv": os.path.basename(run["csv"]),
            "cutoff": run["cutoff"],
            "files": [os.path.basename(p) for p in run["outputs"].values()],
        }

    def _segment_lesion(session, entry, extra_points, logit_threshold):
        label_id = entry["label"]
        frames_dir = session.path("temp_jpeg_stack_native")
        if session.native_frames_dir is None or not os.path.isdir(frames_dir):
            _progress(session, 0.05, "Writing JPEG stack (axial, native orientation)")
            volume_mod.write_jpeg_stack(
                volume_mod.normalize_uint8(session.volume.native), 2, frames_dir
            )
            with session.lock:
                session.native_frames_dir = frames_dir

        _progress(session, 0.15, "Opening SAM2 state for lesion {0}".format(label_id))
        state = app.backend.init_state(frames_dir)

        frame, prompt_points, _all_points = session.catalog.prompt_points(
            label_id, eta=LESION_PROMPT_ETA
        )
        points = [{"x": float(x), "y": float(y), "label": 1} for x, y in prompt_points]
        points.extend(
            {"x": float(p["x"]), "y": float(p["y"]), "label": int(p["label"])}
            for p in extra_points or []
        )
        app.backend.add_prompt(
            state,
            frame_idx=frame,
            obj_id=1,
            points=[(p["x"], p["y"]) for p in points],
            labels=[p["label"] for p in points],
        )

        def progress(done, total):
            _progress(session, 0.2 + 0.7 * done / max(total, 1),
                      "Propagating lesion {0} ({1}/{2})".format(label_id, done, total))

        segments = app.backend.propagate(
            state, progress=progress, should_stop=lambda: session.cancel_requested
        )
        logits = segmenter.assemble_volume(segments, session.volume.native_shape, 2)
        mask = (logits > logit_threshold).astype(np.uint8)

        name = "{0}_240_lesionnum_{1}.nii.gz".format(
            session.threshold_run["csv_name"], label_id
        )
        path = session.path(name)
        volume_mod.save_native(mask, session.volume.affine, path)
        with session.lock:
            session.register_file(path)
            session.lesion_union |= mask.astype(bool)
            session.lesion_results.append(
                {"label": label_id, "file": name, "voxels": int(np.count_nonzero(mask)),
                 "volume": entry["volume"], "points": len(points)}
            )
        app.backend.reset(state)
        return {"label": label_id, "file": name, "voxels": int(np.count_nonzero(mask)),
                "prompt_points": len(points)}

    @app.route("/api/sessions/<session_id>/threshold/run", methods=["POST"])
    def threshold_run(session_id):
        session = _session_or_404(app, session_id)
        if session.volume is None:
            abort(400, description="No volume loaded.")
        payload = request.get_json(force=True)
        csv_name = secure_filename(payload.get("output_csv") or "SAMURAI") or "SAMURAI"
        params = {
            "output_csv": csv_name,
            "mode": (payload.get("mode") or "RAW").upper(),
            "cutoff": float(payload["cutoff"]) if payload.get("cutoff") not in (None, "") else None,
            "flip": bool(payload.get("flip")),
            "sigma": float(payload.get("sigma") or 0),
            "skull_strip": bool(payload.get("skullstrip")),
        }
        if payload.get("zero_slices_below") not in (None, ""):
            params["zero_slices_below"] = int(payload["zero_slices_below"])
        volume_min = float(payload.get("volume_min", threshold_mod.VOLUME_THRESHOLD_MIN))
        volume_max = float(payload.get("volume_max", threshold_mod.VOLUME_THRESHOLD_MAX))

        _start_job(session, "threshold", _run_threshold, session, params, volume_min, volume_max)
        return jsonify({"started": True})

    @app.route("/api/sessions/<session_id>/threshold/lesions")
    def threshold_lesions(session_id):
        session = _session_or_404(app, session_id)
        if session.catalog is None:
            abort(400, description="Run the threshold pass first.")
        done = {r["label"] for r in session.lesion_results}
        entries = []
        for position, entry in enumerate(session.catalog.entries):
            entries.append(
                {
                    "position": position,
                    "label": entry["label"],
                    "volume": entry["volume"],
                    "com": entry["com"],
                    "segmented": entry["label"] in done,
                }
            )
        return jsonify(
            {
                "lesions": entries,
                "results": session.lesion_results,
                "csv": os.path.basename(session.threshold_run["csv"]),
                "num_features": session.threshold_run["num_features"],
            }
        )

    @app.route("/api/sessions/<session_id>/threshold/lesions/<int:position>")
    def threshold_lesion(session_id, position):
        session = _session_or_404(app, session_id)
        if session.catalog is None or position >= len(session.catalog.entries):
            abort(404, description="No such candidate lesion.")
        entry = session.catalog.entries[position]
        frame, prompt_points, all_points = session.catalog.prompt_points(
            entry["label"], eta=LESION_PROMPT_ETA
        )
        return jsonify(
            {
                "position": position,
                "label": entry["label"],
                "volume": entry["volume"],
                "com": entry["com"],
                "frame": int(frame),
                "voxels_on_frame": int(len(all_points)),
                "scatter": all_points.tolist(),
                "prompt_points": prompt_points.tolist(),
                "slice_url": url_for(
                    "slice_png", session_id=session.id, space="native", index=int(frame)
                ),
            }
        )

    @app.route("/api/sessions/<session_id>/threshold/lesions/<int:position>/segment",
               methods=["POST"])
    def threshold_segment(session_id, position):
        session = _session_or_404(app, session_id)
        if session.catalog is None or position >= len(session.catalog.entries):
            abort(404, description="No such candidate lesion.")
        payload = request.get_json(silent=True) or {}
        entry = session.catalog.entries[position]
        logit_threshold = float(payload.get("logit_threshold", LESION_LOGIT_THRESHOLD))
        _start_job(
            session, "lesion", _segment_lesion, session, entry,
            payload.get("points", []), logit_threshold,
        )
        return jsonify({"started": True, "label": entry["label"]})

    @app.route("/api/sessions/<session_id>/threshold/merge", methods=["POST"])
    def threshold_merge(session_id):
        """Union of every approved lesion - the MERGE step of the SOP."""
        session = _session_or_404(app, session_id)
        if session.lesion_union is None or not session.lesion_results:
            abort(400, description="No approved lesions to merge.")
        timestr = time.strftime("%Y%m%d-%H%M%S")
        name = "{0}_merged_{1}.nii.gz".format(
            session.threshold_run.get("csv_name", "SAMURAI"), timestr)
        path = session.path(name)
        volume_mod.save_native(
            session.lesion_union.astype(np.uint8), session.volume.affine, path
        )
        session.register_file(path)
        return jsonify({"file": name,
                        "voxels": int(np.count_nonzero(session.lesion_union)),
                        "lesions": len(session.lesion_results)})


# ---------------------------------------------------------------------------
# Utilities.py operations
# ---------------------------------------------------------------------------

def _register_utilities_api(app):
    def _utility_workspace(name):
        session = app.store.create()
        directory = session.path(name)
        os.makedirs(directory, exist_ok=True)
        return session, directory

    def _save_uploads(files, directory):
        saved = []
        for upload in files:
            if not upload.filename:
                continue
            filename = secure_filename(upload.filename)
            if not filename.endswith(NIFTI_SUFFIXES):
                abort(400, description="{0} is not a NIfTI file.".format(upload.filename))
            path = os.path.join(directory, filename)
            upload.save(path)
            saved.append(path)
        if not saved:
            abort(400, description="No NIfTI files were uploaded.")
        return saved

    @app.route("/api/utilities/merge", methods=["POST"])
    def utility_merge():
        session, directory = _utility_workspace("inputs")
        _save_uploads(request.files.getlist("files"), directory)
        output_name = secure_filename(request.form.get("output") or "merged.nii.gz")
        result = utilities.merge_nifti(directory, session.path(output_name))
        session.register_file(session.path(output_name))
        result["download"] = url_for("download_file", session_id=session.id, name=output_name)
        result["session"] = session.id
        return jsonify(result)

    @app.route("/api/utilities/zero", methods=["POST"])
    def utility_zero():
        session, directory = _utility_workspace("inputs")
        paths = _save_uploads([request.files["file"]], directory)
        output_name = secure_filename(request.form.get("output") or "zeroed.nii.gz")
        result = utilities.zero_slices(
            paths[0], session.path(output_name),
            (int(request.form["start"]), int(request.form["finish"])),
            int(request.form.get("axis", 0)),
        )
        session.register_file(session.path(output_name))
        result["download"] = url_for("download_file", session_id=session.id, name=output_name)
        result["session"] = session.id
        return jsonify(result)

    @app.route("/api/utilities/duplicate", methods=["POST"])
    def utility_duplicate():
        session, directory = _utility_workspace("inputs")
        paths = _save_uploads([request.files["file"]], directory)
        output_name = secure_filename(request.form.get("output") or "duplicated.nii.gz")
        result = utilities.duplicate(
            paths[0], session.path(output_name),
            int(request.form["source"]), int(request.form["destination"]),
        )
        session.register_file(session.path(output_name))
        result["download"] = url_for("download_file", session_id=session.id, name=output_name)
        result["session"] = session.id
        return jsonify(result)

    @app.route("/api/utilities/volume", methods=["POST"])
    def utility_volume():
        session, directory = _utility_workspace("inputs")
        _save_uploads(request.files.getlist("files"), directory)
        output_name = secure_filename(request.form.get("output") or "volumes.csv")
        result = utilities.process_folder(directory, session.path(output_name))
        session.register_file(session.path(output_name))
        result["download"] = url_for("download_file", session_id=session.id, name=output_name)
        result["session"] = session.id
        return jsonify(result)

    @app.route("/api/utilities/irv", methods=["POST"])
    def utility_irv():
        session, dir_a = _utility_workspace("rater_a")
        dir_b = session.path("rater_b")
        os.makedirs(dir_b, exist_ok=True)
        _save_uploads(request.files.getlist("files_a"), dir_a)
        _save_uploads(request.files.getlist("files_b"), dir_b)
        result = utilities.irv_directories(dir_a, dir_b)
        result["session"] = session.id
        return jsonify(result)


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.environ.get("SAMURAI_HOST", "127.0.0.1"),
        port=int(os.environ.get("SAMURAI_PORT", 5000)),
        debug=os.environ.get("SAMURAI_DEBUG", "0") == "1",
        threaded=True,
    )
