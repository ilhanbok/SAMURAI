# SAMURAI web portal

Browser front-end for SAMURAI. The Python back-end is the same pipeline the
command-line scripts run &mdash; SAM2 video propagation over a NIfTI slice stack
&mdash; with the matplotlib/Qt windows replaced by HTML templates.

## Running

From the repository root, with the SAMURAI environment activated:

```bash
pip install -r webapp/requirements-web.txt   # Flask, on top of the existing deps
./run_web.sh                                 # or .\run_web.ps1 on Windows
```

Then open <http://127.0.0.1:5000>.

`run_web.sh` is a wrapper around `python -m webapp.app`; anything that can serve
a WSGI app works too, e.g. `gunicorn 'webapp.app:app'` (use one worker &mdash;
session state, including the SAM2 inference state, lives in the worker process).

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `SAMURAI_HOST` / `SAMURAI_PORT` | `127.0.0.1` / `5000` | bind address |
| `SAMURAI_BACKEND` | `sam2` | `stub` swaps in the offline test backend |
| `SAMURAI_SAM2_CHECKPOINT` | `sam2/checkpoints/sam2.1_hiera_large.pt` | checkpoint path |
| `SAMURAI_SAM2_CONFIG` | `configs/sam2.1/sam2.1_hiera_l.yaml` | model config |
| `SAMURAI_WORKSPACE` | `webapp/workspaces` | per-session working directories |
| `SAMURAI_DATA_ROOTS` | repo root, its parent, cwd | directories the portal may read scans from |
| `SAMURAI_MAX_UPLOAD_MB` | `2048` | upload size limit |
| `SAMURAI_SESSION_TTL` | `21600` | seconds before an idle session is deleted |
| `SAMURAI_ITKSNAP_OFFSET` | `0` | `1` restores the desktop `slice - 1` prompt offset |
| `SAMURAI_DEBUG` | `0` | Flask debug mode |

`scripts/config.json` is still read for `threshold` (the initial mask threshold)
and `dwi` (4-D input default).

## Layout

```
webapp/
├── app.py              Flask routes - thin controllers only
├── core/
│   ├── volume.py       NIfTI IO, orientation, JPEG stack export
│   ├── render.py       slice / overlay PNG encoding (replaces imshow)
│   ├── segmenter.py    SAM2 prompting + propagation, and the stub backend
│   ├── threshold.py    RAW/QSM threshold lesion finder, lesion catalogue
│   ├── utilities.py    MERGE / ZERO / DUPLICATE / IRV / VOLUME
│   └── session.py      per-session workspace, prompts, results, timings
├── templates/          base, index, manual, threshold, utilities, docs
└── static/             css, js (api, viewer, manual, threshold, utilities)
```

Nothing under `core/` imports Flask, and only `segmenter.py` touches torch (and
only when a segmentation actually starts), so the portal loads and the
threshold/utility pages work even on a machine without SAM2 installed.

## The stub backend

`SAMURAI_BACKEND=stub` replaces SAM2 with a placeholder that grows an
intensity-similar blob around the foreground prompts. It exists so the portal
can be exercised without a GPU-sized install; it is **not** a segmentation model
and the UI shows a red banner while it is active. Never use its output.

## API

The pages are ordinary HTML; all state changes go through a small JSON API
(`/api/sessions/...`) documented by the route table in `app.py`. Long steps
&mdash; building the JPEG stack, propagation, the threshold pass, per-lesion
segmentation &mdash; run on a worker thread and are polled at
`/api/sessions/<id>/job`.
