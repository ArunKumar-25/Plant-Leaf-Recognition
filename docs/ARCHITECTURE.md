# Architecture

## Overview

Plantify is an ML project, organized so each trained capability is reachable
through the same three layers:

1. **Training pipeline** (`src/plantify/`) — builds the model and its
   evaluation artifacts.
2. **FastAPI service** (`api/`) — exposes the model over HTTP.
3. **`web/` showcase frontend** — calls the API and renders the result. It
   has no ML logic; it's a demo/integration surface, not the project.

There's also a Streamlit app (`app.py`) used internally for interactive
predictions and the retrain/teaching loop — a developer tool, not the public
surface.

Today there's one trained capability (leaf species identification). The
layering exists specifically so a second one doesn't require restructuring
anything — see [Adding a new ML capability](#adding-a-new-ml-capability).

## Runtime Components

- `src/plantify/streamlit_app.py`: UI, prediction funnel, OOD checks, retrain trigger.
- `src/plantify/training.py`: dataset split, feature extraction, head training, metrics export.
- `src/plantify/data.py`: class mapping and contribution persistence helpers.
- `src/plantify/plantnet_client.py`: optional Pl@ntNet second-opinion fallback.
- `api/main.py`: REST interface (`/health`, `/predict`) for external consumers.

## Data and Model Flow

1. Data is read from `data/<class-folder>/`.
2. Training script extracts MobileNetV2 embeddings and trains a classification head.
3. Outputs are saved to:
   - `leaf_cnn/` model
   - `class_labels.json`
   - `ood.npz`
   - `reports/` charts and metrics
4. Inference path loads model + labels + OOD references and returns prediction decision.

## Decision Funnel

1. Validate upload quality (`looks_like_leaf_scan`).
2. Run model prediction.
3. Compare embedding similarity against OOD threshold.
4. Return one of: `ok`, `uncertain`, `unknown`.
5. Optional: ask Pl@ntNet for external fallback.

## Adding a new ML capability

Identification (`/predict` → `identify.html`) is the reference pattern for
every layer a new ML feature needs:

1. **Train** — add the model/pipeline under `src/plantify/` (a new module,
   following `training.py`'s shape: data in, artifacts out under a clearly
   named directory like `leaf_cnn/`). Export whatever the API needs to load
   (model weights, label list, any thresholding data).
2. **Serve** — add an endpoint to `api/main.py` (e.g. `POST /predict-disease`)
   that loads the model once at startup (see `_load_once()`) and returns a
   JSON payload with a clear status/decision field, the same way `/predict`
   returns `decision: ok | uncertain | unknown`. Don't guess confidently on
   bad input — every capability should have some honesty mechanism, even a
   simple confidence floor.
3. **Show** — add a `web/<feature>.html` + `web/js/<feature>.js`, copying
   `identify.html`/`identify.js`'s structure: a `window.PLANTIFY_API_BASE`
   config line, a `fetch()` call, and result states that map 1:1 to whatever
   your endpoint's decision/status values are. Add it to the nav and to the
   "ML Capabilities" grid on `index.html`.
4. **Document** — add the new endpoint's contract to `docs/INTEGRATION.md`
   next to `/predict`'s.

Nothing about this requires touching the other capabilities — the API and
the web pages are one-to-one with each trained model.

## Compatibility Layer

`app.py` is retained at root as a wrapper around `src/plantify/streamlit_app.py` because Streamlit Community Cloud deploys expect a root-level main file (see `DEPLOY.md`). Training and utility code live solely in `src/plantify/` and `scripts/` — there are no other root-level wrapper files.
