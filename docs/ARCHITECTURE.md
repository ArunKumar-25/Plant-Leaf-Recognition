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
   `identify.html`/`identify.js`'s structure: it loads `js/config.js` for
   `window.PLANTIFY_API_BASE`, then a `fetch()` call, and result states that
   map 1:1 to whatever your endpoint's decision/status values are. Add it to
   the nav and to the "ML Capabilities" grid on `index.html`.
4. **Document** — add the new endpoint's contract to `docs/INTEGRATION.md`
   next to `/predict`'s.

Nothing about this requires touching the other capabilities — the API and
the web pages are one-to-one with each trained model.

## Active Learning & Self-Retraining

When `/predict` returns `unknown`, the API can optionally consult Pl@ntNet
and use a confident answer to grow the training set — fully automated, no
human in the loop for staging (though a regression gate still guards what
ever reaches the committed model). This is off by default; every piece below
requires explicit configuration to activate.

**Per-request (`api/main.py`):**
1. `unknown` only — not `uncertain`, which already has a plausible model
   guess. Gated by `PLANTNET_PUBLIC_FALLBACK_ENABLED` (default off).
2. Rate-limited to `PLANTNET_DAILY_CAP` (default 300) Pl@ntNet calls/day,
   shared quota with the Streamlit tool's manual "Ask Pl@ntNet".
3. If Pl@ntNet's top result scores `>= PLANTNET_STAGE_THRESHOLD` (default
   0.70), the image + a manifest row are staged via the GitHub Contents API
   to a dedicated **`contributions` branch** (never `main` — caps blast
   radius if the write-scoped token in `GITHUB_CONTRIB_TOKEN` ever leaks).
   Staging is best-effort and rolls back the image if the manifest write
   ultimately fails (`_stage_candidate`, with retry-on-conflict).

Nothing here writes to the API's local disk — an ephemeral host's filesystem
doesn't survive a restart, so GitHub itself is the only durable store.

**Staging format:** `data_pending/manifest.jsonl` (append-only, one JSON
object per line — avoids read-modify-write races on a parsed array) +
`data_pending/images/<uuid>.<ext>`. Row `status` moves
`pending → promoted_pending_gate → accepted_committed` (or back to `pending`
with a `reject_reason` if the gate later rejects the retrain) — never
deleted, so the full history of every candidate is auditable.
See `src/plantify/data.py`'s `build_pending_row`/`append_pending`/
`read_pending`/`write_pending`. This is a separate, *unverified* path from
`save_contribution` — the trusted, human-confirmed one used by Streamlit's
"Teach the model".

**Weekly (`.github/workflows/weekly-retrain.yml`, Mondays + manual dispatch):**
1. Evaluate the currently committed model (`scripts/evaluate_model.py`) → baseline.
2. Pull `data_pending/` from the `contributions` branch.
3. `scripts/promote_pending.py` — re-checks the score threshold, maps
   Pl@ntNet's scientific name onto an existing class by genus (never
   auto-creates a new species folder from an unverified external guess), caps
   promotions at `--max-per-class` per cycle, copies accepted images into
   `data/<species>/`.
4. If anything was promoted: retrain (`scripts/train_model.py`, unchanged),
   evaluate the result, and run the **regression gate**
   (`scripts/regression_gate.py`): accept only if aggregate accuracy is
   within tolerance of baseline *and* no single class's recall regressed
   (an aggregate-only check can hide one class collapsing while the average
   looks fine).
5. Accepted → commit `leaf_cnn/`, `class_labels.json`, `ood.npz`, `reports/`,
   `data/`, and the manifest to `main`, authored as
   `github-actions[bot]` — **never the human maintainer's identity**; an
   automated commit must say so honestly, not be styled to look like manual
   work. Rejected → no commit, an issue is opened recording why, and
   promoted rows revert to `pending` for the next cycle.

## Compatibility Layer

`app.py` is retained at root as a wrapper around `src/plantify/streamlit_app.py` because Streamlit Community Cloud deploys expect a root-level main file (see `DEPLOY.md`). Training and utility code live solely in `src/plantify/` and `scripts/` — there are no other root-level wrapper files.
