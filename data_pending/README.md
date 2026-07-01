# contributions branch

Staging area for unverified candidate images the active-learning pipeline
hasn't promoted into `data/` yet. Never merged directly into `main` — only
reaches `main` via a PR that `weekly-retrain.yml`'s regression gate accepts.

- `manifest.jsonl` — one JSON object per candidate row (append-only). See
  `src/plantify/data.py`'s `build_pending_row`/`append_pending`/`read_pending`.
- `images/` — the actual uploaded files referenced by manifest rows, added
  as candidates arrive.
- `staged/` — GBIF-sourced files from `daily-gather.yml`, mirroring `data/`'s
  folder layout, merged into `data/` only if a retrain built on them passes
  the regression gate.

See `docs/ARCHITECTURE.md`'s "Active Learning & Self-Retraining" section for
the full design.
