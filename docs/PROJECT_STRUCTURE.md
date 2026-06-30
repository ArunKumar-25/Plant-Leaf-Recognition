# Professional Project Structure

```text
Plant-Leaf-Recognition/
├── src/
│   └── plantify/                 # THE PROJECT: ML training code and trained-capability logic
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── plantnet_client.py
│       ├── streamlit_app.py      # Internal admin tool: predictions + retrain/teach loop
│       └── training.py
├── api/
│   └── main.py                   # FastAPI inference service (/health, /predict) — exposes src/ over HTTP
├── web/                           # Showcase/demo frontend for the ML capabilities — static HTML/CSS/JS,
│   │                                no build step, no ML logic of its own, calls api/ only
│   ├── index.html                  # Landing page, ML capabilities grid
│   ├── identify.html               # Leaf identification capability (calls api/ via fetch)
│   ├── blog.html                   # How identification works, photo tips, species list
│   ├── partials/                   # header.html / footer.html, fetched + injected by js/partials.js
│   ├── css/ js/ img/ fonts/ style.css
│   └── README.md                   # How to serve web/ locally + CORS/API_BASE wiring + GitHub Pages deploy
├── docs/
│   ├── ARCHITECTURE.md           # Includes "Adding a new ML capability" — the extension pattern
│   ├── PROJECT_STRUCTURE.md
│   ├── RUNBOOK.md
│   ├── SECURITY.md
│   ├── INTEGRATION.md            # API contract for any frontend (web/ is the reference implementation)
│   └── FINDINGS.md               # Project audit log (what was broken, what was fixed)
├── scripts/
│   ├── run_app.py                # Script entrypoint for Streamlit
│   ├── train_model.py            # Script entrypoint for training
│   ├── compress_dataset.py       # Shrink raw .tif scans to committable .jpg
│   └── fetch_full_dataset.py     # Download the full 15-species dataset
├── tests/
│   ├── test_api_security.py
│   └── test_data_helpers.py
├── samples/                      # Sample leaf images for manual testing
├── data/                         # Dataset classes (local / optional in git)
├── leaf_cnn/                     # Trained model artifacts
├── reports/                      # Metrics + plots
├── app.py                        # Compatibility wrapper (Streamlit) — required by Streamlit
│                                  #   Community Cloud, which expects a root-level main file
├── class_labels.json
├── ood.npz
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile.api
├── Procfile
├── DEPLOY.md
└── README.md
```

## Why this structure

- `src/` is the project itself — the ML training code and the logic behind each
  trained capability. Everything else exists to expose or demo what's in here.
- `api/` cleanly exposes model inference from `src/` to `web/` or any other client,
  one endpoint per capability.
- `web/` is a showcase/demo frontend, not the product — it has no ML logic, just UI
  that calls `api/` over CORS. Deployed independently (e.g. GitHub Pages); adding a
  new ML capability means adding one page here, not restructuring it.
- `scripts/` gives explicit operational entrypoints (`scripts/train_model.py` for retraining).
- `docs/` centralizes architecture (including how to add a new ML capability), run
  instructions, and the API integration contract.
- `app.py` is the one intentional root-level wrapper, kept only because Streamlit Community
  Cloud requires a root main file — everything else lives under `src/`, `api/`, or `web/`.
