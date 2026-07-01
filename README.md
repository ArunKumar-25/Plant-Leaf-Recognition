# Plantify

Plantify is a **machine learning project** for plant leaf analysis. The first
trained capability is **species identification**: a CNN classifier that takes
a leaf photo and returns the species with a confidence score, with an honest
"I'm not sure" (via an out-of-distribution guard) instead of a confident wrong
guess.

The `web/` site is a thin showcase layer, not the project itself — it exists
to demo each ML capability and to give new ones a place to plug in. Adding a
new model later (disease detection, growth-stage estimation, whatever's
trained next) means adding one API endpoint and one page, following the same
pattern Identify already uses — see [Adding a new ML capability](docs/ARCHITECTURE.md#adding-a-new-ml-capability).

**[Try Identify locally →](web/identify.html)** (see [Quickstart](#quickstart) to run it)

## How it's built

- **`src/plantify/`** — the ML core: a MobileNetV2-based classifier trained on
  the 15-species Swedish Leaf Dataset, an out-of-distribution (OOD) guard so
  the model says "unknown" instead of guessing on bad input, and a Streamlit
  app (`app.py`) used internally to retrain/teach the model. This is where a
  new ML capability's training code would live.
- **`api/`** — a FastAPI service (`POST /predict`, `GET /health`) that exposes
  the ML core over HTTP. A new capability gets its own endpoint here.
- **`web/`** — the showcase/demo frontend (static HTML/CSS/JS, no build step):
  landing page, the Identify feature, a blog explaining how identification
  works. Calls `api/` over CORS — it has no ML logic of its own, just UI for
  whatever the API exposes.

**Public-facing vs. internal:** `web/` (GitHub Pages) and `api/` (separately
hosted) together are the deployed product. `src/plantify/`'s Streamlit app
is a maintainer-only tool for retraining/teaching the model — it is not
part of the public site and does not need to be deployed for the demo to work.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the prediction
decision funnel works (and how to add a new ML capability), and
[docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for the full repo layout.

## Quickstart

### 1) Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Run the API

```bash
set CORS_ALLOW_ORIGINS=http://localhost:5500,http://127.0.0.1:5500
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

- Health check: `GET /health`
- Prediction: `POST /predict` (multipart file)

### 3) Run the website

```bash
cd web
python -m http.server 5500
```

Open `http://localhost:5500/index.html`, then try **Identify**. See
[web/README.md](web/README.md) if the two don't talk to each other.

### 4) (Internal) Run the Streamlit admin/retrain tool

```bash
streamlit run app.py
```

### 5) Retrain the model

```bash
python scripts/train_model.py
```

## Deploying to GitHub Pages

`web/` auto-deploys to GitHub Pages on every push to `main` that touches
`web/**`, via [`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml).
One-time setup: repo Settings → Pages → Source → **GitHub Actions** (not
"Deploy from a branch" — that option can't target a `/web` subfolder).

The API (`api/`) is not part of this workflow — GitHub Pages only serves
static files, so it needs separate hosting. See [DEPLOY.md](DEPLOY.md) for
the full breakdown of what goes where.

## Core ML capabilities

- 15-class Swedish Leaf Dataset classifier, 98.7% held-out test accuracy
  (see [artifacts/reports/metrics.md](artifacts/reports/metrics.md))
- OOD thresholding using embedding similarity (`artifacts/ood.npz`) — rejects
  out-of-domain photos instead of guessing
- Optional Pl@ntNet second-opinion fallback

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/INTEGRATION.md](docs/INTEGRATION.md) — API contract for `web/` or any other frontend
- [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)
- [docs/RUNBOOK.md](docs/RUNBOOK.md)
- [docs/SECURITY.md](docs/SECURITY.md)
- [docs/FINDINGS.md](docs/FINDINGS.md) — project audit log (what was broken, what was fixed)
- [DEPLOY.md](DEPLOY.md)

## Deployment artifacts included

- `.github/workflows/deploy-pages.yml` — auto-deploys `web/` to GitHub Pages
- `Dockerfile.api` for container deployment of `api/`
- `.github/workflows/ci.yml` for test automation

## Student-friendly deployment path

To keep costs near zero:

- Frontend (`web/`): GitHub Pages, already automated (see above).
- API/Backend (`api/`): Render/Railway free tier, or DigitalOcean
  student credits, using the included `Dockerfile.api`.
- ML admin app (`app.py`, optional): Streamlit Community Cloud.
- Optional AWS learning without spend: LocalStack (via GitHub Student Pack).

## License

MIT
