# Deploying Plantify

Three independently deployable pieces. Only the first two matter for the
public demo — Streamlit is a maintainer-only tool.

| Piece | What it is | Where it goes |
|---|---|---|
| `web/` | Static showcase site | GitHub Pages (automatic) |
| `api/` | FastAPI inference service | Needs separate Python hosting |
| `app.py` (Streamlit) | Internal retrain/teach tool | Optional, not required for the public site |

---

## 1. `web/` → GitHub Pages

Already automated — see [`web/README.md`](web/README.md) for the full recipe.
Short version: push to `main`, the included `.github/workflows/deploy-pages.yml`
deploys `web/` automatically. One-time setup: repo Settings → Pages → Source →
**GitHub Actions**.

`web/js/config.js` needs to point at wherever you host `api/` (step 2) before
the live site's Identify feature will work. It's generated automatically at
deploy time from the `PLANTIFY_API_BASE` GitHub repository **variable**
(Settings → Secrets and variables → Actions → **Variables** tab — not
Secrets, that's a different namespace the deploy workflow doesn't read) —
see `.github/workflows/deploy-pages.yml` and "Wiring notes" in
[`web/README.md`](web/README.md).

## 2. `api/` → needs its own host (GitHub Pages is static-only)

**Azure Container Apps (what's actually deployed)** — chosen because its
Consumption plan scales to **zero replicas** when idle, so a low-traffic demo
costs nothing between visits, and it's covered by the GitHub Student Pack's
Azure credit with no card required. `.github/workflows/build-push-image.yml`
builds `Dockerfile.api` and pushes it to GHCR (free for a public repo) on
every relevant push; Azure just pulls that public image and runs it — no
Azure Container Registry needed, which has no free tier.

1. Make the built image's GHCR package public (one-time): GitHub profile →
   **Packages** → the package → **Package settings** → **Change visibility**.
2. Azure Portal → **Container Apps** → **Create**. Image source: **Docker Hub
   or other registries**, image `ghcr.io/<you>/<repo>:latest`, no registry
   credentials needed (it's public).
3. **Container** tab: **1 vCPU / 2 GiB** (TensorFlow's runtime overhead needs
   more than the 512MB that OOM-crashed this app on Render's free tier — see
   `docs/FINDINGS.md`).
4. **Ingress** tab: enabled, accepting traffic from anywhere, target port `8000`.
5. **Scale** tab: **min replicas 0, max replicas 1** — this is what keeps cost
   near zero. A ping-based "keep it warm" workflow would defeat this by never
   letting it idle down; this repo deliberately doesn't have one.
6. Add the env vars below (as **Secrets**, not plain env vars, for anything
   that's actually a credential), create.
7. Cost Management + Billing → **Budgets** → set an alert well under your
   credit as a backstop.

The tradeoff: the first request after a period of no traffic pays a cold
start (container boot + model load) before it responds — acceptable for a
portfolio demo, not for something expecting steady traffic.

**Alternative, if you don't have Azure/Student Pack access — DigitalOcean App
Platform**: no scale-to-zero, but no cold starts either; runs continuously for
~$10/mo (1 GiB / 1 vCPU tier — the 512MB tier OOMs the same as Render's free
tier does below). Connect this repo in the DigitalOcean dashboard, point it at
`Dockerfile.api`, deploy.

**Alternative — Render / Railway free tier**: same `Dockerfile.api`, no code
changes needed, but expect the same OOM crash on `/predict` this project hit
on Render's free tier unless the workload is lightened first (e.g. converting
to TensorFlow Lite for a smaller runtime footprint).

Whichever host you pick, set these env vars:
```bash
CORS_ALLOW_ORIGINS=https://<you>.github.io   # your GitHub Pages origin
MAX_UPLOAD_BYTES=8388608                      # optional, 8MB default
```

Once it's live, set the `PLANTIFY_API_BASE` repository variable (see above)
to the host's URL, then manually re-run `deploy-pages.yml` (it only
auto-triggers on `web/` changes, not on a variable update) so the live site
picks up the new API address.

## 3. `app.py` (Streamlit) — internal tool, optional

Not part of the public site. Used for interactive predictions and the
retrain/teach loop during development. Deploy it only if you want remote
access to that tooling:

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **New app** → pick your repo, branch `main`, main file `app.py`.
3. **Before clicking Deploy**, open **Advanced settings** and explicitly pick
   **Python 3.11** from the version dropdown. Community Cloud defaults new
   apps to whatever the latest Python is (3.14 as of this writing) —
   `tensorflow-cpu==2.15.0` has no wheel for anything newer than 3.11, so a
   default-version deploy fails immediately with a dependency resolution
   error. Multiple 2026 reports say Community Cloud currently ignores a
   committed `runtime.txt` for this, so the Advanced settings dropdown is
   the reliable way to pin it. **Python version can't be changed after
   deploying** — picking wrong means
   deleting the app and redeploying from scratch, not just retrying.
4. Deploy. It installs `requirements.txt` automatically.

Notes:
- `requirements.txt` already uses `tensorflow-cpu` and `opencv-python-headless`
  (server-friendly). No `packages.txt` is needed.
- The **Retrain** button works locally; on the free cloud tier the filesystem is
  ephemeral and retraining is heavy, so treat it as a local/dev feature.

### Optional: enable the Pl@ntNet "second opinion" (free, Streamlit tool only)

When the model doesn't recognise a leaf, the Streamlit tool can ask the free
**Pl@ntNet API** (500 identifications/day, no card). Off until you add a key:

1. Create a free account at <https://my.plantnet.org/> and copy your API key.
2. **Local:** set an environment variable before running —
   `set PLANTNET_API_KEY=your_key` (Windows) / `export PLANTNET_API_KEY=your_key`.
3. **Streamlit Cloud:** app → *Settings → Secrets* → add
   `PLANTNET_API_KEY = "your_key"`.

The key is read from the env var / secret only — never commit it.

### Recommended: protect the Retrain button with an admin token

The Streamlit tool's **Retrain** button runs training via a subprocess with no
authentication by default — anyone with access to the deployed app can trigger
it. Set an `ADMIN_TOKEN` to require a matching token before retraining runs:

1. **Local:** `set ADMIN_TOKEN=your_token` (Windows) / `export ADMIN_TOKEN=your_token`.
2. **Streamlit Cloud:** app → *Settings → Secrets* → add `ADMIN_TOKEN = "your_token"`.

If unset, the button stays functional but the UI shows an "unprotected"
warning — fine for local/dev use, but set this for any non-local deployment.

---

## Publishing a clean repo (one-time, if starting fresh)

A compact, compressed `data/` (~41 MB, all 15 classes, via
`scripts/compress_dataset.py`) is committed to the repo — this is required for
the weekly self-retraining workflow (see `docs/ARCHITECTURE.md`'s
"Active Learning & Self-Retraining" section) to have anything to train from.
If you're re-fetching the raw dataset locally, re-run the compression before
committing:

```bash
python scripts/compress_dataset.py      # ~2 GB raw -> ~41 MB compact, in place
git add -A
git commit -m "..."
git push
```

The trained model (`artifacts/model/`, ~1.6 MB) and `samples/` are already committed,
so the app runs immediately after cloning — no data download required.
