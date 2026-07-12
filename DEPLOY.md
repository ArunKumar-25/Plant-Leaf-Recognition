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

**DigitalOcean App Platform (recommended if you have GitHub Student Pack
access)** — Render/Railway's free tiers spin down after ~15 min idle (first
request after a quiet period takes 60-90s), and their free/cheapest tier's
512MB RAM cap is not enough headroom for this app — confirmed live: `/predict`
OOM-crashes on Render's 512MB tier every time (TensorFlow's own runtime
overhead plus inference-time buffers routinely peaks at 600+MB). DigitalOcean
App Platform's **1 GiB** tier (~$10/mo, still covered for years by the
Student Pack's $200 credit) runs continuously with no spin-down and enough
memory, for the same "connect GitHub repo, auto-deploy" simplicity as Render:

1. Activate the DigitalOcean offer at <https://education.github.com/pack> if you haven't.
2. DigitalOcean dashboard → **Apps** → **Create App** → connect this GitHub repo, branch `main`.
3. It should detect `Dockerfile.api` automatically (or point it at that path explicitly).
4. Pick the **Shared (Fixed)** plan at **1 GiB / 1 vCPU** (~$10/mo) — not the
   512MB tier (same one DigitalOcean also offers at ~$5/mo), and not the
   free/dev tier (that one sleeps too).
5. Add the env vars below, deploy.

**Render / Railway (free tier)** — same `Dockerfile.api`, no code changes needed, but expect the same OOM crash on `/predict` this project hit on Render's free tier unless the workload is lightened first (e.g. converting to TensorFlow Lite for a smaller runtime footprint). Cold-starts are the lesser problem here.

Either way, set these env vars on the host:
```bash
CORS_ALLOW_ORIGINS=https://<you>.github.io   # your GitHub Pages origin
MAX_UPLOAD_BYTES=8388608                      # optional, 8MB default
```

Once it's live, set the `PLANTIFY_API_BASE` repository variable (see above)
to the host's URL and trigger the Pages deploy — no code changes needed.

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
   error. `runtime.txt` (`python-3.11`) is also committed in the repo root
   for this, but multiple 2026 reports say Community Cloud currently ignores
   `runtime.txt` — the Advanced settings dropdown is the reliable one.
   **Python version can't be changed after deploying** — picking wrong means
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
