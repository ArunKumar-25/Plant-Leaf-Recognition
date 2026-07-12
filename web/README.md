# Plantify website

Static HTML/CSS/JS frontend. No build step, no framework — a single hand-written
stylesheet ([css/main.css](css/main.css)) and vanilla JS, no jQuery/Bootstrap/
plugin dependencies. Header and footer are loaded at runtime from
[`partials/`](partials/) via `fetch()` (see [js/partials.js](js/partials.js)),
so the site must be served over HTTP, not opened via `file://` (`fetch()` is
blocked by CORS on `file://` in most browsers). The only dynamic piece is
[identify.html](identify.html), which calls the FastAPI backend in
[`../api/`](../api/main.py).

## Run locally

In one terminal, start the API:

```bash
set CORS_ALLOW_ORIGINS=http://localhost:5500,http://127.0.0.1:5500
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

(run from the repo root, not from `web/`)

In another terminal, serve this folder:

```bash
cd web
python -m http.server 5500
```

Open `http://localhost:5500/index.html`.

## Wiring notes

Two things have to agree on the API's URL:

1. **API side** — `CORS_ALLOW_ORIGINS` env var (above) must include whatever
   origin you're serving `web/` from.
2. **Frontend side** — `identify.html` loads [js/config.js](js/config.js),
   which sets `window.PLANTIFY_API_BASE`. Locally it defaults to
   `http://localhost:8000`; in production it's generated at deploy time by
   [`../.github/workflows/deploy-pages.yml`](../.github/workflows/deploy-pages.yml)
   from the `PLANTIFY_API_BASE` repository variable — no manual file edits needed.

See [`../docs/INTEGRATION.md`](../docs/INTEGRATION.md) for the full API contract.

## Pages

Three pages, each earning its place — no decorative filler:

- `index.html` — landing page, pitches the ML capability
- `identify.html` — the actual feature: upload a leaf photo, get a live prediction
- `journal.html` — real technical content: how identification works, photo tips, the species list

(Earlier drafts of this site had `login.html`/`signup.html`/`contact.html` —
non-functional UI mockups inherited from the Bootstrap template. They were
cut: no auth backend exists anywhere in this stack, and a dead contact form
undercuts more than it adds. The real contact channel is the GitHub repo.)

`partials/header.html` and `partials/footer.html` are not standalone pages —
they're fetched and injected into every page by `js/partials.js`.

## Deploying

Deploys automatically to GitHub Pages on every push to `main` that touches
`web/**`, via [`../.github/workflows/deploy-pages.yml`](../.github/workflows/deploy-pages.yml)
(`actions/upload-pages-artifact` + `actions/deploy-pages`). This sidesteps
GitHub Pages' simple "deploy from branch, folder" UI, which only supports
`/` (repo root) or `/docs`, not arbitrary folder names like `/web`.

**One-time setup:** in the repo's Settings → Pages, set Source to
**GitHub Actions** (not "Deploy from a branch" — that option can't target a
`/web` subfolder, which is why this repo uses the Actions-based path).

Before relying on a production deploy: set the `PLANTIFY_API_BASE` repository
variable (see "Wiring notes" above) to your hosted API's URL, and add the
Pages URL (`https://<you>.github.io/<repo>/`) to the API's
`CORS_ALLOW_ORIGINS`.
