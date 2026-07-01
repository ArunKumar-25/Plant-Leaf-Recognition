# Security Notes

This repository includes basic hardening for a public demo API and app.

## Implemented controls

- Upload validation in API and Streamlit:
  - allowed content types only
  - allowed extensions only
  - max upload size: 8 MB (configurable in API)
- Temporary files are written to OS temp paths and cleaned up after use.
- CORS is restricted by default to local development origins.
- Per-client rate limiting on `/predict` (`RATE_LIMIT_REQUESTS_PER_MINUTE`, default 30/min).
- Secrets are not hard-coded.
  - Pl@ntNet key is read from environment variable or Streamlit secrets.
  - `.streamlit/secrets.toml` is ignored by git.
- The active-learning staging path (see `docs/ARCHITECTURE.md`) is off by
  default and requires explicit opt-in (`PLANTNET_PUBLIC_FALLBACK_ENABLED`).

## Runtime configuration

- `CORS_ALLOW_ORIGINS`
  - comma-separated allowlist, for example:
  - `http://localhost:3000,https://your-site.github.io`
- `MAX_UPLOAD_BYTES`
  - API upload limit in bytes (default: 8 MB)
- `RATE_LIMIT_REQUESTS_PER_MINUTE`
  - per-client `/predict` rate limit (default: 30; `0` disables)
- `PLANTNET_PUBLIC_FALLBACK_ENABLED`
  - `true`/`1`/`yes` to enable the API's automatic Pl@ntNet consult on
    `unknown` predictions (default: off)
- `PLANTNET_DAILY_CAP`
  - max Pl@ntNet calls/day from the live API (default: 300), shared quota
    with the Streamlit tool's manual "Ask Pl@ntNet"
- `PLANTNET_STAGE_THRESHOLD`
  - min Pl@ntNet confidence to stage a candidate for the weekly retrain
    (default: `0.70`)
- `GITHUB_CONTRIB_TOKEN`
  - **fine-grained PAT, scoped to this one repo, `Contents: Read and write`
    only** — used to stage active-learning candidates via the GitHub
    Contents API. A write-capable token living on a public-facing host is a
    real trade-off: it's mitigated by (a) the narrow scope above, (b) it
    only ever targets the `contributions` branch, never `main` — so a leaked
    token can pollute a side branch, not touch model artifacts, trigger the
    Pages deploy, or reach `main` at all. Only the weekly GitHub Actions job
    (using its own auto-issued token, not this one) can ever write to `main`.
  - unset by default — the staging path silently no-ops without it.
- `GITHUB_REPO` / `GITHUB_BRANCH`
  - `owner/repo` and the staging branch name (default: `contributions`)
- `STAGING_MAX_RETRIES`
  - retry attempts on manifest write conflicts (default: 3)

## Operational recommendations

- Do not run model retraining from an internet-facing app unless protected by auth.
  The live API only ever *stages* candidate data (to a non-default branch);
  retraining itself runs exclusively in scheduled GitHub Actions, never in
  the request path.
- Put the API behind HTTPS in production.
- Keep dependencies updated and run CI tests on every pull request.
- Rotate third-party API keys and the `GITHUB_CONTRIB_TOKEN` PAT periodically,
  and immediately if exposure is suspected.

## Out of scope for this repository

- Authentication/authorization middleware
- WAF rules
- Multi-tenant isolation

These should be added at deployment layer depending on hosting platform.
