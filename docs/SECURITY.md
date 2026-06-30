# Security Notes

This repository includes basic hardening for a public demo API and app.

## Implemented controls

- Upload validation in API and Streamlit:
  - allowed content types only
  - allowed extensions only
  - max upload size: 8 MB (configurable in API)
- Temporary files are written to OS temp paths and cleaned up after use.
- CORS is restricted by default to local development origins.
- Secrets are not hard-coded.
  - Pl@ntNet key is read from environment variable or Streamlit secrets.
  - `.streamlit/secrets.toml` is ignored by git.

## Runtime configuration

- `CORS_ALLOW_ORIGINS`
  - comma-separated allowlist, for example:
  - `http://localhost:3000,https://your-site.github.io`
- `MAX_UPLOAD_BYTES`
  - API upload limit in bytes (default: 8 MB)

## Operational recommendations

- Do not run model retraining from an internet-facing app unless protected by auth.
- Put the API behind HTTPS in production.
- Keep dependencies updated and run CI tests on every pull request.
- Rotate third-party API keys if exposure is suspected.

## Out of scope for this repository

- Authentication/authorization middleware
- Rate limiting and WAF rules
- Multi-tenant isolation

These should be added at deployment layer depending on hosting platform.
