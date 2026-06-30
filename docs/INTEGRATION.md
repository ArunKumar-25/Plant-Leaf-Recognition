# Phase 2: Website Integration

This document explains how the Plantify website should consume the ML API.
A working reference implementation lives at [`../web/identify.html`](../web/identify.html)
and [`../web/js/identify.js`](../web/js/identify.js) — see [`../web/README.md`](../web/README.md)
for the local-dev CORS/API-base wiring recipe.

## API Base URL

- Local: `http://localhost:8000`
- Production: your deployed API URL

## Endpoints

### `GET /health`

Returns model/service status.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "num_classes": 15,
  "ood_enabled": true
}
```

### `POST /predict`

Upload an image file as multipart form-data.

Form field:

- `file`: image binary (`jpg`, `png`, etc.)

Constraints:

- max file size: 8 MB (default)
- accepted types: jpg, png, bmp, tiff, webp

Example response:

```json
{
  "species": "Acer",
  "confidence": 0.982,
  "top_k": [
    { "species": "Acer", "confidence": 0.982 },
    { "species": "Ulmus carpinifolia", "confidence": 0.013 },
    { "species": "Quercus", "confidence": 0.005 }
  ],
  "decision": "ok",
  "quality_ok": true,
  "domain_similarity": 0.91,
  "num_classes": 15
}
```

`decision` meanings:

- `ok`: confident in known species
- `uncertain`: weak match, show warning in UI
- `unknown`: likely outside trained domain/species

## Frontend integration snippet

```javascript
async function predictLeaf(file) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch("http://localhost:8000/predict", {
    method: "POST",
    body: form,
  });

  if (!res.ok) throw new Error("Prediction failed");
  return await res.json();
}
```

UI mapping recommendation:

- if `decision === "ok"`: show success card
- if `decision === "uncertain"`: show amber warning + top_k options
- if `decision === "unknown"`: show fallback message + ask user to teach species

## CORS configuration

Set `CORS_ALLOW_ORIGINS` to your frontend origins:

```bash
set CORS_ALLOW_ORIGINS=http://localhost:3000,https://your-site.github.io
```

Avoid using `*` in production. Use explicit origins only.

For local dev against `web/` served via `python -m http.server 5500`, use
`CORS_ALLOW_ORIGINS=http://localhost:5500,http://127.0.0.1:5500` — see
`web/README.md` for the full recipe.
