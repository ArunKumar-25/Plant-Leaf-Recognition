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

Before running the model, the upload is scored by `_leaf_scan_quality` — the
fraction of near-white background pixels, bucketed into three bands:

- **reject** (doesn't look like an attempted leaf photo at all — undecodable,
  or the background is nowhere close to plain): the model never runs;
  responds `422` with `{"detail": "This doesn't look like a leaf photo. ..."}`.
- **warn** (leaf-like but off-format — busier background/lighting than the
  training domain): the model runs normally, and the response includes an
  extra `quality_warning` string the UI should surface.
- **ok** (matches the training domain): the model runs normally, no warning.

Example response (`ok` quality, `ok` decision):

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
  "quality": "ok",
  "domain_similarity": 0.91,
  "num_classes": 15
}
```

`quality: "warn"` responses add a `quality_warning` string field alongside
the fields above — see the frontend integration snippet's UI mapping.

`decision` meanings:

- `ok`: confident in known species
- `uncertain`: weak match, show warning in UI
- `unknown`: likely outside trained domain/species

Both `uncertain` and `unknown` responses may include a `plantnet` field
(`{name, common, score, staged}`, second-opinion suggestion) if Pl@ntNet was
consulted. `uncertain` is deliberately included, not just `unknown` — the
model's raw softmax confidence can be high even when the OOD guard flags the
photo as a weak domain match, so `uncertain` is exactly the band where a
second opinion is most useful. `staged` is `true` only if `score` cleared
`PLANTNET_STAGE_THRESHOLD` (default 0.70) — below that, Pl@ntNet's guess is
shown but was never queued for review; say so in the UI rather than implying
it might have been. See `docs/ARCHITECTURE.md`'s "Active Learning &
Self-Retraining" section.

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

- if the request fails with `422`: show the `detail` message directly —
  the upload didn't look like an attempted leaf photo, the model never ran
- if `quality === "warn"`: show the `quality_warning` string as a caveat,
  regardless of which `decision` card is also shown
- if `decision === "ok"`: show success card
- if `decision === "uncertain"`: show amber warning + top_k options; if a
  `plantnet` field is present, show its suggestion too
- if `decision === "unknown"`: show fallback message + ask user to teach
  species; if a `plantnet` field is present, show its suggestion too

## CORS configuration

Set `CORS_ALLOW_ORIGINS` to your frontend origins:

```bash
set CORS_ALLOW_ORIGINS=http://localhost:3000,https://your-site.github.io
```

Avoid using `*` in production. Use explicit origins only.

For local dev against `web/` served via `python -m http.server 5500`, use
`CORS_ALLOW_ORIGINS=http://localhost:5500,http://127.0.0.1:5500` — see
`web/README.md` for the full recipe.
