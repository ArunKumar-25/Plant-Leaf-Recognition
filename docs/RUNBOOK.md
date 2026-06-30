# Runbook

## 1) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Run Streamlit app

```bash
streamlit run app.py
```

## 3) Retrain model

```bash
python scripts/train_model.py
```

## 4) Run REST API (for website integration)

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API endpoints:

- `GET /health`
- `POST /predict` (multipart file upload)

## 5) Run tests

```bash
pytest -q
```

## 6) Optional Pl@ntNet key

Set locally:

```bash
set PLANTNET_API_KEY=your_key_here
```

For Streamlit Cloud, add it to app secrets.

## 7) API security env vars

```bash
set CORS_ALLOW_ORIGINS=http://localhost:3000,https://your-site.github.io
set MAX_UPLOAD_BYTES=8388608
```
