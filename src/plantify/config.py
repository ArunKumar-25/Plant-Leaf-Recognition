"""Central project configuration constants."""

import os

IMG_SIZE = 128
ARTIFACTS_DIR = "artifacts"
MODEL_DIR = "artifacts/model"
LABELS_FILE = "artifacts/class_labels.json"
DATA_DIR = "data"
REPORTS_DIR = "artifacts/reports"
OOD_FILE = "artifacts/ood.npz"
CONFIDENCE_FLOOR = 0.40

# Upload quality gate (api/main.py's _leaf_scan_quality), by near-white
# background fraction: outside REJECT_MIN/MAX -> reject, inside MIN/MAX_WHITE_BG
# -> ok, else -> warn (predict anyway, flag as less reliable).
MIN_WHITE_BG = 0.40
MAX_WHITE_BG = 0.97
REJECT_MIN_WHITE_BG = 0.15
REJECT_MAX_WHITE_BG = 0.995

# Self-sourced reinforcement of existing classes on quiet days (no real
# pending data) -- see promote_pending.py / fetch_species_dataset.py.
REINFORCEMENT_FETCH_COUNT = 5

# New-species growth: maintainer candidate list or repeated Pl@ntNet
# agreement from visitors, both gated and never auto-merged (docs/ARCHITECTURE.md).
NEW_SPECIES_TRIGGER_MIN_SIGNALS = 3
NEW_SPECIES_TRIGGER_MIN_DIVERSITY_DAYS = 2
NEW_SPECIES_FETCH_COUNT = 30
NEW_SPECIES_MIN_FETCHED = 15
NEW_SPECIES_MIN_RECALL = 0.60

# api/main.py runtime configuration, centralized instead of scattered
# os.environ reads.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))

PLANTNET_DAILY_CAP = int(os.environ.get("PLANTNET_DAILY_CAP", "300"))
PLANTNET_PUBLIC_FALLBACK_ENABLED = os.environ.get("PLANTNET_PUBLIC_FALLBACK_ENABLED", "").lower() in {
    "1",
    "true",
    "yes",
}
PLANTNET_STAGE_THRESHOLD = float(os.environ.get("PLANTNET_STAGE_THRESHOLD", "0.70"))

GITHUB_CONTRIB_TOKEN = os.environ.get("GITHUB_CONTRIB_TOKEN", "").strip()
GITHUB_REPO = os.environ.get("GITHUB_REPO", "").strip()
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "contributions")
STAGING_MAX_RETRIES = int(os.environ.get("STAGING_MAX_RETRIES", "3"))

RATE_LIMIT_REQUESTS_PER_MINUTE = int(os.environ.get("RATE_LIMIT_REQUESTS_PER_MINUTE", "30"))

CORS_ALLOW_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ALLOW_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if origin.strip()
]
# Not a wildcard -- the API only ever needs GET (/health) and POST (/predict).
CORS_ALLOW_METHODS = ["GET", "POST", "OPTIONS"]
CORS_ALLOW_HEADERS = ["Content-Type", "Authorization"]

# Retrain-trigger gate (streamlit_app.py). Unset by default -- see docs/SECURITY.md.
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "").strip()
