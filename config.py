"""config.py - central settings. Every other module imports its settings from here."""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

SAMPLE_LOGS_CSV = DATA_DIR / "sample_logs.csv"
KNOWLEDGE_BASE_JSON = DATA_DIR / "troubleshooting.json"
MODEL_PATH = MODELS_DIR / "error_classifier.keras"
VOCAB_PATH = MODELS_DIR / "vocab.json"
LABELS_PATH = MODELS_DIR / "labels.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
DB_PATH = BASE_DIR / "history.db"

# Read the .env file (if it exists) so GOOGLE_API_KEY becomes available.
load_dotenv(BASE_DIR / ".env")

ALLOWED_EXTENSIONS = (".log", ".txt", ".csv")
MAX_FILE_MB = 5
MAX_AI_CHARS = 6000          # keeps prompts short to save tokens
CONFIDENCE_THRESHOLD = 0.50  # below this, the classifier says "Unknown"
UNKNOWN_CATEGORY = "Unknown / Unsupported"

CATEGORIES = [
    "Build Failure",
    "Dependency Issue",
    "Configuration Issue",
    "Database Connectivity Failure",
    "Permission Issue",
    "Deployment Timeout",
    "Resource Exhaustion",
    "Missing Environment Variables",
]


def get_api_key() -> str:
    """Return the Gemini API key from .env / environment, or from Streamlit secrets."""
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key:
        try:
            import streamlit as st
            key = str(st.secrets.get("GOOGLE_API_KEY", "")).strip()
        except Exception:  # no secrets file, or not running inside Streamlit
            key = ""
    return "" if key in ("", "your_api_key_here") else key


def get_model_name() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()


def setup_logging() -> None:
    """Log to the console and to app.log. We never log API keys or raw log contents."""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(BASE_DIR / "app.log", encoding="utf-8")],
    )
