"""Central configuration: paths, environment variables and app constants."""
from __future__ import annotations

import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# --- LLM (Groq) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "").strip() or "openai/gpt-oss-120b"
LLM_TEMPERATURE = 0.3
MAX_CONCURRENCY = max(1, _int_env("MAX_CONCURRENCY", 3))
TIMETABLE_LLM = os.getenv("TIMETABLE_LLM", "1").strip() not in ("0", "false", "False", "no")

# --- Embeddings / RAG ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = "curriculum"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
RETRIEVAL_K_PER_GRADE = 3
MIN_RELEVANCE = _float_env("MIN_RELEVANCE", 0.30)

# --- Paths ---
CURRICULUM_DIR = BASE_DIR / "data" / "curriculum"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
OUTPUT_DIR = BASE_DIR / "output"
PDF_DIR = OUTPUT_DIR / "pdfs"
LESSON_DIR = OUTPUT_DIR / "lessons"
FONTS_DIR = BASE_DIR / "fonts"
ADAPTIVE_DIR = BASE_DIR / "data" / "adaptive"  # class profiles for the Adaptive Classroom Assistant

for _d in (CURRICULUM_DIR, UPLOAD_DIR, VECTORSTORE_DIR, PDF_DIR, LESSON_DIR, FONTS_DIR, ADAPTIVE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Uploads ---
MAX_UPLOAD_MB = 10
ALLOWED_EXTENSIONS = {".pdf", ".txt"}

# --- Lesson form ---
SUPPORTED_GRADES = (1, 2, 3, 4, 5)
LANGUAGES = {
    "English": "English",
    "Kannada": "Kannada (ಕನ್ನಡ)",
    "Hindi": "Hindi (हिन्दी)",
}
SUBJECTS = ["Maths", "EVS", "Science", "English", "Kannada", "Hindi", "Social Studies",
            "Art and Craft", "Music", "Physical Education", "Computer Science"]
MIN_DURATION = 15
MAX_DURATION = 180
TOPIC_MAX_LEN = 120

# --- Adaptive Classroom Assistant ---
ADAPTIVE_USE_AI = os.getenv("ADAPTIVE_USE_AI", "1").strip() not in ("0", "false", "False", "no")
ADAPTIVE_RESTARTS = max(1, _int_env("ADAPTIVE_RESTARTS", 10))
# Periods per school day (Adaptive Assistant). Breaks can only follow periods 1..MAX-1.
MIN_PERIODS_PER_DAY = 1
MAX_PERIODS_PER_DAY = min(12, max(1, _int_env("MAX_PERIODS_PER_DAY", 10)))
MIN_BREAK_MINUTES = 5
MAX_BREAK_MINUTES = 120

# --- Flask ---
SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "").strip()
if not SECRET_KEY or SECRET_KEY.startswith("change-me"):
    SECRET_KEY = secrets.token_hex(24)
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "0").strip() in ("1", "true", "True")
PORT = _int_env("PORT", 5000)
