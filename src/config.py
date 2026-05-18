"""
config.py
Central configuration for the Email Spam Detection System.

Paths are derived from the project root so the app works both locally
and on Render (where the working directory is the repo root).
"""

import os

# ── Project root ───────────────────────────────────────────────────────────────
# src/config.py is at <root>/src/config.py, so root = two levels up.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR      = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "spam.csv")
MODELS_DIR    = os.path.join(BASE_DIR, "models", "saved")
REPORTS_DIR   = os.path.join(BASE_DIR, "reports")
FIGURES_DIR   = os.path.join(REPORTS_DIR, "figures")

# ── Dataset ────────────────────────────────────────────────────────────────────
TEXT_COLUMN  = "text"
LABEL_COLUMN = "label"
SPAM_LABEL   = 1
HAM_LABEL    = 0
TEST_SIZE    = 0.20
RANDOM_STATE = 42
CV_FOLDS     = 10

# ── Preprocessing ──────────────────────────────────────────────────────────────
LOWERCASE        = True
REMOVE_PUNCT     = True
REMOVE_NUMBERS   = True
REMOVE_STOPWORDS = True
APPLY_STEMMING   = True

# ── Feature Extraction ─────────────────────────────────────────────────────────
MAX_FEATURES = 5000
NGRAM_RANGE  = (1, 2)
MIN_DF       = 2
CHI2_K       = 3000

# ── Model names ────────────────────────────────────────────────────────────────
MODEL_NAMES = [
    "naive_bayes",
    "decision_tree",
    "random_forest",
    "svm",
    "logistic_regression",
]

# ── Evaluation ─────────────────────────────────────────────────────────────────
SCORING_METRIC = "f1"
AVERAGE        = "binary"
