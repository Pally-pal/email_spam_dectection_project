#!/usr/bin/env bash
# build.sh
# =========================================================
# Render build script — runs ONCE during deployment build.
# Installs Python dependencies and downloads required data.
#
# Render calls this automatically via render.yaml buildCommand.
# You can also run it locally before your first launch:
#     bash build.sh
# =========================================================

set -e   # exit immediately on any error

echo "=================================================="
echo "  Email Spam Detection — Build Script"
echo "=================================================="

# ── 1. Upgrade pip ────────────────────────────────────────
echo ""
echo "[1/4] Upgrading pip ..."
pip install --upgrade pip --quiet

# ── 2. Install Python dependencies ────────────────────────
echo "[2/4] Installing dependencies from requirements.txt ..."
pip install -r requirements.txt --quiet
echo "      Done."

# ── 3. Download NLTK corpora ──────────────────────────────
echo "[3/4] Downloading NLTK data (stopwords, punkt) ..."
python - << 'PYEOF'
import nltk, os

# Store NLTK data inside the project so Render can find it
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

for resource in ["stopwords", "punkt", "punkt_tab"]:
    try:
        nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
        print(f"  Downloaded: {resource}")
    except Exception as e:
        print(f"  Warning: could not download {resource}: {e}")
PYEOF
echo "      Done."

# ── 4. Create required directories ────────────────────────
echo "[4/4] Creating runtime directories ..."
mkdir -p data models/saved reports/figures
echo "      Done."

echo ""
echo "=================================================="
echo "  Build complete."
echo "  Next step: place spam.csv in the data/ folder."
echo "  Then start the app:"
echo "    gunicorn 'dashboard.app:app' --bind 0.0.0.0:10000"
echo "=================================================="
