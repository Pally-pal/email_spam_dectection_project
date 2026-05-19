"""
dashboard/app.py
Flask web dashboard for the Email Spam Detection System.

Routes
------
GET  /                  -> Main dashboard page
POST /api/predict       -> Classify a single email (JSON)
GET  /api/models        -> Return all model metrics
GET  /api/dataset/stats -> Return dataset statistics
GET  /api/eda/<name>    -> Serve EDA figure as base64
POST /api/train         -> (Re)train all models on the current dataset
GET  /api/status        -> Pipeline status (trained / not trained)
"""

import os
import sys
import base64
import logging
import threading
from datetime import datetime

from flask import Flask, jsonify, render_template, request

# ── Path setup ─────────────────────────────────────────────────────────────────
# Works both locally (python dashboard/app.py) and on Render (gunicorn
# "dashboard.app:app" launched from repo root).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import (
    RAW_DATA_PATH, TEST_SIZE, RANDOM_STATE,
    MODELS_DIR, FIGURES_DIR, REPORTS_DIR,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder = os.path.join(os.path.dirname(__file__), "templates"),
    static_folder   = os.path.join(os.path.dirname(__file__), "static"),
)

# ── Global pipeline state ──────────────────────────────────────────────────────
_state = {
    "trained"       : False,
    "training"      : False,
    "train_log"     : [],
    "results"       : {},
    "best_model"    : None,
    "dataset_stats" : {},
    "classifiers"   : {},
    "extractor"     : None,
    "preprocessor"  : None,
    "train_started" : None,
    "train_finished": None,
    "error"         : None,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    _state["train_log"].append(f"[{ts}] {msg}")
    logger.info(msg)


def _figure_to_b64(filename: str):
    path = os.path.join(FIGURES_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _run_pipeline_thread():
    """Execute the full ML pipeline in a background thread."""
    from sklearn.model_selection import train_test_split
    from src.data.data_loader           import DataLoader
    from src.features.preprocessor      import TextPreprocessor
    from src.features.feature_extractor import FeatureExtractor
    from src.models.model_trainer       import ModelTrainer
    from src.evaluation.evaluator       import ModelEvaluator
    from src.utils.eda                  import EDAAnalyser

    _state["training"]      = True
    _state["train_log"]     = []
    _state["error"]         = None
    _state["train_started"] = datetime.now().isoformat()

    try:
        os.makedirs(MODELS_DIR,  exist_ok=True)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        os.makedirs(REPORTS_DIR, exist_ok=True)

        # 1. Load
        _log("Loading dataset ...")
        loader = DataLoader(filepath=RAW_DATA_PATH)
        df     = loader.load()
        audit  = loader.audit()
        _state["dataset_stats"] = audit
        X_raw, y = loader.get_X_y()
        _log(f"Dataset loaded: {audit['total_samples']} samples "
             f"(spam={audit['spam_count']}, ham={audit['ham_count']})")

        # 2. EDA
        _log("Running EDA ...")
        EDAAnalyser(df, figures_dir=FIGURES_DIR).run_all()
        _log("EDA figures saved.")

        # 3. Preprocess
        _log("Preprocessing text ...")
        preprocessor = TextPreprocessor()
        X_clean = preprocessor.transform(X_raw)
        _state["preprocessor"] = preprocessor

        # 4. Feature extraction
        _log("Extracting TF-IDF features ...")
        X_tr, X_te, y_train, y_test = train_test_split(
            X_clean, y,
            test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
        )
        extractor = FeatureExtractor()
        X_train   = extractor.fit_transform(X_tr, y_train)
        X_test    = extractor.transform(X_te)
        _state["extractor"] = extractor
        _log(f"Feature matrix: train={X_train.shape}, test={X_test.shape}")

        # 5. Train
        _log("Training all classifiers ...")
        trainer     = ModelTrainer(optimise=False)
        classifiers = trainer.train_all(X_train, y_train)
        trainer.save_models(directory=MODELS_DIR)
        _state["classifiers"] = classifiers
        for name, sc in trainer.cv_scores.items():
            _log(f"  {name}: CV F1 = {sc['mean']:.4f} +/- {sc['std']:.4f}")

        # 6. Evaluate
        _log("Evaluating classifiers ...")
        evaluator = ModelEvaluator(figures_dir=FIGURES_DIR)
        results   = evaluator.evaluate_all(classifiers, X_test, y_test)
        evaluator.plot_confusion_matrices(classifiers, X_test, y_test)
        evaluator.plot_roc_curves(classifiers, X_test, y_test)
        evaluator.plot_metrics_comparison()

        _state["results"] = {
            name: {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in m.items()
                if k != "report"
            }
            for name, m in results.items()
        }
        _state["best_model"] = evaluator.best_model_name
        _state["trained"]    = True
        _log(f"Pipeline complete. Best model: {evaluator.best_model_name}")

    except Exception as exc:
        _state["error"] = str(exc)
        _log(f"ERROR: {exc}")
        logger.exception("Pipeline error")

    finally:
        _state["training"]       = False
        _state["train_finished"] = datetime.now().isoformat()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "trained"   : _state["trained"],
        "training"  : _state["training"],
        "best_model": _state["best_model"],
        "error"     : _state["error"],
        "started"   : _state["train_started"],
        "finished"  : _state["train_finished"],
        "log"       : _state["train_log"][-30:],
    })


@app.route("/api/train", methods=["POST"])
def api_train():
    if _state["training"]:
        return jsonify({"error": "Training already in progress."}), 409
    if not os.path.exists(RAW_DATA_PATH):
        return jsonify({
            "error": (
                f"Dataset not found at {RAW_DATA_PATH}. "
                "Please place spam.csv in the data/ directory."
            )
        }), 400
    threading.Thread(target=_run_pipeline_thread, daemon=True).start()
    return jsonify({"message": "Training started."})


@app.route("/api/models")
def api_models():
    if not _state["trained"]:
        return jsonify({"error": "Models not trained yet."}), 404
    return jsonify({
        "results"   : _state["results"],
        "best_model": _state["best_model"],
    })


@app.route("/api/dataset/stats")
def api_dataset_stats():
    if not _state["dataset_stats"]:
        return jsonify({"error": "Dataset not loaded yet."}), 404
    return jsonify(_state["dataset_stats"])


@app.route("/api/eda/<name>")
def api_eda(name):
    allowed = {
        "class_distribution", "text_length_distribution",
        "word_count_distribution", "top_words",
        "roc_curves", "metrics_comparison",
    }
    if name not in allowed and not name.startswith("cm_"):
        return jsonify({"error": "Unknown figure."}), 404
    b64 = _figure_to_b64(f"{name}.png")
    if b64 is None:
        return jsonify({"error": "Figure not generated yet."}), 404
    return jsonify({"image": f"data:image/png;base64,{b64}"})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not _state["trained"]:
        return jsonify({"error": "Models not trained yet."}), 404

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No email text provided."}), 400

    try:
        clean = _state["preprocessor"].preprocess_pipeline(text)
        X     = _state["extractor"].transform([clean])

        predictions = {}
        for name, clf in _state["classifiers"].items():
            pred = int(clf.predict(X)[0])
            try:
                prob = float(clf.predict_proba(X)[0])
            except Exception:
                prob = None
            predictions[name] = {
                "label"      : "spam" if pred == 1 else "ham",
                "is_spam"    : bool(pred),
                "probability": round(prob, 4) if prob is not None else None,
            }

        best_pred = predictions.get(_state["best_model"], {})
        return jsonify({
            "text"           : text[:200],
            "best_model"     : _state["best_model"],
            "verdict"        : best_pred.get("label", "unknown"),
            "is_spam"        : best_pred.get("is_spam", False),
            "confidence"     : best_pred.get("probability"),
            "all_predictions": predictions,
        })

    except Exception as exc:
        logger.exception("Prediction error")
        return jsonify({"error": str(exc)}), 500


# ── Entry point (local dev only) ───────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Email Spam Detection Dashboard")
    print("  http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=True, port=5000, threaded=True)
