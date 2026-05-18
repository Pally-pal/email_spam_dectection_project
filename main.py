"""
main.py
Full end-to-end pipeline for the Email Spam Detection System.

Usage
-----
    python main.py                              # train with default hyperparameters
    python main.py --optimise                   # run grid-search hyperparameter tuning
    python main.py --eda-only                   # run EDA only (no model training)
    python main.py --predict "Email text here"  # classify a single string
    python main.py --scan-mbox path/to/file.mbox        # scan exported Gmail mbox
    python main.py --scan-mbox path/to/eml_folder/      # scan a folder of .eml files
    python main.py --scan-mbox path/to/email.eml        # scan a single .eml file

How to export your emails (no credentials needed)
--------------------------------------------------
Gmail   -> https://takeout.google.com -> select "Mail" -> download zip
          -> extract -> find "All mail Including Spam and Trash.mbox"
Outlook -> File -> Open & Export -> Import/Export -> Export to .pst
          -> convert .pst to .mbox (free tool: pstconv / libpff)
          OR save individual emails as .eml files
Apple   -> Mailbox menu -> Export Mailbox -> produces a .mbox bundle

Pipeline stages
---------------
    1. Load & audit data         (DataLoader)
    2. Exploratory Data Analysis (EDAAnalyser)
    3. Preprocess text           (TextPreprocessor)
    4. Extract TF-IDF features   (FeatureExtractor)
    5. Train classifiers         (ModelTrainer)
    6. Evaluate classifiers      (ModelEvaluator)
    7. Save artefacts            (ModelTrainer + ModelEvaluator)
"""

import argparse
import logging
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

# -- Add project root to path -----------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    RAW_DATA_PATH, TEST_SIZE, RANDOM_STATE,
    MODELS_DIR, FIGURES_DIR, REPORTS_DIR,
)
from src.data.data_loader import DataLoader
from src.features.preprocessor import TextPreprocessor
from src.features.feature_extractor import FeatureExtractor
from src.models.model_trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator
from src.utils.eda import EDAAnalyser

# -- Logging ----------------------------------------------------------------
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(optimise: bool = False, eda_only: bool = False):
    """Execute the complete machine learning pipeline."""

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # -- 1. Data Loading ----------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 1 - Data Loading")
    logger.info("=" * 60)
    loader = DataLoader(filepath=RAW_DATA_PATH)
    df     = loader.load()
    audit  = loader.audit()
    print("\nData Audit Report:")
    for k, v in audit.items():
        print(f"  {k:<25}: {v}")

    X_raw, y = loader.get_X_y()

    # -- 2. EDA -------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 2 - Exploratory Data Analysis")
    logger.info("=" * 60)
    eda = EDAAnalyser(df, figures_dir=FIGURES_DIR)
    eda.run_all()
    print(f"  EDA figures saved to: {FIGURES_DIR}")

    if eda_only:
        logger.info("EDA-only mode: exiting after EDA.")
        return

    # -- 3. Preprocessing ---------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 3 - Text Preprocessing")
    logger.info("=" * 60)
    preprocessor = TextPreprocessor()
    X_clean = preprocessor.transform(X_raw)
    logger.info("Sample preprocessed email:\n  BEFORE: %s\n  AFTER : %s",
                X_raw.iloc[0][:120], X_clean[0][:120])

    # -- 4. Feature Extraction ----------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 4 - Feature Extraction (TF-IDF + Chi-Squared)")
    logger.info("=" * 60)

    X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
        X_clean, y,
        test_size    = TEST_SIZE,
        stratify     = y,
        random_state = RANDOM_STATE,
    )

    extractor = FeatureExtractor()
    X_train   = extractor.fit_transform(X_tr_raw, y_train)
    X_test    = extractor.transform(X_te_raw)
    logger.info("Feature matrix: train=%s, test=%s",
                X_train.shape, X_test.shape)

    # -- 5. Model Training --------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 5 - Model Training (optimise=%s)", optimise)
    logger.info("=" * 60)
    trainer     = ModelTrainer(optimise=optimise)
    classifiers = trainer.train_all(X_train, y_train)
    trainer.save_models(directory=MODELS_DIR)

    print("\nCross-Validation F1-Scores (mean +/- std):")
    for name, scores in trainer.cv_scores.items():
        print(f"  {name:<28}: {scores['mean']:.4f} +/- {scores['std']:.4f}")

    # -- 6. Evaluation ------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 6 - Model Evaluation")
    logger.info("=" * 60)
    evaluator = ModelEvaluator(figures_dir=FIGURES_DIR)
    evaluator.evaluate_all(classifiers, X_test, y_test)
    evaluator.print_report()
    evaluator.plot_confusion_matrices(classifiers, X_test, y_test)
    evaluator.plot_roc_curves(classifiers, X_test, y_test)
    evaluator.plot_metrics_comparison()

    best = evaluator.best_model_name
    print(f"\nBest model: {best.replace('_', ' ').upper()}")
    print(f"  F1-Score  : {evaluator.results[best]['f1']:.4f}")
    print(f"  Accuracy  : {evaluator.results[best]['accuracy']:.4f}")
    print(f"  AUC       : {evaluator.results[best]['auc']:.4f}")
    print(f"\nAll figures saved to: {FIGURES_DIR}")
    print(f"Trained models saved to: {MODELS_DIR}\n")

    return classifiers, extractor, preprocessor, evaluator


# ---------------------------------------------------------------------------
# Single-string prediction
# ---------------------------------------------------------------------------

def predict_single(text: str):
    """Classify a single email string using the saved best model."""
    import joblib, glob

    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pkl"))
    if not model_files:
        print("No saved models found. Run the training pipeline first.")
        return

    best_name    = "logistic_regression"
    preprocessor = TextPreprocessor()
    clf          = ModelTrainer.load_model(best_name, MODELS_DIR)
    clean        = preprocessor.preprocess_pipeline(text)

    print(f"\nInput text  : {text[:100]}")
    print(f"Preprocessed: {clean[:100]}")
    print("\nNote: for full confidence scores, use the dashboard or --scan-mbox.")


# ---------------------------------------------------------------------------
# Mbox / EML export scanner
# ---------------------------------------------------------------------------

def scan_mbox(path: str):
    """
    Classify every email in an exported .mbox file, a folder of .eml
    files, or a single .eml file.

    The training pipeline must have been run at least once before calling
    this function so that the fitted models and feature extractor are
    available.  The scanner re-trains a fresh preprocessor and reloads
    the saved classifiers, then re-fits the feature extractor on the
    training data so the transform is consistent.

    Parameters
    ----------
    path : str
        Path to a .mbox file, a .eml file, or a directory of .eml files.
    """
    from src.utils.mbox_scanner import MboxScanner

    # -- Check models exist -------------------------------------------------
    import glob as _glob
    saved = _glob.glob(os.path.join(MODELS_DIR, "*.pkl"))
    if not saved:
        print("\n[ERROR] No trained models found.")
        print("        Run  python main.py  first to train the pipeline.\n")
        sys.exit(1)

    # -- Re-build the fitted pipeline (preprocessor + extractor + models) ---
    print("\nLoading trained pipeline for mbox scan ...")

    # 1. Reload training data to refit the extractor (same transform)
    loader      = DataLoader(filepath=RAW_DATA_PATH)
    df          = loader.load()
    X_raw, y    = loader.get_X_y()

    preprocessor = TextPreprocessor()
    X_clean      = preprocessor.transform(X_raw)

    X_tr_raw, _, y_train, _ = train_test_split(
        X_clean, y,
        test_size    = TEST_SIZE,
        stratify     = y,
        random_state = RANDOM_STATE,
    )

    extractor = FeatureExtractor()
    extractor.fit_transform(X_tr_raw, y_train)   # refit so transform works

    # 2. Load saved classifiers
    from src.models.classifiers import ALL_CLASSIFIERS
    classifiers  = {}
    best_name    = "logistic_regression"          # default
    best_f1      = -1.0

    for name in ALL_CLASSIFIERS:
        pkl = os.path.join(MODELS_DIR, f"{name}.pkl")
        if os.path.exists(pkl):
            clf = ModelTrainer.load_model(name, MODELS_DIR)
            classifiers[name] = clf

    if not classifiers:
        print("[ERROR] Could not load any classifiers from", MODELS_DIR)
        sys.exit(1)

    # Try to pick the best model from a saved results summary if present
    results_file = os.path.join(REPORTS_DIR, "best_model.txt")
    if os.path.exists(results_file):
        with open(results_file) as f:
            best_name = f.read().strip()

    print(f"  Loaded {len(classifiers)} classifier(s).")
    print(f"  Primary verdict model: {best_name}\n")

    # -- Run the scanner ----------------------------------------------------
    scanner = MboxScanner(
        preprocessor    = preprocessor,
        extractor       = extractor,
        classifiers     = classifiers,
        best_model_name = best_name,
        reports_dir     = REPORTS_DIR,
    )
    scanner.scan(path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Email Spam Detection - ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                      # train pipeline
  python main.py --optimise                           # train + grid search
  python main.py --eda-only                           # EDA plots only
  python main.py --predict "Win a free iPhone now!"   # classify one email
  python main.py --scan-mbox "C:/exports/mail.mbox"  # scan Gmail export
  python main.py --scan-mbox "C:/exports/eml_folder" # scan EML folder
        """,
    )
    parser.add_argument("--optimise",  action="store_true",
                        help="Run grid-search hyperparameter optimisation.")
    parser.add_argument("--eda-only",  action="store_true",
                        help="Run EDA only (no model training).")
    parser.add_argument("--predict",   type=str, default=None,
                        help="Classify a single email string.")
    parser.add_argument("--scan-mbox", type=str, default=None,
                        metavar="PATH",
                        help="Path to .mbox file, .eml file, or folder of "
                             ".eml files to scan and classify.")
    args = parser.parse_args()

    if args.predict:
        predict_single(args.predict)
    elif args.scan_mbox:
        scan_mbox(args.scan_mbox)
    else:
        run_pipeline(optimise=args.optimise, eda_only=args.eda_only)
