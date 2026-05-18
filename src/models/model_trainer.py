"""
models/model_trainer.py
Handles stratified train/test splitting, k-fold cross-validation, and
grid-search hyperparameter optimisation.

Corresponds to Section 3.5 (Model Training).
"""

import logging
import os

import joblib
import numpy as np
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, GridSearchCV
)

from src.config import (
    TEST_SIZE, RANDOM_STATE, CV_FOLDS, SCORING_METRIC, MODELS_DIR
)
from src.models.classifiers import ALL_CLASSIFIERS, BaseClassifier

logger = logging.getLogger(__name__)

# Hyperparameter grids for grid-search (top candidates only)
PARAM_GRIDS: dict[str, dict] = {
    "naive_bayes": {
        "alpha": [0.1, 0.5, 1.0, 2.0],
    },
    "decision_tree": {
        "max_depth": [None, 10, 20, 30],
    },
    "random_forest": {
        "n_estimators": [100, 200, 300],
    },
    "svm": {
        "estimator__C": [0.1, 1.0, 10.0],
    },
    "logistic_regression": {
        "C": [0.1, 1.0, 10.0],
    },
}


class ModelTrainer:
    """
    Trains every classifier, optionally tuning hyperparameters via
    grid-search cross-validation, and saves fitted models to disk.
    """

    def __init__(self, optimise: bool = False):
        """
        Args:
            optimise: If True, run grid-search CV for top classifiers.
                      If False, train with default hyperparameters.
        """
        self.optimise      = optimise
        self.trained: dict[str, BaseClassifier] = {}
        self.cv_scores: dict[str, dict]         = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def train_all(self, X_train, y_train) -> dict[str, BaseClassifier]:
        """Train all classifiers and return dict name → fitted classifier."""
        for name, clf in ALL_CLASSIFIERS.items():
            logger.info("Training %s …", name)
            try:
                if self.optimise and name in PARAM_GRIDS:
                    clf = self._grid_search(clf, X_train, y_train, name)
                else:
                    clf.train(X_train, y_train)

                self.cv_scores[name] = self._cross_validate(
                    clf, X_train, y_train)
                self.trained[name]   = clf
                logger.info(
                    "  CV F1: %.4f ± %.4f",
                    self.cv_scores[name]["mean"],
                    self.cv_scores[name]["std"],
                )
            except Exception as exc:                      # pragma: no cover
                logger.error("Failed to train %s: %s", name, exc)

        return self.trained

    def save_models(self, directory: str = MODELS_DIR):
        """Persist all fitted classifiers to disk with joblib."""
        os.makedirs(directory, exist_ok=True)
        for name, clf in self.trained.items():
            path = os.path.join(directory, f"{name}.pkl")
            joblib.dump(clf, path)
            logger.info("Saved %s → %s", name, path)

    @staticmethod
    def load_model(name: str, directory: str = MODELS_DIR) -> BaseClassifier:
        """Load a previously saved classifier from disk."""
        path = os.path.join(directory, f"{name}.pkl")
        clf  = joblib.load(path)
        logger.info("Loaded %s from %s", name, path)
        return clf

    # ── Private helpers ────────────────────────────────────────────────────────

    def _cross_validate(self, clf: BaseClassifier, X, y) -> dict:
        """Run stratified k-fold CV and return mean/std F1."""
        skf    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                                 random_state=RANDOM_STATE)
        scores = cross_val_score(
            clf.model, X, y,
            cv      = skf,
            scoring = SCORING_METRIC,
            n_jobs  = -1,
        )
        return {"mean": float(scores.mean()), "std": float(scores.std()),
                "all": scores.tolist()}

    def _grid_search(
        self,
        clf: BaseClassifier,
        X, y,
        name: str,
    ) -> BaseClassifier:
        """Fit a GridSearchCV and update clf.model with the best estimator."""
        clf.build_model()
        grid = GridSearchCV(
            clf.model,
            PARAM_GRIDS[name],
            scoring   = SCORING_METRIC,
            cv        = StratifiedKFold(
                n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            n_jobs    = -1,
            refit     = True,
            verbose   = 0,
        )
        grid.fit(X, y)
        clf.model     = grid.best_estimator_
        clf.is_fitted = True
        logger.info("  Best params for %s: %s", name, grid.best_params_)
        return clf
