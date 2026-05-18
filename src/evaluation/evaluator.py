"""
evaluation/evaluator.py
Computes and stores all evaluation metrics, generates confusion matrices,
and ROC curves.

Corresponds to Section 3.6 (Model Evaluation).
"""

import logging
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
)

from src.config import AVERAGE, FIGURES_DIR, REPORTS_DIR, SCORING_METRIC
from src.models.base_classifier import BaseClassifier

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates a collection of trained classifiers on held-out test data and
    identifies the best model by F1-score (Section 3.6).
    """

    METRICS = ("accuracy", "precision", "recall", "f1", "auc")

    def __init__(self, figures_dir: str = FIGURES_DIR):
        self.figures_dir = figures_dir
        os.makedirs(figures_dir, exist_ok=True)
        self.results: dict[str, dict] = {}
        self.best_model_name: str | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate_all(
        self,
        classifiers: dict[str, BaseClassifier],
        X_test, y_test,
    ) -> dict[str, dict]:
        """Evaluate every classifier and store results."""
        for name, clf in classifiers.items():
            logger.info("Evaluating %s …", name)
            self.results[name] = self._compute_metrics(clf, X_test, y_test)

        self._identify_best_model()
        logger.info("Best model: %s (F1=%.4f)",
                    self.best_model_name,
                    self.results[self.best_model_name]["f1"])
        return self.results

    def print_report(self):
        """Print a formatted comparison table to stdout."""
        header = f"{'Model':<28} {'Accuracy':>10} {'Precision':>10} " \
                 f"{'Recall':>10} {'F1':>10} {'AUC':>10}"
        print("\n" + "=" * len(header))
        print(header)
        print("=" * len(header))
        for name, m in sorted(
            self.results.items(),
            key=lambda kv: kv[1]["f1"],
            reverse=True,
        ):
            marker = " ✓" if name == self.best_model_name else ""
            print(
                f"{name:<28} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
                f"{m['recall']:>10.4f} {m['f1']:>10.4f} {m['auc']:>10.4f}"
                f"{marker}"
            )
        print("=" * len(header) + "\n")

    def plot_confusion_matrices(
        self,
        classifiers: dict[str, BaseClassifier],
        X_test, y_test,
    ):
        """Save one confusion-matrix heatmap per classifier."""
        for name, clf in classifiers.items():
            y_pred = clf.predict(X_test)
            cm     = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Ham", "Spam"],
                yticklabels=["Ham", "Spam"],
                ax=ax,
            )
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            ax.set_title(f"Confusion Matrix – {name.replace('_', ' ').title()}")
            path = os.path.join(self.figures_dir, f"cm_{name}.png")
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            logger.info("Saved confusion matrix → %s", path)

    def plot_roc_curves(
        self,
        classifiers: dict[str, BaseClassifier],
        X_test, y_test,
    ):
        """Save a combined ROC curve plot for all classifiers."""
        fig, ax = plt.subplots(figsize=(8, 6))
        colors  = plt.cm.tab10.colors

        for i, (name, clf) in enumerate(classifiers.items()):
            try:
                scores = clf.predict_proba(X_test)
            except Exception:
                continue
            fpr, tpr, _ = roc_curve(y_test, scores)
            auc         = self.results.get(name, {}).get("auc", 0.0)
            label       = f"{name.replace('_', ' ').title()} (AUC={auc:.3f})"
            ax.plot(fpr, tpr, color=colors[i % 10], lw=2, label=label)

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves – All Classifiers")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
        path = os.path.join(self.figures_dir, "roc_curves.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved ROC curves → %s", path)

    def plot_metrics_comparison(self):
        """Bar chart comparing all metrics across classifiers."""
        if not self.results:
            return
        names   = list(self.results.keys())
        metrics = ["accuracy", "precision", "recall", "f1", "auc"]
        x       = np.arange(len(names))
        width   = 0.15

        fig, ax = plt.subplots(figsize=(12, 6))
        for j, metric in enumerate(metrics):
            vals = [self.results[n][metric] for n in names]
            ax.bar(x + j * width, vals, width,
                   label=metric.capitalize())

        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(
            [n.replace("_", "\n") for n in names], fontsize=9)
        ax.set_ylim(0.5, 1.02)
        ax.set_ylabel("Score")
        ax.set_title("Performance Comparison – All Classifiers")
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        path = os.path.join(self.figures_dir, "metrics_comparison.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved metrics comparison → %s", path)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _compute_metrics(
        self, clf: BaseClassifier, X_test, y_test
    ) -> dict:
        y_pred = clf.predict(X_test)
        try:
            scores = clf.predict_proba(X_test)
            auc    = float(roc_auc_score(y_test, scores))
        except Exception:
            auc    = float("nan")

        report = classification_report(
            y_test, y_pred,
            target_names=["ham", "spam"],
            output_dict=True,
        )
        logger.info("\n%s", classification_report(
            y_test, y_pred, target_names=["ham", "spam"]))

        return {
            "accuracy" : float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(
                y_test, y_pred, average=AVERAGE, zero_division=0)),
            "recall"   : float(recall_score(
                y_test, y_pred, average=AVERAGE, zero_division=0)),
            "f1"       : float(f1_score(
                y_test, y_pred, average=AVERAGE, zero_division=0)),
            "auc"      : auc,
            "report"   : report,
        }

    def _identify_best_model(self):
        if self.results:
            self.best_model_name = max(
                self.results, key=lambda k: self.results[k]["f1"]
            )
