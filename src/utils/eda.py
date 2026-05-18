"""
utils/eda.py
Exploratory Data Analysis (EDA) utilities.

Corresponds to Section 3.2 – initial data audit and EDA objectives stated
in Section 1.3.
"""

import logging
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.config import TEXT_COLUMN, LABEL_COLUMN, FIGURES_DIR

logger = logging.getLogger(__name__)


class EDAAnalyser:
    """Generates and saves EDA plots for the email dataset."""

    def __init__(self, df: pd.DataFrame, figures_dir: str = FIGURES_DIR):
        self.df          = df.copy()
        self.figures_dir = figures_dir
        os.makedirs(figures_dir, exist_ok=True)
        self.df["text_length"] = self.df[TEXT_COLUMN].str.len()
        self.df["word_count"]  = self.df[TEXT_COLUMN].str.split().str.len()

    # ── Public API ─────────────────────────────────────────────────────────────

    def run_all(self):
        """Execute the full EDA pipeline and save all figures."""
        self.plot_class_distribution()
        self.plot_text_length_distribution()
        self.plot_word_count_distribution()
        self.plot_top_words()
        logger.info("EDA complete. Figures saved to %s", self.figures_dir)

    def plot_class_distribution(self):
        counts = self.df[LABEL_COLUMN].value_counts()
        labels = ["Ham (Legitimate)", "Spam"]
        fig, ax = plt.subplots(figsize=(6, 5))
        bars = ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)],
                      color=["#4CAF50", "#F44336"])
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 20,
                    str(int(bar.get_height())),
                    ha="center", va="bottom", fontweight="bold")
        ax.set_title("Class Distribution: Ham vs Spam")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.3)
        self._save(fig, "class_distribution.png")

    def plot_text_length_distribution(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        for label, colour, name in [(0, "#4CAF50", "Ham"), (1, "#F44336", "Spam")]:
            subset = self.df[self.df[LABEL_COLUMN] == label]["text_length"]
            ax.hist(subset, bins=50, alpha=0.6, color=colour, label=name)
        ax.set_title("Email Text Length Distribution")
        ax.set_xlabel("Character Count")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(alpha=0.3)
        self._save(fig, "text_length_distribution.png")

    def plot_word_count_distribution(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        for label, colour, name in [(0, "#4CAF50", "Ham"), (1, "#F44336", "Spam")]:
            subset = self.df[self.df[LABEL_COLUMN] == label]["word_count"]
            ax.hist(subset, bins=50, alpha=0.6, color=colour, label=name)
        ax.set_title("Word Count Distribution")
        ax.set_xlabel("Word Count")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(alpha=0.3)
        self._save(fig, "word_count_distribution.png")

    def plot_top_words(self, top_n: int = 20):
        """Bar charts of the most frequent words for spam and ham."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, label, name, colour in [
            (axes[0], 0, "Ham",  "#4CAF50"),
            (axes[1], 1, "Spam", "#F44336"),
        ]:
            texts  = self.df[self.df[LABEL_COLUMN] == label][TEXT_COLUMN]
            words  = " ".join(texts).lower().split()
            counts = Counter(words).most_common(top_n)
            terms, freqs = zip(*counts)
            ax.barh(list(reversed(terms)), list(reversed(freqs)), color=colour)
            ax.set_title(f"Top {top_n} Words – {name}")
            ax.set_xlabel("Frequency")
            ax.grid(axis="x", alpha=0.3)
        fig.suptitle("Most Frequent Words by Class", fontsize=13,
                     fontweight="bold")
        self._save(fig, "top_words.png")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _save(self, fig, filename: str):
        path = os.path.join(self.figures_dir, filename)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved EDA figure → %s", path)
