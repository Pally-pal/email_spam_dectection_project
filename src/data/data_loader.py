"""
data/data_loader.py
Handles data acquisition, loading, and initial audit.

Corresponds to Section 3.2 (Data Collection) of the methodology.
"""

import os
import logging

import pandas as pd
import numpy as np

from src.config import (
    RAW_DATA_PATH, TEXT_COLUMN, LABEL_COLUMN,
    SPAM_LABEL, HAM_LABEL, RANDOM_STATE
)

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads a labelled email dataset from CSV, performs an initial audit,
    and exposes clean DataFrames ready for preprocessing.

    Expected CSV columns (flexible – remapped internally):
        - A text column (e.g. 'v2', 'message', 'email', 'text')
        - A label column (e.g. 'v1', 'label') containing 'spam'/'ham'
    """

    # Common column-name aliases used in public datasets (e.g. UCI SMS Spam)
    TEXT_ALIASES  = ["v2", "message", "email", "body", "text", "content"]
    LABEL_ALIASES = ["v1", "label", "category", "class", "type"]

    def __init__(self, filepath: str = RAW_DATA_PATH):
        self.filepath = filepath
        self.raw_df: pd.DataFrame = pd.DataFrame()
        self.clean_df: pd.DataFrame = pd.DataFrame()
        self._is_loaded = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """Load and return the cleaned DataFrame."""
        logger.info("Loading dataset from: %s", self.filepath)
        self.raw_df = pd.read_csv(
            self.filepath,
            encoding="latin-1",
            on_bad_lines="skip",
        )
        self._remap_columns()
        self._encode_labels()
        self._drop_duplicates()
        self.clean_df = self.raw_df[[TEXT_COLUMN, LABEL_COLUMN]].copy()
        self._is_loaded = True
        logger.info("Dataset loaded. Shape: %s", self.clean_df.shape)
        return self.clean_df

    def audit(self) -> dict:
        """Return a summary statistics dict for the loaded dataset."""
        self._assert_loaded()
        total      = len(self.clean_df)
        spam_count = int(self.clean_df[LABEL_COLUMN].sum())
        ham_count  = total - spam_count
        missing    = int(self.clean_df.isnull().sum().sum())

        report = {
            "total_samples"      : total,
            "spam_count"         : spam_count,
            "ham_count"          : ham_count,
            "spam_ratio"         : round(spam_count / total, 4),
            "ham_ratio"          : round(ham_count  / total, 4),
            "missing_values"     : missing,
            "avg_text_length"    : round(
                self.clean_df[TEXT_COLUMN].str.len().mean(), 1),
            "median_text_length" : round(
                self.clean_df[TEXT_COLUMN].str.len().median(), 1),
        }
        logger.info("Data audit:\n%s", report)
        return report

    def get_X_y(self):
        """Return feature series X and label series y."""
        self._assert_loaded()
        X = self.clean_df[TEXT_COLUMN]
        y = self.clean_df[LABEL_COLUMN]
        return X, y

    # ── Private helpers ────────────────────────────────────────────────────────

    def _remap_columns(self):
        """Rename dataset columns to standard TEXT_COLUMN / LABEL_COLUMN."""
        cols = {c.lower(): c for c in self.raw_df.columns}

        text_col = next(
            (cols[a] for a in self.TEXT_ALIASES  if a in cols), None)
        lbl_col  = next(
            (cols[a] for a in self.LABEL_ALIASES if a in cols), None)

        if text_col is None or lbl_col is None:
            raise ValueError(
                f"Could not auto-detect text/label columns. "
                f"Found columns: {list(self.raw_df.columns)}"
            )

        self.raw_df.rename(
            columns={text_col: TEXT_COLUMN, lbl_col: LABEL_COLUMN},
            inplace=True,
        )
        # Drop any extra columns produced by some CSV exports
        extra = [c for c in self.raw_df.columns
                 if c not in (TEXT_COLUMN, LABEL_COLUMN)]
        if extra:
            self.raw_df.drop(columns=extra, inplace=True)

    def _encode_labels(self):
        """Convert string 'spam'/'ham' labels to integer 1/0.

        Handles all pandas string dtypes (object, StringDtype, ArrowDtype)
        and strips leading/trailing whitespace before mapping.
        """
        col = self.raw_df[LABEL_COLUMN]

        # Check whether the column contains string-like values.
        # We test the first non-null value rather than relying on dtype,
        # because newer pandas uses StringDtype / ArrowDtype instead of object.
        is_string_col = (
            col.dtype == object
            or hasattr(col.dtype, "name") and "string" in col.dtype.name.lower()
            or str(col.dtype).lower() in ("string", "string[python]",
                                           "string[pyarrow]", "large_string")
            or (col.dropna().shape[0] > 0
                and isinstance(col.dropna().iloc[0], str))
        )

        if is_string_col:
            mapping = {"spam": SPAM_LABEL, "ham": HAM_LABEL}
            # .str accessor works on all string-backed dtypes;
            # strip() removes any invisible whitespace that breaks mapping.
            self.raw_df[LABEL_COLUMN] = (
                col.astype(str)          # normalise to plain Python str dtype
                   .str.strip()          # remove leading/trailing whitespace
                   .str.lower()          # normalise case
                   .map(mapping)         # 'spam' → 1, 'ham' → 0
            )

        # Drop any rows where the label could not be mapped (NaN after map)
        before = len(self.raw_df)
        self.raw_df.dropna(subset=[LABEL_COLUMN], inplace=True)

        # Safe int conversion: use float first to avoid issues with NaN
        # remnants on some pandas builds, then cast to int.
        self.raw_df[LABEL_COLUMN] = (
            self.raw_df[LABEL_COLUMN].astype(float).astype(int)
        )

        after = len(self.raw_df)
        if before != after:
            logger.warning(
                "Dropped %d rows with unrecognised labels.", before - after)

    def _drop_duplicates(self):
        before = len(self.raw_df)
        self.raw_df.drop_duplicates(subset=[TEXT_COLUMN], inplace=True)
        self.raw_df.reset_index(drop=True, inplace=True)
        dropped = before - len(self.raw_df)
        if dropped:
            logger.info("Removed %d duplicate rows.", dropped)

    def _assert_loaded(self):
        if not self._is_loaded:
            raise RuntimeError("Call .load() before accessing data.")