"""
features/feature_extractor.py
TF-IDF vectorisation and chi-squared feature selection.

Corresponds to Section 3.3 – feature extraction sub-pipeline.
"""

import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

from src.config import MAX_FEATURES, NGRAM_RANGE, MIN_DF, CHI2_K

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Transforms preprocessed email text into a numerical TF-IDF feature matrix,
    with optional chi-squared feature selection as described in Section 3.3.

    Usage::

        extractor = FeatureExtractor()
        X_train = extractor.fit_transform(train_texts, y_train)
        X_test  = extractor.transform(test_texts)
    """

    def __init__(
        self,
        max_features: int = MAX_FEATURES,
        ngram_range: tuple= NGRAM_RANGE,
        min_df: int       = MIN_DF,
        chi2_k: int       = CHI2_K,
        use_chi2: bool    = True,
    ):
        self.max_features = max_features
        self.ngram_range  = ngram_range
        self.min_df       = min_df
        self.chi2_k       = chi2_k
        self.use_chi2     = use_chi2

        self._vectorizer = TfidfVectorizer(
            max_features = max_features,
            ngram_range  = ngram_range,
            min_df       = min_df,
            sublinear_tf = True,    # use 1 + log(tf) instead of raw tf
            norm         = "l2",    # L2-normalise each row
            strip_accents= "unicode",
        )
        self._selector: SelectKBest | None = None
        self._is_fitted = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def fit_transform(self, texts, y) -> np.ndarray:
        """
        Fit the TF-IDF vectoriser and chi-squared selector on training data,
        then transform and return the feature matrix.
        """
        logger.info("Fitting TF-IDF vectoriser (max_features=%d, ngram=%s) …",
                    self.max_features, self.ngram_range)
        X = self._vectorizer.fit_transform(texts)
        logger.info("Vocabulary size after TF-IDF: %d", X.shape[1])

        if self.use_chi2 and self.chi2_k < X.shape[1]:
            logger.info("Applying chi-squared selection (k=%d) …", self.chi2_k)
            self._selector = SelectKBest(chi2, k=self.chi2_k)
            X = self._selector.fit_transform(X, y)
            logger.info("Feature matrix shape after chi2: %s", X.shape)

        self._is_fitted = True
        return X

    def transform(self, texts) -> np.ndarray:
        """Transform new text(s) using the fitted vectoriser (+selector)."""
        self._assert_fitted()
        X = self._vectorizer.transform(texts)
        if self._selector is not None:
            X = self._selector.transform(X)
        return X

    def get_feature_names(self) -> list[str]:
        """Return the feature names retained after selection."""
        self._assert_fitted()
        names = np.array(self._vectorizer.get_feature_names_out())
        if self._selector is not None:
            names = names[self._selector.get_support()]
        return names.tolist()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _assert_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Call fit_transform() before transform().")
