"""
tests/test_suite.py
Unit tests for the Email Spam Detection System.
Run with: pytest tests/test_suite.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.features.preprocessor import TextPreprocessor
from src.features.feature_extractor import FeatureExtractor
from src.models.classifiers import (
    NaiveBayesClassifier, LogisticRegressionClassifier,
    RandomForestClassifier, DecisionTreeClassifier, SVMClassifier,
)
from src.evaluation.evaluator import ModelEvaluator


# ── Fixtures ──────────────────────────────────────────────────────────────────

SPAM_TEXTS = [
    "Congratulations! You have won a FREE prize. Click here NOW to claim!!!",
    "Urgent: Your account has been compromised. Verify immediately.",
    "Get rich quick with our proven system. Make $5000 a week from home!",
    "You are selected for an exclusive offer. Limited time only. Act fast!",
    "Win a brand new iPhone! Click the link below and enter your details.",
]

HAM_TEXTS = [
    "Can we schedule a meeting for tomorrow at 3pm to discuss the project?",
    "Please find attached the quarterly report as requested.",
    "The team lunch is confirmed for Friday. Let me know if you can make it.",
    "Thank you for your email. I will get back to you by end of the week.",
    "Just a reminder that the deadline for the assignment is next Monday.",
]

ALL_TEXTS  = SPAM_TEXTS + HAM_TEXTS
ALL_LABELS = np.array([1] * 5 + [0] * 5)


# ── TextPreprocessor tests ─────────────────────────────────────────────────────

class TestTextPreprocessor:
    def setup_method(self):
        self.prep = TextPreprocessor()

    def test_lowercase(self):
        result = self.prep.clean_text("HELLO WORLD")
        assert result == result.lower()

    def test_removes_punctuation(self):
        result = self.prep.clean_text("Hello, World!!!")
        assert "," not in result and "!" not in result

    def test_removes_urls(self):
        result = self.prep.clean_text("Visit https://example.com now")
        assert "http" not in result

    def test_removes_numbers(self):
        result = self.prep.clean_text("Call 08012345678 for info")
        assert not any(c.isdigit() for c in result)

    def test_pipeline_returns_string(self):
        result = self.prep.preprocess_pipeline("Win a free prize now!")
        assert isinstance(result, str)

    def test_transform_list(self):
        results = self.prep.transform(["Hello world", "Spam email"])
        assert len(results) == 2

    def test_empty_string(self):
        result = self.prep.preprocess_pipeline("")
        assert result == ""

    def test_non_string_input(self):
        result = self.prep.clean_text(None)
        assert result == ""


# ── FeatureExtractor tests ─────────────────────────────────────────────────────

class TestFeatureExtractor:
    def setup_method(self):
        self.prep = TextPreprocessor()
        self.clean_texts = self.prep.transform(ALL_TEXTS)
        self.extractor = FeatureExtractor(
            max_features=100, chi2_k=50, min_df=1)

    def test_fit_transform_shape(self):
        X = self.extractor.fit_transform(self.clean_texts, ALL_LABELS)
        assert X.shape[0] == len(ALL_TEXTS)
        assert X.shape[1] <= 50          # chi2 selection

    def test_transform_unseen(self):
        self.extractor.fit_transform(self.clean_texts, ALL_LABELS)
        X_new = self.extractor.transform(["Free prize win now"])
        assert X_new.shape[0] == 1

    def test_feature_names_returned(self):
        self.extractor.fit_transform(self.clean_texts, ALL_LABELS)
        names = self.extractor.get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_transform_before_fit_raises(self):
        ext = FeatureExtractor(max_features=100, chi2_k=50, min_df=1)
        with pytest.raises(RuntimeError):
            ext.transform(["some text"])


# ── Classifier tests ───────────────────────────────────────────────────────────

def make_features():
    prep = TextPreprocessor()
    ext  = FeatureExtractor(max_features=100, chi2_k=50, min_df=1)
    clean = prep.transform(ALL_TEXTS)
    X     = ext.fit_transform(clean, ALL_LABELS)
    return X, ALL_LABELS


class TestNaiveBayes:
    def test_train_predict(self):
        X, y = make_features()
        clf  = NaiveBayesClassifier()
        clf.train(X, y)
        preds = clf.predict(X)
        assert preds.shape == y.shape

    def test_predict_proba(self):
        X, y = make_features()
        clf  = NaiveBayesClassifier()
        clf.train(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape[0] == len(y)
        assert np.all((proba >= 0) & (proba <= 1))


class TestRandomForest:
    def test_train_predict(self):
        X, y = make_features()
        clf  = RandomForestClassifier(n_estimators=10)
        clf.train(X, y)
        preds = clf.predict(X)
        assert set(preds).issubset({0, 1})


class TestLogisticRegression:
    def test_train_predict(self):
        X, y = make_features()
        clf  = LogisticRegressionClassifier()
        clf.train(X, y)
        preds = clf.predict(X)
        assert preds.shape == y.shape


class TestPredict_Before_Train:
    def test_raises(self):
        clf = NaiveBayesClassifier()
        X, _ = make_features()
        with pytest.raises(RuntimeError):
            clf.predict(X)


# ── ModelEvaluator tests ───────────────────────────────────────────────────────

class TestModelEvaluator:
    def setup_method(self):
        X, y = make_features()
        self.clf = NaiveBayesClassifier()
        self.clf.train(X, y)
        self.X_test = X
        self.y_test = y
        self.evaluator = ModelEvaluator(figures_dir="/tmp/spam_test_figs")

    def test_evaluate_all(self):
        results = self.evaluator.evaluate_all(
            {"naive_bayes": self.clf}, self.X_test, self.y_test)
        assert "naive_bayes" in results
        m = results["naive_bayes"]
        for key in ("accuracy", "precision", "recall", "f1"):
            assert 0.0 <= m[key] <= 1.0

    def test_best_model_identified(self):
        self.evaluator.evaluate_all(
            {"naive_bayes": self.clf}, self.X_test, self.y_test)
        assert self.evaluator.best_model_name == "naive_bayes"
