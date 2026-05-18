"""
models/classifiers.py
Concrete classifier implementations for all five algorithms evaluated in
Section 3.4 (Model Selection).

Algorithms:
    1. Naïve Bayes           (MultinomialNB)
    2. Decision Tree         (CART)
    3. Random Forest         (Ensemble)
    4. Support Vector Machine(linear kernel)
    5. Logistic Regression
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier as _DTC
from sklearn.ensemble import RandomForestClassifier as _RFC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression as _LR
from sklearn.calibration import CalibratedClassifierCV

from src.models.base_classifier import BaseClassifier
from src.config import RANDOM_STATE


# ── 1. Naïve Bayes ─────────────────────────────────────────────────────────────

class NaiveBayesClassifier(BaseClassifier):
    """
    Multinomial Naïve Bayes – probabilistic baseline well-suited to
    discrete TF-IDF count features (Section 3.4).
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__(name="naive_bayes")
        self.alpha = alpha

    def build_model(self):
        self.model = MultinomialNB(alpha=self.alpha)


# ── 2. Decision Tree ───────────────────────────────────────────────────────────

class DecisionTreeClassifier(BaseClassifier):
    """
    CART Decision Tree – interpretable rule-learning baseline (Section 3.4).
    """

    def __init__(self, max_depth: int = None, random_state: int = RANDOM_STATE):
        super().__init__(name="decision_tree")
        self.max_depth    = max_depth
        self.random_state = random_state

    def build_model(self):
        self.model = _DTC(
            max_depth    = self.max_depth,
            random_state = self.random_state,
            class_weight = "balanced",
        )


# ── 3. Random Forest ───────────────────────────────────────────────────────────

class RandomForestClassifier(BaseClassifier):
    """
    Random Forest ensemble – reduces variance of single Decision Trees
    via bootstrap aggregation (Section 3.4).
    """

    def __init__(self, n_estimators: int = 200, random_state: int = RANDOM_STATE):
        super().__init__(name="random_forest")
        self.n_estimators = n_estimators
        self.random_state = random_state

    def build_model(self):
        self.model = _RFC(
            n_estimators = self.n_estimators,
            random_state = self.random_state,
            class_weight = "balanced",
            n_jobs       = -1,
        )


# ── 4. SVM (Linear) ────────────────────────────────────────────────────────────

class SVMClassifier(BaseClassifier):
    """
    Linear SVM – margin-maximising classifier with demonstrated superiority
    in high-dimensional sparse text spaces (Section 3.4).

    LinearSVC is wrapped in CalibratedClassifierCV so that predict_proba()
    is available for ROC curve computation.
    """

    def __init__(self, C: float = 1.0, random_state: int = RANDOM_STATE):
        super().__init__(name="svm")
        self.C            = C
        self.random_state = random_state

    def build_model(self):
        base = LinearSVC(
            C            = self.C,
            random_state = self.random_state,
            class_weight = "balanced",
            max_iter     = 2000,
        )
        # Wrap for probability calibration (needed for ROC curves)
        self.model = CalibratedClassifierCV(base, cv=5)


# ── 5. Logistic Regression ─────────────────────────────────────────────────────

class LogisticRegressionClassifier(BaseClassifier):
    """
    Logistic Regression – linear probabilistic classifier with efficient
    training and well-calibrated probability outputs (Section 3.4).
    """

    def __init__(self, C: float = 1.0, random_state: int = RANDOM_STATE):
        super().__init__(name="logistic_regression")
        self.C            = C
        self.random_state = random_state

    def build_model(self):
        self.model = _LR(
            C            = self.C,
            random_state = self.random_state,
            solver       = "lbfgs",
            max_iter     = 1000,
            class_weight = "balanced",
        )


# ── Registry ───────────────────────────────────────────────────────────────────

ALL_CLASSIFIERS: dict[str, BaseClassifier] = {
    "naive_bayes"         : NaiveBayesClassifier(),
    "decision_tree"       : DecisionTreeClassifier(),
    "random_forest"       : RandomForestClassifier(),
    "svm"                 : SVMClassifier(),
    "logistic_regression" : LogisticRegressionClassifier(),
}
