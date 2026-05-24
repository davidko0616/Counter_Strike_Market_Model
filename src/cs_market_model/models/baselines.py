"""Rule and sklearn baseline models for Day 8-9."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LABEL_BAD_TIME = 0
LABEL_GOOD_TIME = 1
LABEL_STRONG_BUY = 2


@dataclass(frozen=True)
class RuleBaselineConfig:
    """Configuration for quantile-based rule baselines."""

    model_name: str
    score_column: str
    higher_is_better: bool = True
    lower_quantile: float = 0.25
    upper_quantile: float = 0.75


@dataclass
class QuantileRuleBaseline:
    """Three-class rule model driven by a single continuous feature."""

    config: RuleBaselineConfig
    lower_threshold: float | None = None
    upper_threshold: float | None = None
    fallback_score: float = 0.0

    def fit(self, feature_frame: pd.DataFrame) -> QuantileRuleBaseline:
        scores = self._score(feature_frame)
        finite_scores = scores[np.isfinite(scores)]
        if finite_scores.empty:
            self.lower_threshold = self.fallback_score
            self.upper_threshold = self.fallback_score
            return self

        self.fallback_score = float(finite_scores.median())
        self.lower_threshold = float(finite_scores.quantile(self.config.lower_quantile))
        self.upper_threshold = float(finite_scores.quantile(self.config.upper_quantile))
        return self

    def predict(self, feature_frame: pd.DataFrame) -> np.ndarray:
        if self.lower_threshold is None or self.upper_threshold is None:
            raise ValueError("Rule baseline must be fitted before prediction")

        scores = self.score(feature_frame)
        predictions = np.full(len(scores), LABEL_GOOD_TIME, dtype=int)
        predictions[scores <= self.lower_threshold] = LABEL_BAD_TIME
        predictions[scores >= self.upper_threshold] = LABEL_STRONG_BUY
        return predictions

    def score(self, feature_frame: pd.DataFrame) -> np.ndarray:
        scores = self._score(feature_frame)
        return scores.fillna(self.fallback_score).to_numpy(dtype=float)

    def _score(self, feature_frame: pd.DataFrame) -> pd.Series:
        if self.config.score_column not in feature_frame.columns:
            raise ValueError(f"Missing rule score column: {self.config.score_column}")
        scores = pd.to_numeric(feature_frame[self.config.score_column], errors="coerce")
        if not self.config.higher_is_better:
            scores = -scores
        return scores


def default_rule_configs() -> list[RuleBaselineConfig]:
    """Return the initial rule baselines for Day 8-9."""
    return [
        RuleBaselineConfig(
            model_name="momentum_rule_14d",
            score_column="log_return_14d",
            higher_is_better=True,
        ),
        RuleBaselineConfig(
            model_name="mean_reversion_rule_30d",
            score_column="close_to_ma_30d",
            higher_is_better=False,
        ),
    ]


def make_logistic_regression(random_state: int = 42) -> Pipeline:
    """Create a balanced multinomial logistic regression baseline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_random_forest(random_state: int = 42) -> Pipeline:
    """Create a balanced random forest baseline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=250,
                    min_samples_leaf=10,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def strong_buy_probability(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return model probability for the Strong Buy class, if available."""
    return class_probability(model, features, positive_class=LABEL_STRONG_BUY)


def class_probability(
    model: Pipeline,
    features: pd.DataFrame,
    *,
    positive_class: int,
) -> np.ndarray:
    """Return model probability for a selected class, if available."""
    probabilities = model.predict_proba(features)
    classes = getattr(model, "classes_", model.named_steps["model"].classes_)
    matches = np.flatnonzero(classes == positive_class)
    if len(matches) == 0:
        return np.zeros(len(features), dtype=float)
    return probabilities[:, int(matches[0])]
