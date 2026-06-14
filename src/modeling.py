"""Обучение и оценка моделей предсказания одобрения кредита."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False


@dataclass
class ModelResult:
    name: str
    accuracy: float
    f1: float
    roc_auc: float
    report: str
    confusion: np.ndarray
    model: Pipeline
    y_proba: np.ndarray


def get_models(random_state: int = 42) -> dict[str, object]:
    """Возвращает набор моделей для сравнения."""
    models: dict[str, object] = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=random_state,
            class_weight="balanced",
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
        )
    else:
        models["Gradient Boosting"] = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state,
        )

    return models


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[dict[str, ModelResult], pd.DataFrame, np.ndarray, np.ndarray]:
    """Обучает модели и возвращает метрики на тестовой выборке."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(
        numeric_features=[col for col in X.columns if col not in ("education", "self_employed")],
        categorical_features=[col for col in ("education", "self_employed") if col in X.columns],
    )

    results: dict[str, ModelResult] = {}
    for name, estimator in get_models(random_state).items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        results[name] = ModelResult(
            name=name,
            accuracy=accuracy_score(y_test, y_pred),
            f1=f1_score(y_test, y_pred),
            roc_auc=roc_auc_score(y_test, y_proba),
            report=classification_report(y_test, y_pred, digits=3),
            confusion=confusion_matrix(y_test, y_pred),
            model=pipeline,
            y_proba=y_proba,
        )

    metrics_df = pd.DataFrame(
        [
            {
                "model": result.name,
                "accuracy": result.accuracy,
                "f1": result.f1,
                "roc_auc": result.roc_auc,
            }
            for result in results.values()
        ]
    ).sort_values("roc_auc", ascending=False)

    return results, metrics_df, y_test.to_numpy(), X_test.index.to_numpy()


def get_best_tree_model_name() -> str:
    """Возвращает имя наиболее мощной доступной tree-based модели."""
    return "XGBoost" if XGBOOST_AVAILABLE else "Gradient Boosting"


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str | None = None,
    cv: int = 5,
    random_state: int = 42,
) -> pd.Series:
    """Кросс-валидация лучшей модели по ROC-AUC."""
    model_name = model_name or get_best_tree_model_name()
    preprocessor = build_preprocessor(
        numeric_features=[col for col in X.columns if col not in ("education", "self_employed")],
        categorical_features=[col for col in ("education", "self_employed") if col in X.columns],
    )
    estimator = get_models(random_state)[model_name]
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
    return pd.Series(scores, name="roc_auc")


def get_roc_curve_data(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает данные для построения ROC-кривой."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    return fpr, tpr, thresholds


def extract_feature_importance(result: ModelResult, feature_names: list[str]) -> pd.DataFrame:
    """Извлекает важность признаков для tree-based моделей."""
    model = result.model.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    importances = model.feature_importances_
    return (
        pd.DataFrame({"feature": feature_names[: len(importances)], "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
