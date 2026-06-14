"""Очистка данных и создание признаков для модели одобрения кредита."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
    "loan_to_income",
    "debt_burden_ratio",
    "total_assets_value",
    "assets_per_income",
    "monthly_payment_estimate",
    "payment_to_income",
    "avg_consumer_loan_rate_rub",
    "key_rate",
    "macro_spread",
]

CATEGORICAL_FEATURES = ["education", "self_employed"]


def clean_loan_data(df: pd.DataFrame) -> pd.DataFrame:
    """Базовая очистка: типы, дубликаты, выбросы."""
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip()

    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.drop(columns=["loan_id"], errors="ignore")

    cleaned["loan_status"] = cleaned["loan_status"].str.strip()
    cleaned["education"] = cleaned["education"].str.strip()
    cleaned["self_employed"] = cleaned["self_employed"].str.strip()

    numeric_cols = [
        "no_of_dependents",
        "income_annum",
        "loan_amount",
        "loan_term",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value",
        "avg_consumer_loan_rate_rub",
        "key_rate",
    ]
    for col in numeric_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned = _cap_outliers_iqr(
        cleaned,
        columns=[
            "income_annum",
            "loan_amount",
            "residential_assets_value",
            "commercial_assets_value",
            "luxury_assets_value",
            "bank_asset_value",
        ],
    )

    return cleaned


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Создаёт расчётные признаки для кредитного скоринга."""
    featured = df.copy()

    featured["loan_to_income"] = featured["loan_amount"] / featured["income_annum"].replace(0, np.nan)
    featured["debt_burden_ratio"] = featured["loan_amount"] / (
        featured["income_annum"] * featured["loan_term"].replace(0, np.nan)
    )
    featured["total_assets_value"] = (
        featured["residential_assets_value"]
        + featured["commercial_assets_value"]
        + featured["luxury_assets_value"]
        + featured["bank_asset_value"]
    )
    featured["assets_per_income"] = featured["total_assets_value"] / featured["income_annum"].replace(0, np.nan)

    annual_rate = featured.get("avg_consumer_loan_rate_rub", pd.Series(12.0, index=featured.index)).fillna(12.0) / 100
    featured["monthly_payment_estimate"] = (
        featured["loan_amount"] * (annual_rate / 12) * (1 + annual_rate / 12) ** (featured["loan_term"] * 12)
    ) / ((1 + annual_rate / 12) ** (featured["loan_term"] * 12) - 1)
    featured["payment_to_income"] = featured["monthly_payment_estimate"] / (featured["income_annum"] / 12).replace(0, np.nan)

    if "avg_consumer_loan_rate_rub" in featured.columns and "key_rate" in featured.columns:
        featured["macro_spread"] = featured["avg_consumer_loan_rate_rub"] - featured["key_rate"]

    featured["is_graduate"] = (featured["education"] == "Graduate").astype(int)
    featured["is_self_employed"] = (featured["self_employed"] == "Yes").astype(int)

    return featured


def prepare_target(df: pd.DataFrame, target_col: str = "loan_status") -> tuple[pd.DataFrame, pd.Series]:
    """Кодирует целевую переменную: 1 — одобрен, 0 — отказ."""
    data = df.copy()
    y = (data[target_col] == "Approved").astype(int)
    data = data.drop(columns=[target_col])
    return data, y


def _cap_outliers_iqr(df: pd.DataFrame, columns: list[str], factor: float = 1.5) -> pd.DataFrame:
    capped = df.copy()
    for col in columns:
        if col not in capped.columns:
            continue
        q1 = capped[col].quantile(0.25)
        q3 = capped[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        capped[col] = capped[col].clip(lower=lower, upper=upper)
    return capped


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Создаёт sklearn-препроцессор для числовых и категориальных признаков."""
    numeric_features = numeric_features or NUMERIC_FEATURES
    categorical_features = categorical_features or CATEGORICAL_FEATURES

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Возвращает таблицу признаков для обучения модели."""
    available_numeric = [col for col in NUMERIC_FEATURES if col in df.columns]
    available_categorical = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    return df[available_numeric + available_categorical]
