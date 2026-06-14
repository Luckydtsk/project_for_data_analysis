"""Генерирует графики для ноутбука и презентации."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu

from src.data_collection import build_enriched_dataset
from src.modeling import get_best_tree_model_name, train_and_evaluate
from src.preprocessing import clean_loan_data, engineer_features, get_feature_matrix, prepare_target

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({"figure.figsize": (10, 6), "font.size": 11})


def main() -> None:
    _setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = build_enriched_dataset()
    cleaned = clean_loan_data(df)
    featured = engineer_features(cleaned)
    X_df, y = prepare_target(featured)
    X = get_feature_matrix(X_df)

    # 1. Распределение целевой переменной
    fig, ax = plt.subplots()
    target_counts = featured["loan_status"].value_counts()
    sns.barplot(x=target_counts.index, y=target_counts.values, ax=ax, hue=target_counts.index, legend=False)
    ax.set_title("Распределение решений по заявкам на кредит")
    ax.set_xlabel("Статус заявки")
    ax.set_ylabel("Количество заявок")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_target_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Boxplot CIBIL score
    fig, ax = plt.subplots()
    sns.boxplot(data=featured, x="loan_status", y="cibil_score", ax=ax)
    ax.set_title("CIBIL score по статусу заявки")
    ax.set_xlabel("Статус заявки")
    ax.set_ylabel("CIBIL score")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_cibil_boxplot.png", dpi=150)
    plt.close(fig)

    # 3. Heatmap корреляций
    numeric_cols = [
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
        "loan_to_income",
        "payment_to_income",
        "total_assets_value",
        "avg_consumer_loan_rate_rub",
        "key_rate",
    ]
    corr = featured[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Корреляционная матрица числовых признаков")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    # 4. Модели
    results, metrics_df, y_test, _ = train_and_evaluate(X, y)
    best_name = metrics_df.iloc[0]["model"]
    best_result = results[best_name]

    fig, ax = plt.subplots()
    sns.barplot(data=metrics_df, x="model", y="roc_auc", ax=ax, hue="model", legend=False)
    ax.set_title("Сравнение моделей по ROC-AUC")
    ax.set_xlabel("Модель")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.9, 1.01)
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_model_comparison.png", dpi=150)
    plt.close(fig)

    # 5. ROC curve
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_test, best_result.y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"{best_name} (AUC={best_result.roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Случайный классификатор")
    ax.set_title("ROC-кривая лучшей модели")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_roc_curve.png", dpi=150)
    plt.close(fig)

    # 6. Confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(
        best_result.confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Отказ", "Одобрен"],
        yticklabels=["Отказ", "Одобрен"],
        ax=ax,
    )
    ax.set_title(f"Матрица ошибок: {best_name}")
    ax.set_xlabel("Предсказание")
    ax.set_ylabel("Факт")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # 7. Feature importance
    tree_name = get_best_tree_model_name()
    if tree_name in results:
        pipeline = results[tree_name].model
        model = pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            preprocessor = pipeline.named_steps["preprocessor"]
            feature_names = preprocessor.get_feature_names_out()
            importances = pd.DataFrame(
                {"feature": feature_names, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False).head(10)

            fig, ax = plt.subplots()
            sns.barplot(data=importances, y="feature", x="importance", ax=ax, hue="feature", legend=False)
            ax.set_title(f"Важность признаков ({tree_name})")
            ax.set_xlabel("Важность")
            ax.set_ylabel("Признак")
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "07_feature_importance.png", dpi=150)
            plt.close(fig)

    metrics_df.to_csv(FIGURES_DIR / "model_metrics.csv", index=False)
    print(f"Графики сохранены в {FIGURES_DIR}")


if __name__ == "__main__":
    main()
