"""Обучение и сравнение моделей прогнозирования велопроката: Prophet и PatchTST.

Модели:
    Prophet  — байесовская аддитивная модель (Meta, 2017), обрабатывает
               множественные сезонности и праздники.
    PatchTST — трансформерная модель для временных рядов (Nie et al., 2023),
               разбивает ряд на патчи как Vision Transformer.
    Baseline — наивный прогноз: для каждого часа берётся среднее значение
               в эту же комбинацию час+день недели на обучающей выборке.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import PatchTST
    NEURALFORECAST_AVAILABLE = True
except ImportError:
    NEURALFORECAST_AVAILABLE = False


# ---------------------------------------------------------------------------
# Структура результата
# ---------------------------------------------------------------------------

@dataclass
class ForecastResult:
    """Результат прогнозирования одной модели."""
    model_name: str
    predictions: pd.Series          # индекс = timestamp
    y_true: pd.Series                # индекс = timestamp
    mae:  float = field(init=False)
    rmse: float = field(init=False)
    mape: float = field(init=False)

    def __post_init__(self) -> None:
        self.mae, self.rmse, self.mape = _compute_metrics(
            self.y_true, self.predictions
        )

    def summary(self) -> dict[str, float | str]:
        return {
            "Модель":   self.model_name,
            "MAE":      round(self.mae,  2),
            "RMSE":     round(self.rmse, 2),
            "MAPE (%)": round(self.mape, 2),
        }


# ---------------------------------------------------------------------------
# Вспомогательная функция метрик
# ---------------------------------------------------------------------------

def _compute_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> tuple[float, float, float]:
    """Вычисляет MAE, RMSE, MAPE."""
    mask = y_true.notna() & y_pred.notna() & (y_true > 0)
    yt, yp = y_true[mask], y_pred[mask]

    mae  = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mape = float(np.mean(np.abs((yt - yp) / yt)) * 100)
    return mae, rmse, mape


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def train_baseline(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
) -> ForecastResult:
    """Наивный базовый прогноз: среднее по часу и дню недели.

    Для каждого часа в тестовой выборке предсказывается среднее значение
    cnt за тот же час (±1 ч) и день недели на обучающей выборке.
    """
    ref = train_df.copy()
    ref["hour"] = ref["timestamp"].dt.hour
    ref["dow"]  = ref["timestamp"].dt.dayofweek
    means = ref.groupby(["dow", "hour"])["cnt"].mean()

    preds = []
    for _, row in test_df.iterrows():
        h   = row["timestamp"].hour
        dow = row["timestamp"].dayofweek
        val = means.get((dow, h), train_df["cnt"].mean())
        preds.append(val)

    predictions = pd.Series(preds, index=test_df["timestamp"])
    y_true      = test_df.set_index("timestamp")["cnt"]

    return ForecastResult(
        model_name="Baseline (avg по часу/дню)",
        predictions=predictions,
        y_true=y_true,
    )


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------

def build_prophet_holidays(holidays_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Готовит DataFrame праздников в формате Prophet.

    Args:
        holidays_df: DataFrame с колонками date и holiday_name
                     (результат fetch_uk_bank_holidays).

    Returns:
        DataFrame с колонками: holiday, ds, lower_window, upper_window.
    """
    if holidays_df is None or holidays_df.empty:
        return None
    ph = holidays_df[["date", "holiday_name"]].copy()
    ph["ds"] = pd.to_datetime(ph["date"])
    ph["holiday"] = ph["holiday_name"].fillna("bank_holiday")
    ph["lower_window"] = 0
    ph["upper_window"] = 1
    return ph[["holiday", "ds", "lower_window", "upper_window"]]


def train_prophet(
    train_df:    pd.DataFrame,
    test_df:     pd.DataFrame,
    holidays_df: pd.DataFrame | None = None,
) -> ForecastResult:
    """Обучает модель Prophet и прогнозирует тестовый период.

    Особенности настройки:
        - seasonality_mode='multiplicative': велопрокат имеет мультипликативную
          сезонность (летом не просто больше, а кратно больше)
        - Включены три уровня сезонности: годовая, еженедельная, ежедневная
        - bank holidays переданы как внешний регрессор сезонности

    Args:
        train_df:    DataFrame с колонками timestamp, cnt.
        test_df:     DataFrame с колонками timestamp, cnt.
        holidays_df: DataFrame праздников (optional).

    Returns:
        ForecastResult с предсказаниями и метриками.
    """
    if not PROPHET_AVAILABLE:
        raise ImportError(
            "Prophet не установлен. Выполните: pip install prophet"
        )

    ph_holidays = build_prophet_holidays(holidays_df)

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        holidays=ph_holidays,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
    )

    # Обучение
    train_prophet_df = train_df[["timestamp", "cnt"]].rename(
        columns={"timestamp": "ds", "cnt": "y"}
    )
    m.fit(train_prophet_df)

    # Прогноз
    future = test_df[["timestamp"]].rename(columns={"timestamp": "ds"})
    forecast = m.predict(future)

    predictions = forecast.set_index("ds")["yhat"].clip(lower=0)
    y_true      = test_df.set_index("timestamp")["cnt"]

    return ForecastResult(
        model_name="Prophet",
        predictions=predictions,
        y_true=y_true,
    ), m, forecast  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# PatchTST
# ---------------------------------------------------------------------------

def train_patchtst(
    train_df:          pd.DataFrame,
    test_df:           pd.DataFrame,
    context_length:    int = 336,   # 2 недели контекста
    prediction_length: int = 168,   # 1 неделя прогноза
    max_steps:         int = 500,
    learning_rate:     float = 1e-4,
    batch_size:        int = 32,
    random_seed:       int = 42,
) -> ForecastResult:
    """Обучает PatchTST через библиотеку neuralforecast.

    PatchTST (Nie et al., 2023) делит исторический ряд на непересекающиеся
    «патчи» фиксированной длины и обрабатывает их как последовательность токенов
    для трансформера — аналог подхода Vision Transformer для изображений.
    Патчи сохраняют локальную семантику (суточные паттерны) и сокращают
    длину последовательности, что делает обучение эффективнее.

    Args:
        train_df:          DataFrame с колонками timestamp, cnt.
        test_df:           DataFrame с колонками timestamp, cnt.
        context_length:    Длина окна контекста (в часах).
        prediction_length: Горизонт прогноза (в часах). Должен совпадать
                           с длиной test_df.
        max_steps:         Количество шагов обучения.
        learning_rate:     Скорость обучения.
        batch_size:        Размер батча.
        random_seed:       Зерно генератора случайных чисел.

    Returns:
        ForecastResult с предсказаниями и метриками.
    """
    if not NEURALFORECAST_AVAILABLE:
        raise ImportError(
            "neuralforecast не установлен. "
            "Выполните: pip install neuralforecast"
        )

    # Формат neuralforecast
    nf_train = pd.DataFrame({
        "unique_id": "london_bikes",
        "ds":        pd.to_datetime(train_df["timestamp"]),
        "y":         train_df["cnt"].values,
    })

    actual_h = len(test_df)

    model = PatchTST(
        h=actual_h,
        input_size=context_length,
        patch_len=24,           # суточный патч
        stride=12,              # полусуточный сдвиг
        d_model=128,
        n_heads=8,
        e_layers=3,
        d_ff=256,
        dropout=0.1,
        max_steps=max_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        scaler_type="standard",
        random_seed=random_seed,
        enable_progress_bar=True,
    )

    # Определяем частоту
    freq = "h"
    nf = NeuralForecast(models=[model], freq=freq)
    nf.fit(nf_train)

    raw_preds = nf.predict()

    # Извлекаем предсказания
    pred_values = raw_preds["PatchTST"].values.clip(min=0)

    # Индекс = временные метки тестового периода
    test_timestamps = pd.to_datetime(test_df["timestamp"])
    if len(pred_values) >= len(test_timestamps):
        pred_values = pred_values[: len(test_timestamps)]
    else:
        # Дополняем, если предсказаний меньше
        pad = np.full(len(test_timestamps) - len(pred_values), np.nan)
        pred_values = np.concatenate([pred_values, pad])

    predictions = pd.Series(pred_values, index=test_timestamps)
    y_true      = test_df.set_index("timestamp")["cnt"]

    return ForecastResult(
        model_name="PatchTST",
        predictions=predictions,
        y_true=y_true,
    ), nf  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Сравнение моделей
# ---------------------------------------------------------------------------

def compare_models(results: list[ForecastResult]) -> pd.DataFrame:
    """Создаёт таблицу сравнения моделей по метрикам.

    Args:
        results: Список ForecastResult от каждой модели.

    Returns:
        DataFrame, отсортированный по RMSE (ascending).
    """
    rows = [r.summary() for r in results]
    df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    df.index += 1  # нумерация с 1
    return df
