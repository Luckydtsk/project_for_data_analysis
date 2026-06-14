"""Очистка данных и создание признаков для проекта London Bike Sharing."""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Словари декодирования
# ---------------------------------------------------------------------------

# WMO Weather Interpretation Codes (используются Open-Meteo)
WMO_CODE_MAP: dict[int, str] = {
    0:  "ясно",
    1:  "преимущественно ясно",
    2:  "переменная облачность",
    3:  "пасмурно",
    45: "туман",
    48: "изморозевый туман",
    51: "лёгкая морось",
    53: "умеренная морось",
    55: "плотная морось",
    61: "лёгкий дождь",
    63: "умеренный дождь",
    65: "сильный дождь",
    66: "лёгкий ледяной дождь",
    67: "сильный ледяной дождь",
    71: "лёгкий снег",
    73: "умеренный снег",
    75: "сильный снег",
    77: "снежная крупа",
    80: "лёгкий ливень",
    81: "умеренный ливень",
    82: "сильный ливень",
    85: "лёгкий снегопад",
    86: "сильный снегопад",
    95: "гроза",
    96: "гроза с градом",
    99: "гроза с сильным градом",
}

# Укрупнённые категории погоды
WMO_CATEGORY_MAP: dict[int, str] = {
    0: "ясно", 1: "ясно", 2: "облачно", 3: "облачно",
    45: "туман", 48: "туман",
    51: "дождь", 53: "дождь", 55: "дождь",
    61: "дождь", 63: "дождь", 65: "дождь",
    66: "дождь", 67: "дождь",
    71: "снег",  73: "снег",  75: "снег", 77: "снег",
    80: "дождь", 81: "дождь", 82: "дождь",
    85: "снег",  86: "снег",
    95: "гроза", 96: "гроза", 99: "гроза",
}

# Сезоны из оригинального Kaggle-датасета
KAGGLE_SEASON_MAP: dict[int, str] = {
    0: "весна",
    1: "лето",
    2: "осень",
    3: "зима",
}

# Часы пик в Лондоне (рабочие дни)
MORNING_RUSH = frozenset(range(7, 10))    # 07:00–09:59
EVENING_RUSH = frozenset(range(17, 20))   # 17:00–19:59


# ---------------------------------------------------------------------------
# Основные функции очистки
# ---------------------------------------------------------------------------

def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Парсит столбец timestamp и сортирует по времени."""
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def decode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Декодирует числовые коды в читаемые категории.

    Декодирует:
    - weather_code_om (WMO-коды Open-Meteo) → weather_desc, weather_category
    - season (0–3) → season_name
    """
    out = df.copy()

    if "weather_code_om" in out.columns:
        codes = out["weather_code_om"].fillna(-1).astype(int)
        out["weather_desc"] = codes.map(WMO_CODE_MAP).fillna("неизвестно")
        out["weather_category"] = codes.map(WMO_CATEGORY_MAP).fillna("неизвестно")

    if "season" in out.columns:
        out["season_name"] = out["season"].map(KAGGLE_SEASON_MAP).fillna("неизвестно")

    return out


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет дублирующиеся временные метки."""
    n_before = len(df)
    out = df.drop_duplicates(subset=["timestamp"])
    n_dropped = n_before - len(out)
    if n_dropped > 0:
        print(f"  Удалено {n_dropped} дублирующихся строк")
    return out


def cap_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str],
    factor: float = 1.5,
) -> pd.DataFrame:
    """Ограничивает выбросы методом IQR (winsorizing).

    Значения ниже Q1 - factor*IQR заменяются на нижнюю границу,
    значения выше Q3 + factor*IQR — на верхнюю.
    """
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        q1 = out[col].quantile(0.25)
        q3 = out[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        n_clipped = ((out[col] < lower) | (out[col] > upper)).sum()
        out[col] = out[col].clip(lower=lower, upper=upper)
        if n_clipped > 0:
            print(f"  {col}: ограничено {n_clipped} значений [{lower:.1f}, {upper:.1f}]")
    return out


def fill_missing_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Заполняет пропущенные значения в погодных столбцах медианой."""
    out = df.copy()
    weather_cols = [
        "temp_c", "feels_like_c", "precipitation_mm",
        "windspeed_kmh", "humidity_pct", "cloudcover_pct", "visibility_m",
    ]
    for col in weather_cols:
        if col not in out.columns:
            continue
        n_missing = out[col].isna().sum()
        if n_missing > 0:
            median_val = out[col].median()
            out[col] = out[col].fillna(median_val)
            print(f"  {col}: заполнено {n_missing} пропусков (медиана={median_val:.2f})")

    # Осадки: пропуск = 0
    if "precipitation_mm" in out.columns:
        out["precipitation_mm"] = out["precipitation_mm"].fillna(0.0)

    return out


def clean_bike_data(df: pd.DataFrame) -> pd.DataFrame:
    """Основной этап очистки данных.

    Выполняет:
      1. Парсинг временных меток
      2. Удаление дубликатов
      3. Декодирование категориальных переменных
      4. Ограничение выбросов (IQR)
      5. Заполнение пропусков в погодных данных

    Args:
        df: Сырой обогащённый DataFrame.

    Returns:
        Очищенный DataFrame.
    """
    print("Очистка данных:")
    out = parse_timestamps(df)
    out = remove_duplicates(out)
    out = decode_categoricals(out)
    print("  Ограничение выбросов (IQR × 1.5):")
    out = cap_outliers_iqr(out, ["cnt"], factor=1.5)
    out = cap_outliers_iqr(out, [
        "temp_c", "feels_like_c", "windspeed_kmh",
        "humidity_pct", "visibility_m",
    ], factor=3.0)  # для погоды более мягкое ограничение
    print("  Заполнение пропусков:")
    out = fill_missing_weather(out)
    return out


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Создаёт производные признаки для анализа и моделирования.

    Временные признаки:
        hour, day_of_week, month, week_of_year, quarter

    Поведенческие признаки:
        is_rush_hour    — часы пик в рабочие дни (7–9, 17–19)
        is_working_hour — рабочие часы (9–17 пн–пт)
        is_night        — ночное время (22:00–05:59)
        is_daylight     — текущий час в светлое время суток

    Погодные признаки:
        temp_feels_diff   — разница ощущаемой и реальной температуры
        is_raining        — наличие осадков (>0.1 мм)
        is_heavy_rain     — сильный дождь (>2 мм)
        bad_weather_index — составной индекс плохой погоды [0, 1]
        precip_lag_1h     — осадки час назад
        precip_lag_2h     — осадки два часа назад

    Статистические признаки:
        cnt_rolling_24h   — скользящее среднее спроса за 24 ч
        cnt_rolling_168h  — скользящее среднее спроса за 7 дней
    """
    out = df.copy()

    # --- Временные признаки ---
    ts = out["timestamp"]
    out["hour"]         = ts.dt.hour
    out["day_of_week"]  = ts.dt.dayofweek   # 0 = понедельник
    out["month"]        = ts.dt.month
    out["day_of_month"] = ts.dt.day
    out["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    out["quarter"]      = ts.dt.quarter
    out["year"]         = ts.dt.year

    # --- Поведенческие признаки ---
    is_weekday = out["day_of_week"] < 5
    out["is_rush_hour"] = (
        (out["hour"].isin(MORNING_RUSH) | out["hour"].isin(EVENING_RUSH)) & is_weekday
    ).astype(int)

    out["is_working_hour"] = (
        out["hour"].between(9, 17) & is_weekday
    ).astype(int)

    out["is_night"] = out["hour"].apply(
        lambda h: 1 if h < 6 or h >= 22 else 0
    )

    # Световой день
    if "sunrise_hour" in out.columns and "sunset_hour" in out.columns:
        out["is_daylight"] = (
            (out["hour"] >= out["sunrise_hour"]) &
            (out["hour"] < out["sunset_hour"])
        ).astype(int)
    else:
        out["is_daylight"] = 0

    # --- Погодные признаки ---
    if "temp_c" in out.columns and "feels_like_c" in out.columns:
        out["temp_feels_diff"] = out["feels_like_c"] - out["temp_c"]

    if "precipitation_mm" in out.columns:
        out["is_raining"]    = (out["precipitation_mm"] > 0.1).astype(int)
        out["is_heavy_rain"] = (out["precipitation_mm"] > 2.0).astype(int)
        out["precip_lag_1h"] = out["precipitation_mm"].shift(1).fillna(0)
        out["precip_lag_2h"] = out["precipitation_mm"].shift(2).fillna(0)

    # Составной индекс плохой погоды: [0, 1], чем выше — тем хуже
    bad_weather = pd.Series(0.0, index=out.index)
    if "is_raining" in out.columns:
        bad_weather += out["is_raining"] * 0.40
    if "windspeed_kmh" in out.columns:
        bad_weather += (out["windspeed_kmh"] > 30).astype(float) * 0.30
    if "temp_c" in out.columns:
        bad_weather += (out["temp_c"] < 3).astype(float) * 0.20
    if "visibility_m" in out.columns:
        bad_weather += (out["visibility_m"] < 5000).astype(float) * 0.10
    out["bad_weather_index"] = bad_weather.round(3)

    # --- Статистические признаки ---
    # Используем shift(1), чтобы избежать утечки данных
    cnt_shifted = out["cnt"].shift(1)
    out["cnt_rolling_24h"]  = cnt_shifted.rolling(24,  min_periods=1).mean().round(1)
    out["cnt_rolling_168h"] = cnt_shifted.rolling(168, min_periods=1).mean().round(1)

    return out


# ---------------------------------------------------------------------------
# Подготовка данных для моделей
# ---------------------------------------------------------------------------

def train_test_split_ts(
    df: pd.DataFrame,
    test_hours: int = 168,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Делит временной ряд на обучающую и тестовую выборки.

    Тест = последние `test_hours` часов (по умолчанию 7 дней).
    """
    cutoff = df["timestamp"].max() - pd.Timedelta(hours=test_hours)
    train = df[df["timestamp"] <= cutoff].copy()
    test  = df[df["timestamp"] > cutoff].copy()
    print(f"  Train: {train['timestamp'].min()} — {train['timestamp'].max()} ({len(train):,} ч)")
    print(f"  Test:  {test['timestamp'].min()} — {test['timestamp'].max()} ({len(test):,} ч)")
    return train, test


def prepare_prophet_df(df: pd.DataFrame) -> pd.DataFrame:
    """Переименовывает колонки в формат Prophet (ds, y)."""
    return df[["timestamp", "cnt"]].rename(
        columns={"timestamp": "ds", "cnt": "y"}
    ).reset_index(drop=True)


def prepare_neuralforecast_df(
    df: pd.DataFrame,
    unique_id: str = "london_bikes",
) -> pd.DataFrame:
    """Готовит DataFrame для neuralforecast (unique_id, ds, y)."""
    return pd.DataFrame({
        "unique_id": unique_id,
        "ds":        df["timestamp"].values,
        "y":         df["cnt"].values,
    })
