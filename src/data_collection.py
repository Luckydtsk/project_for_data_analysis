"""Сбор и обогащение данных London Bike Sharing.

Источники данных:
  1. Kaggle — london_merged.csv (базовый датасет TfL Cycle Hire)
  2. Open-Meteo Archive API — исторические почасовые данные погоды
  3. GOV.UK Bank Holidays API — официальные праздники Англии и Уэльса
  4. astral — астрономический расчёт времени восхода/заката (локально)
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests
from astral import LocationInfo
from astral.sun import sun

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

LONDON_LAT: float = 51.5074
LONDON_LNG: float = -0.1278

# Столбцы погоды из оригинального Kaggle-датасета, которые мы заменяем
ORIGINAL_WEATHER_COLS = ["t1", "t2", "hum", "wind_speed", "weather_code"]


# ---------------------------------------------------------------------------
# 1. Загрузка Kaggle-датасета
# ---------------------------------------------------------------------------

def download_kaggle_dataset(destination: Path) -> Path:
    """Скачивает датасет через Kaggle CLI.

    Требует настроенного ~/.kaggle/kaggle.json.
    Если Kaggle CLI недоступен, выводит инструкцию для ручной загрузки.
    """
    import subprocess

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", "hmavrodiev/london-bike-sharing-dataset",
                "-p", str(destination.parent),
                "--unzip",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        # Kaggle распаковывает в ту же директорию
        downloaded = destination.parent / "london_merged.csv"
        if downloaded.exists() and downloaded != destination:
            downloaded.rename(destination)
        return destination
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            f"Не удалось скачать датасет через Kaggle CLI: {exc}\n\n"
            "Инструкция для ручной загрузки:\n"
            "  1. Перейдите на https://www.kaggle.com/datasets/hmavrodiev/london-bike-sharing-dataset\n"
            "  2. Скачайте archive.zip и распакуйте\n"
            f"  3. Поместите 'london_merged.csv' в папку: {destination.parent}\n"
        ) from exc


def load_kaggle_dataset(path: Path | None = None) -> pd.DataFrame:
    """Загружает базовый датасет велопроката.

    Если файл отсутствует — пытается скачать через Kaggle CLI.
    """
    path = path or RAW_DATA_DIR / "london_merged.csv"
    if not path.exists():
        download_kaggle_dataset(path)
    df = pd.read_csv(path)
    return df


def drop_original_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет исходные погодные столбцы Kaggle-датасета.

    Эти столбцы будут заменены более детальными данными Open-Meteo.
    """
    cols_to_drop = [c for c in ORIGINAL_WEATHER_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop)


# ---------------------------------------------------------------------------
# 2. Open-Meteo Archive API
# ---------------------------------------------------------------------------

def fetch_openmeteo_weather(
    start_date: str,
    end_date: str,
    retries: int = 3,
    retry_delay: float = 5.0,
) -> pd.DataFrame:
    """Получает исторические почасовые данные погоды через Open-Meteo Archive API.

    API документация: https://open-meteo.com/en/docs/historical-weather-api
    Лимиты: бесплатно, без ключа, до 10 000 записей в запрос.

    Args:
        start_date: Начало периода в формате 'YYYY-MM-DD'.
        end_date:   Конец периода в формате 'YYYY-MM-DD'.
        retries:    Количество попыток при ошибке сети.
        retry_delay: Пауза между попытками (секунды).

    Returns:
        DataFrame с колонками: timestamp, temp_c, feels_like_c,
        precipitation_mm, windspeed_kmh, humidity_pct,
        cloudcover_pct, weather_code_om, visibility_m.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LONDON_LAT,
        "longitude": LONDON_LNG,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": (
            "temperature_2m,"
            "apparent_temperature,"
            "precipitation,"
            "wind_speed_10m,"
            "relative_humidity_2m,"
            "cloud_cover,"
            "weather_code,"
            "visibility"
        ),
        "timezone": "Europe/London",
    }

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=120)
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(retry_delay)
    else:
        raise RuntimeError(
            f"Open-Meteo API недоступен после {retries} попыток: {last_exc}"
        )

    data = response.json()
    h = data["hourly"]

    df = pd.DataFrame({
        "timestamp":        pd.to_datetime(h["time"]),
        "temp_c":           h["temperature_2m"],
        "feels_like_c":     h["apparent_temperature"],
        "precipitation_mm": h["precipitation"],
        "windspeed_kmh":    h["wind_speed_10m"],
        "humidity_pct":     h["relative_humidity_2m"],
        "cloudcover_pct":   h["cloud_cover"],
        "weather_code_om":  h["weather_code"],
        "visibility_m":     h["visibility"],
    })
    return df


# ---------------------------------------------------------------------------
# 3. GOV.UK Bank Holidays API
# ---------------------------------------------------------------------------

def fetch_uk_bank_holidays() -> pd.DataFrame:
    """Получает официальные праздники Англии и Уэльса через GOV.UK API.

    API: https://www.gov.uk/bank-holidays.json
    Источник: правительство Великобритании, данные открытые (OGL v3).

    Returns:
        DataFrame с колонками: date (datetime.date), holiday_name, is_bank_holiday.
    """
    url = "https://www.gov.uk/bank-holidays.json"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    data = response.json()
    events = data["england-and-wales"]["events"]

    df = pd.DataFrame(events)[["date", "title"]].rename(
        columns={"title": "holiday_name"}
    )
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["is_bank_holiday"] = True
    return df


# ---------------------------------------------------------------------------
# 4. Расчёт длины светового дня (astral)
# ---------------------------------------------------------------------------

def compute_daylight_info(dates: list) -> pd.DataFrame:
    """Вычисляет время восхода/заката и длину светового дня для Лондона.

    Использует библиотеку astral с алгоритмом USNO.
    Для каждой уникальной даты рассчитывается одна запись.

    Args:
        dates: Список объектов datetime.date.

    Returns:
        DataFrame с колонками: date, sunrise_hour, sunset_hour, daylight_hours.
    """
    london = LocationInfo(
        name="London",
        region="England",
        timezone="Europe/London",
        latitude=LONDON_LAT,
        longitude=LONDON_LNG,
    )
    records = []
    for d in dates:
        try:
            s = sun(london.observer, date=d, tzinfo=london.timezone)
            sunrise_h = s["sunrise"].hour + s["sunrise"].minute / 60
            sunset_h = s["sunset"].hour + s["sunset"].minute / 60
            daylight = (s["sunset"] - s["sunrise"]).total_seconds() / 3600
        except Exception:
            # Fallback для дат с ошибками вычисления (редко)
            sunrise_h, sunset_h, daylight = 6.5, 17.5, 11.0
        records.append({
            "date":          d,
            "sunrise_hour":  round(sunrise_h, 3),
            "sunset_hour":   round(sunset_h, 3),
            "daylight_hours": round(daylight, 3),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 5. Полный пайплайн сбора и обогащения данных
# ---------------------------------------------------------------------------

def build_enriched_dataset(
    raw_path: Path | None = None,
    processed_path: Path | None = None,
) -> pd.DataFrame:
    """Полный пайплайн: загрузка → удаление старой погоды → API-обогащение → сохранение.

    Этапы:
      1. Загружает базовый датасет Kaggle (london_merged.csv).
      2. Удаляет оригинальные погодные столбцы (t1, t2, hum, wind_speed, weather_code).
      3. Запрашивает Open-Meteo Archive API — почасовые данные погоды за 2015–2017.
      4. Запрашивает GOV.UK API — официальные праздники Англии и Уэльса.
      5. Вычисляет длину светового дня через astral.
      6. Объединяет все источники.
      7. Сохраняет обогащённый датасет в data/processed/bikes_enriched.csv.

    Returns:
        Обогащённый DataFrame.
    """
    print("=" * 60)
    print("Шаг 1: Загрузка базового датасета Kaggle")
    print("=" * 60)
    df = load_kaggle_dataset(raw_path)
    print(f"  Загружено: {df.shape[0]:,} строк, {df.shape[1]} столбцов")
    print(f"  Столбцы: {df.columns.tolist()}")

    # Парсим timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    start_date = df["timestamp"].dt.date.min().isoformat()
    end_date = df["timestamp"].dt.date.max().isoformat()
    print(f"  Период: {start_date} — {end_date}")

    print("\nШаг 2: Удаление оригинальных погодных столбцов")
    dropped = [c for c in ORIGINAL_WEATHER_COLS if c in df.columns]
    df = drop_original_weather(df)
    print(f"  Удалены: {dropped}")
    print(f"  Осталось столбцов: {df.shape[1]}")

    print("\nШаг 3: Open-Meteo Archive API — почасовые данные погоды")
    weather_df = fetch_openmeteo_weather(start_date, end_date)
    print(f"  Получено: {weather_df.shape[0]:,} почасовых записей")
    print(f"  Переменные: {weather_df.columns.drop('timestamp').tolist()}")

    # Объединяем с основным датасетом по временной метке
    df = df.merge(weather_df, on="timestamp", how="left")
    missing_weather = df["temp_c"].isna().sum()
    if missing_weather > 0:
        print(f"  Предупреждение: {missing_weather} строк без совпадения погоды")

    print("\nШаг 4: GOV.UK Bank Holidays API")
    holidays_df = fetch_uk_bank_holidays()
    print(f"  Получено: {len(holidays_df)} праздников Англии и Уэльса")

    df["date"] = df["timestamp"].dt.date
    df = df.merge(holidays_df[["date", "is_bank_holiday", "holiday_name"]],
                  on="date", how="left")
    df["is_bank_holiday"] = df["is_bank_holiday"].fillna(False)
    df["holiday_name"] = df["holiday_name"].fillna("")
    bank_hol_count = df["is_bank_holiday"].sum()
    print(f"  Часов в праздники в датасете: {bank_hol_count:,}")

    print("\nШаг 5: Расчёт длины светового дня (astral)")
    unique_dates = sorted(df["date"].unique())
    daylight_df = compute_daylight_info(unique_dates)
    daylight_df["date"] = pd.to_datetime(daylight_df["date"]).dt.date
    df = df.merge(daylight_df, on="date", how="left")
    print(f"  Рассчитано для {len(unique_dates)} уникальных дат")
    avg_daylight = df.groupby("date")["daylight_hours"].first().mean()
    print(f"  Среднее количество световых часов: {avg_daylight:.1f} ч")

    # Итоговый датасет
    df = df.drop(columns=["date"])
    print(f"\nИтого: {df.shape[0]:,} строк, {df.shape[1]} столбцов")
    print(f"Столбцы: {df.columns.tolist()}")

    # Сохранение
    processed_path = processed_path or PROCESSED_DATA_DIR / "bikes_enriched.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"\nСохранено: {processed_path}")
    print("=" * 60)

    return df


if __name__ == "__main__":
    build_enriched_dataset()
