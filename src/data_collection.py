"""Загрузка датасета заявок на кредит и макроданных Банка России."""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
LOAN_DATASET_URL = (
    "https://raw.githubusercontent.com/AbhishekBiswas-github/"
    "AI-Engineer-Projects/refs/heads/main/Machine-Learning/"
    "Loan-Approval-Prediction/loan_approval_dataset.csv"
)
CBR_BASE_URL = "https://www.cbr.ru/dataservice"
def download_loan_dataset(
    destination: Path | None = None,
    force: bool = False,
) -> Path:
    """Скачивает датасет Loan Approval с Kaggle (зеркало GitHub)."""
    destination = destination or RAW_DATA_DIR / "loan_approval_dataset.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        return destination

    response = requests.get(LOAN_DATASET_URL, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def load_loan_dataset(path: Path | None = None) -> pd.DataFrame:
    """Загружает датасет заявок на кредит и нормализует названия колонок."""
    path = path or RAW_DATA_DIR / "loan_approval_dataset.csv"
    if not path.exists():
        download_loan_dataset(path)

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def fetch_cbr_key_rate(
    year_from: int = 2020,
    year_to: int = 2024,
) -> pd.DataFrame:
    """Получает историю ключевой ставки ЦБ РФ через SOAP API DailyInfo."""
    soap_url = "https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx"
    headers = {
        "Content-Type": "text/xml; charset=utf-8",
        "SOAPAction": "http://web.cbr.ru/KeyRateXML",
    }
    body = f"""<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
  <soap:Body>
    <KeyRateXML xmlns="http://web.cbr.ru/">
      <fromDate>{year_from}-01-01T00:00:00</fromDate>
      <ToDate>{year_to}-12-31T00:00:00</ToDate>
    </KeyRateXML>
  </soap:Body>
</soap:Envelope>"""

    response = requests.post(soap_url, data=body.encode("utf-8"), headers=headers, timeout=60)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    rows = []
    for kr in root.iter():
        if not kr.tag.endswith("KR"):
            continue
        date_value = rate_value = None
        for child in kr:
            if child.tag.endswith("DT"):
                date_value = child.text
            elif child.tag.endswith("Rate"):
                rate_value = child.text
        if date_value and rate_value:
            rows.append(
                {
                    "date": pd.to_datetime(date_value),
                    "key_rate": float(rate_value),
                }
            )

    if not rows:
        raise ValueError("SOAP API ЦБ не вернул данные по ключевой ставке.")

    df = pd.DataFrame(rows).drop_duplicates().sort_values("date")
    if hasattr(df["date"].dt, "tz") and df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df["year_month"] = df["date"].dt.to_period("M").map(str)
    return df


def fetch_cbr_consumer_loan_rates(
    year_from: int = 2020,
    year_to: int = 2024,
) -> pd.DataFrame:
    """
    Получает средневзвешенные ставки по кредитам физлиц через API ЦБ РФ.

    Показатель: «Ставки по кредитам физическим лицам» (indicator_id=27),
    категория 14 — «В целом по Российской Федерации».
    """
    params = {
        "categoryId": 14,
        "y1": year_from,
        "y2": year_to,
        "iIds": 27,
    }
    response = requests.get(f"{CBR_BASE_URL}/dataNew", params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()

    links = {
        (
            link["indicator_id"],
            link.get("measure1_id"),
            link.get("measure2_id"),
            link.get("unit_id"),
        ): link
        for link in payload.get("Links", [])
    }

    records = []
    for row in payload.get("RowData", []):
        if row.get("obs_val") is None:
            continue

        meta = links.get(
            (
                row["indicator_id"],
                row.get("measure1_id"),
                row.get("measure2_id"),
                row.get("unit_id"),
            ),
            {},
        )
        records.append(
            {
                "date": pd.to_datetime(row["date"]),
                "year_month": str(pd.to_datetime(row["date"]).to_period("M")),
                "consumer_loan_rate": row["obs_val"],
                "currency": meta.get("measure1_name", "Не указано"),
                "loan_term_bucket": meta.get("measure2_name", "Не указано"),
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        return df

    rub_rates = df[df["currency"].str.contains("руб", case=False, na=False)].copy()
    monthly_avg = (
        rub_rates.groupby("year_month", as_index=False)["consumer_loan_rate"]
        .mean()
        .rename(columns={"consumer_loan_rate": "avg_consumer_loan_rate_rub"})
    )
    return monthly_avg.sort_values("year_month")


def enrich_with_macro_data(
    loans: pd.DataFrame,
    key_rate: pd.DataFrame | None = None,
    consumer_rates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Обогащает заявки макропоказателями ЦБ РФ.

    У датасета нет даты заявки, поэтому для каждой записи назначается
    синтетический месяц подачи (детерминированно по loan_id), после чего
    подтягиваются ключевая ставка и средняя ставка по кредитам физлиц.
    """
    enriched = loans.copy()
    available_months = sorted(consumer_rates["year_month"].unique())
    enriched["application_month"] = enriched["loan_id"].apply(
        lambda loan_id: available_months[loan_id % len(available_months)]
    )

    key_rate_monthly = (
        key_rate.assign(year_month=key_rate["year_month"])
        .groupby("year_month", as_index=False)["key_rate"]
        .mean()
    )

    enriched = enriched.merge(consumer_rates, left_on="application_month", right_on="year_month", how="left")
    enriched = enriched.merge(
        key_rate_monthly,
        left_on="application_month",
        right_on="year_month",
        how="left",
        suffixes=("", "_key"),
    )
    enriched = enriched.drop(columns=[col for col in enriched.columns if col.startswith("year_month")])
    return enriched


def save_macro_data(
    key_rate: pd.DataFrame,
    consumer_rates: pd.DataFrame,
    output_dir: Path | None = None,
) -> None:
    """Сохраняет макроданные ЦБ в CSV и JSON."""
    output_dir = output_dir or RAW_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    key_rate.to_csv(output_dir / "cbr_key_rate.csv", index=False)
    consumer_rates.to_csv(output_dir / "cbr_consumer_loan_rates.csv", index=False)

    metadata = {
        "sources": [
            {
                "name": "Ключевая ставка Банка России",
                "url": "https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx",
                "method": "SOAP API (KeyRateXML)",
            },
            {
                "name": "Ставки по кредитам физическим лицам",
                "url": f"{CBR_BASE_URL}/dataNew",
                "method": "REST API (categoryId=14, indicatorId=27)",
            },
        ]
    }
    (output_dir / "macro_sources.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_enriched_dataset(
    loan_path: Path | None = None,
    processed_path: Path | None = None,
) -> pd.DataFrame:
    """Полный пайплайн сбора и обогащения данных."""
    loans = load_loan_dataset(loan_path)
    key_rate = fetch_cbr_key_rate(year_from=2020, year_to=2024)
    consumer_rates = fetch_cbr_consumer_loan_rates(year_from=2020, year_to=2024)
    save_macro_data(key_rate, consumer_rates)

    enriched = enrich_with_macro_data(loans, key_rate, consumer_rates)
    processed_path = processed_path or PROJECT_ROOT / "data" / "processed" / "loans_enriched.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(processed_path, index=False)
    return enriched
