# Предсказание одобрения кредита

Итоговый проект по дисциплине «Анализ данных на Python» (НИУ ВШЭ).

## Цель

Построить модель машинного обучения, которая по данным заявителя предсказывает, будет ли одобрена заявка на кредит.

## Источники данных

| Источник | Описание | Метод получения |
|----------|----------|-----------------|
| [Kaggle: Loan Approval Prediction](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset) | 4 269 заявок с признаками заявителя и решением банка | Загрузка CSV |
| [Банк России: ключевая ставка](https://www.cbr.ru/hd_base/KeyRate/) | Макроэкономический контекст | HTML-парсинг |
| [Банк России: Data Service API](https://www.cbr.ru/statistics/data-service/) | Средневзвешенные ставки по кредитам физлиц | REST API |

> Персональные данные реальных заявителей банки не публикуют. Для обучения модели используется открытый датасет Kaggle; макроданные ЦБ РФ обогащают записи контекстом рыночных ставок.

## Структура проекта

```
project_for_data_analysis/
├── data/
│   ├── raw/                  # Исходные данные
│   └── processed/            # Обогащённый датасет
├── notebooks/
│   └── 01_loan_approval_analysis.ipynb
├── reports/
│   └── figures/              # Графики для презентации
├── scripts/
│   └── generate_figures.py   # Генерация графиков
├── src/
│   ├── data_collection.py    # Сбор и обогащение данных
│   ├── preprocessing.py      # Очистка и feature engineering
│   └── modeling.py           # Обучение моделей
├── requirements.txt
└── README.md
```

## Быстрый старт

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Собрать и обогатить данные
python -c "from src.data_collection import build_enriched_dataset; build_enriched_dataset()"

# 3. Запустить анализ
jupyter notebook notebooks/01_loan_approval_analysis.ipynb
```

## Этапы анализа

1. **Сбор данных** — загрузка датасета Kaggle + макроданные ЦБ РФ через API
2. **Очистка** — обработка типов, выбросов, создание расчётных признаков
3. **EDA** — проверка гипотез (влияние CIBIL score, образования, самозанятости)
4. **Моделирование** — Logistic Regression, Random Forest, XGBoost
5. **Визуализация** — распределения, heatmap, ROC-кривая, confusion matrix, feature importance

## Использование AI

При защите проекта необходимо указать, какие задачи выполнялись с помощью AI-инструментов. В данном проекте AI использовался для:
- проектирования структуры репозитория;
- генерации шаблонов модулей `src/`;
- подготовки Markdown-ячеек ноутбука.

## Команда

_Укажите ФИО участников команды и распределение ролей перед загрузкой в SmartLMS._
