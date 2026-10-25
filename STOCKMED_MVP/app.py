from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO
import re

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="СТОКМЕД | MVP", page_icon="📦", layout="wide")

REQUIRED = [
    "item_name", "current_stock", "avg_monthly_usage", "unit_price",
    "lead_time_days", "contract_available", "expected_delivery",
    "additional_requests", "planned_purchase_qty", "criticality"
]
OPTIONAL_PREFIX = "usage_"
COLUMN_HELP = {
    "item_name": "Наименование позиции",
    "current_stock": "Фактический остаток",
    "avg_monthly_usage": "Среднемесячный расход (если нет истории)",
    "unit_price": "Цена за единицу, ₽",
    "lead_time_days": "Срок поставки, дней",
    "contract_available": "Доступное количество по действующим договорам",
    "expected_delivery": "Подтверждённые ожидаемые поставки",
    "additional_requests": "Дозаявки/резерв, который нужно учесть",
    "planned_purchase_qty": "Дополнительные подтверждённые закупки, не учтённые в договорном остатке",
    "criticality": "Критичность: высокая / обычная",
}

def sample_data() -> pd.DataFrame:
    rows = [
        ["Парацетамол (пример)", 120, 80, 12.5, 14, 0, 0, 0, 0, "высокая",
         72, 78, 82, 75, 80, 84, 79, 86, 88, 83, 80, 80],
        ["Физраствор (пример)", 900, 500, 18, 10, 200, 100, 0, 0, "обычная",
         450, 470, 490, 510, 500, 520, 480, 510, 530, 500, 510, 500],
        ["Перчатки (пример)", 80, 150, 4.2, 21, 100, 0, 20, 0, "обычная",
         120, 130, 145, 150, 155, 140, 160, 170, 165, 150, 145, 150],
    ]
    hist_cols = [f"usage_{y}_{m:02d}" for y, m in [
        (2025, 10), (2025, 11), (2025, 12), (2026, 1), (2026, 2), (2026, 3),
        (2026, 4), (2026, 5), (2026, 6), (2026, 7), (2026, 8), (2026, 9)
    ]]
    return pd.DataFrame(rows, columns=REQUIRED + hist_cols)

def parse_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Не найдены обязательные столбцы: " + ", ".join(missing))
    hist_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"usage_\d{4}_\d{2}", c)],
        key=lambda x: x
    )
    numeric = [
        "current_stock", "avg_monthly_usage", "unit_price", "lead_time_days",
        "contract_available", "expected_delivery", "additional_requests",
        "planned_purchase_qty"
    ] + hist_cols
    df = parse_numeric(df, numeric)
    for col in numeric:
        if col not in df.columns:
            continue
        if df[col].isna().any():
            raise ValueError(f"В столбце «{col}» есть пустые или нечисловые значения. Исправьте их в исходном файле.")
        if (df[col] < 0).any():
            raise ValueError(f"В столбце «{col}» есть отрицательные значения. Проверьте исходный файл.")
    if (df["unit_price"] < 0).any():
        raise ValueError("Цена не может быть отрицательной.")
    if df["item_name"].isna().any() or (df["item_name"].astype(str).str.strip() == "").any():
        raise ValueError("У каждой позиции должно быть наименование.")
    if df["item_name"].astype(str).duplicated().any():
        raise ValueError("Есть повторяющиеся наименования. Для MVP оставьте одну строку на позицию.")
    if (df["avg_monthly_usage"] == 0).all() and not hist_cols:
        st.warning("Среднемесячный расход равен нулю и история расхода не загружена.")
    df["criticality"] = df["criticality"].astype(str).str.strip().str.lower()
    df.loc[~df["criticality"].isin(["высокая", "обычная", "критичная", "высокий", "да", "yes"]), "criticality"] = "обычная"
    return df, hist_cols

def forecast_monthly(row: pd.Series, hist_cols: list[str], horizon: int = 3) -> list[float]:
    """Simple transparent baseline: same-month history when available, otherwise recent mean."""
    history = {}
    for col in hist_cols:
        match = re.fullmatch(r"usage_(\d{4})_(\d{2})", col)
        if match:
            year, month = int(match.group(1)), int(match.group(2))
            history[(year, month)] = float(row[col])
    if not history:
        return [max(0.0, float(row["avg_monthly_usage"]))] * horizon
    latest = max(history)
    recent_values = [v for k, v in sorted(history.items())[-3:]]
    fallback = sum(recent_values) / len(recent_values) if recent_values else float(row["avg_monthly_usage"])
    results = []
    year, month = latest
    for step in range(1, horizon + 1):
        m = month + step
        y = year
        while m > 12:
            m -= 12
            y += 1
        same_month = [v for (hy, hm), v in history.items() if hm == m and (hy, hm) <= latest]
        results.append(sum(same_month) / len(same_month) if len(same_month) >= 2 else fallback)
    return results

def month_label(offset: int, hist_cols: list[str]) -> str:
    # Label forecast months relative to the latest uploaded historical month.
    if hist_cols:
        match = re.fullmatch(r"usage_(\d{4})_(\d{2})", max(hist_cols))
        if match:
            target = pd.Timestamp(year=int(match.group(1)), month=int(match.group(2)), day=1) + pd.DateOffset(months=offset)
            return target.strftime("%Y-%m")
    target = pd.Timestamp.today().normalize() + pd.DateOffset(months=offset - 1)
    return target.strftime("%Y-%m")

def calculate(df: pd.DataFrame, hist_cols: list[str], red_days: int, yellow_days: int, safety_days: int) -> pd.DataFrame:
    output = []
    for _, row in df.iterrows():
        fc = forecast_monthly(row, hist_cols, 3)
        daily = max(0.0, float(sum(fc) / 3 / 30.4375))
        stock = float(row["current_stock"])
        contract = float(row["contract_available"])
        delivery = float(row["expected_delivery"])
        requests = float(row["additional_requests"])
        planned = float(row["planned_purchase_qty"])
        # These fields must represent non-overlapping quantities to avoid double counting.
        available = max(0.0, stock + contract + delivery + planned - requests)
        days_cover = available / daily if daily > 0 else float("inf")
        shortage_date = (date.today() + timedelta(days=max(0, int(days_cover)))) if daily > 0 else None
        safety_stock = daily * safety_days
        lead = float(row["lead_time_days"])
        critical = row["criticality"] in ["высокая", "критичная", "высокий", "да", "yes"]
        if daily == 0:
            status, recommendation = "Зелёный", "не закупать"
        elif days_cover <= red_days or (critical and days_cover < lead + safety_days):
            status = "Красный"
            recommendation = "дозаказать" if contract + delivery + planned > 0 else "закупить"
        elif days_cover <= yellow_days or days_cover < lead + safety_days:
            status, recommendation = "Жёлтый", "закупить"
        else:
            status, recommendation = "Зелёный", "можно отложить"
        target_stock = daily * (lead + safety_days)
        suggested = max(0.0, target_stock - available)
        suggested_cost = suggested * float(row["unit_price"])
        output.append({
            **row.to_dict(),
            "forecast_month_1": fc[0],
            "forecast_month_2": fc[1],
            "forecast_month_3": fc[2],
            "forecast_3m": sum(fc),
            "available_total": available,
            "days_cover": days_cover if daily > 0 else None,
            "shortage_date": shortage_date.isoformat() if shortage_date else "Нет прогноза расхода",
            "safety_stock": safety_stock,
            "priority": status,
            "recommendation": recommendation,
            "suggested_qty": suggested,
            "suggested_cost": suggested_cost,
        })
    return pd.DataFrame(output)

def fmt_num(v):
    if pd.isna(v):
        return "—"
    return f"{v:,.1f}".replace(",", " ").replace(".", ",")

st.title("📦 СТОКМЕД")
st.caption("MVP 0.1 · Прогноз потребности и поддержка решений по запасам лекарственных средств")
st.info("Демонстрационная версия. Результаты не являются автоматическим решением о закупке. Используйте только разрешённые обезличенные данные; пороги и формулы необходимо согласовать с заказчиком.")

with st.sidebar:
    st.header("Настройки приоритетов")
    red_days = st.number_input("Красный при запасе до, дней", min_value=1, max_value=60, value=14)
    yellow_days = st.number_input("Жёлтый при запасе до, дней", min_value=int(red_days), max_value=180, value=30)
    safety_days = st.number_input("Страховой запас, дней расхода", min_value=1, max_value=90, value=14)
    st.caption("Начальные демонстрационные значения. Утвердите их с сотрудниками аптеки/закупок.")

template = sample_data()
csv_template = template.to_csv(index=False).encode("utf-8-sig")
st.download_button("⬇️ Скачать шаблон CSV с демонстрационными данными", csv_template, "stockmed_template.csv", "text/csv")

uploaded = st.file_uploader("Загрузите Excel (.xlsx) или CSV", type=["xlsx", "csv"])
use_demo = st.checkbox("Использовать демонстрационные данные", value=uploaded is None)

raw = None
if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            raw = pd.read_csv(uploaded, encoding="utf-8-sig")
        else:
            raw = pd.read_excel(uploaded)
    except Exception as exc:
        st.error(f"Не удалось прочитать файл: {exc}")
elif use_demo:
    raw = template.copy()

if raw is not None:
    try:
        df, hist_cols = clean_data(raw)
        result = calculate(df, hist_cols, int(red_days), int(yellow_days), int(safety_days))
        red_count = int((result["priority"] == "Красный").sum())
        yellow_count = int((result["priority"] == "Жёлтый").sum())
        green_count = int((result["priority"] == "Зелёный").sum())
        total_cost = float(result["suggested_cost"].sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Позиций", len(result))
        c2.metric("🔴 Срочно", red_count)
        c3.metric("🟡 Внимание", yellow_count)
        c4.metric("Оценка закупки", f"{total_cost:,.0f} ₽".replace(",", " "))

        st.subheader("Приоритеты и рекомендации")
        priority_filter = st.multiselect("Показать статусы", ["Красный", "Жёлтый", "Зелёный"], default=["Красный", "Жёлтый", "Зелёный"])
        view = result[result["priority"].isin(priority_filter)].copy()
        display_cols = [
            "item_name", "priority", "recommendation", "current_stock", "available_total",
            "forecast_3m", "days_cover", "shortage_date", "safety_stock", "suggested_qty", "suggested_cost"
        ]
        view = view[display_cols].rename(columns={
            "item_name": "Позиция", "priority": "Приоритет", "recommendation": "Рекомендация",
            "current_stock": "Факт. остаток", "available_total": "Расчётная обеспеченность",
            "forecast_3m": "Прогноз на 3 мес.", "days_cover": "Запас, дней",
            "shortage_date": "Оценка даты дефицита", "safety_stock": "Страховой запас",
            "suggested_qty": "Ориентир закупки, ед.", "suggested_cost": "Ориентир стоимости, ₽"
        })
        st.dataframe(view, use_container_width=True, hide_index=True)

        st.subheader("Факт расхода и прогноз")
        if hist_cols:
            selected = st.selectbox("Позиция для графика", result["item_name"].tolist())
            selected_row = result[result["item_name"] == selected].iloc[0]
            history_values = []
            for col in hist_cols:
                match = re.fullmatch(r"usage_(\d{4})_(\d{2})", col)
                if match:
                    history_values.append({"Период": f"{match.group(1)}-{match.group(2)}", "Расход": float(selected_row[col]), "Тип": "Факт"})
            forecast_rows = [
                {"Период": month_label(i, hist_cols), "Расход": float(selected_row[f"forecast_month_{i}"]), "Тип": "Прогноз"}
                for i in range(1, 4)
            ]
            chart_df = pd.DataFrame(history_values + forecast_rows)
            fig = px.line(chart_df, x="Период", y="Расход", color="Тип", markers=True, title=f"Расход: {selected}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Для графика «факт + прогноз» загрузите историю в столбцах usage_ГГГГ_ММ, например usage_2026_09.")

        st.subheader("Панель руководителя")
        a, b, c = st.columns(3)
        a.metric("Прогноз потребности на 3 месяца", f"{result['forecast_3m'].sum():,.0f}".replace(",", " ") + " ед.")
        b.metric("Ориентир финансирования", f"{total_cost:,.0f} ₽".replace(",", " "))
        c.metric("Позиций требуют действий", int(result["priority"].isin(["Красный", "Жёлтый"]).sum()))

        out = result.copy()
        out["days_cover"] = out["days_cover"].round(1)
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            out.to_excel(writer, index=False, sheet_name="СТОКМЕД")
        st.download_button("⬇️ Скачать результаты в Excel", excel_buffer.getvalue(), "stockmed_results.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.caption("Расчёты предварительные: сезонность оценивается простым базовым алгоритмом; дата дефицита рассчитана по средней скорости расхода и общей расчётной обеспеченности. Перед пилотом проверьте правила учёта поставок, договорных остатков и дозаявок с заказчиком.")
    except Exception as exc:
        st.error(f"Не удалось обработать данные: {exc}")
        st.markdown("Проверьте названия обязательных столбцов и значения. Шаблон CSV выше содержит пример структуры.")
else:
    st.markdown("Загрузите файл или включите демонстрационные данные, чтобы увидеть панель.")
