from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Прогноз вигорання", page_icon="🔥", layout="wide")

BASE_DIR = Path(__file__).parent

# Ознаки, на яких навчалася модель у ЛР1 (порядок як у ColumnTransformer)
NUMERIC_FEATURES = ["task_completion_rate", "work_hours", "meetings_count", "non_work_screen_time"]
CATEGORICAL_FEATURES = ["day_type"]

# Діапазони ознак у навчальному наборі даних
RANGES = {
    "work_hours": (3.0, 12.17),
    "non_work_screen_time": (1.5, 4.0),
    "meetings_count": (0, 10),
    "task_completion_rate": (40.0, 100.0),
}

RISK = {
    "Low": ("Низький", "#2e7d32", "Ризик вигорання низький. Поточний режим роботи збалансований."),
    "Medium": ("Середній", "#ef6c00", "Помітні ознаки перевантаження. Варто переглянути обсяг задач, кількість зустрічей і тривалість роботи."),
    "High": ("Високий", "#c62828", "Високий ризик вигорання. Потрібно переглянути навантаження та режим відпочинку."),
}

EXAMPLES = {
    "Власні значення": None,
    "Збалансований працівник": dict(day_type="Weekday", work_hours=6.0, screen_time_hours=8.5,
                                    meetings_count=2, task_completion_rate=88.0),
    "Перевантажений працівник": dict(day_type="Weekday", work_hours=11.2, screen_time_hours=14.5,
                                     meetings_count=7, task_completion_rate=45.0),
    "Вихідний день": dict(day_type="Weekend", work_hours=3.5, screen_time_hours=6.5,
                          meetings_count=0, task_completion_rate=95.0),
}

DEFAULTS = dict(day_type="Weekday", work_hours=8.0, screen_time_hours=11.0, meetings_count=3, task_completion_rate=75.0)


@st.cache_resource
def load_model_and_resources():
    model = load_model(BASE_DIR / "burnout_model.keras", compile=False)
    preprocessor = joblib.load(BASE_DIR / "preprocessor.pkl")
    return model, preprocessor


def burnout_category(score):
    if score < 70:
        return "Low"
    elif score < 110:
        return "Medium"
    return "High"


def build_features(df):
    """Ті самі перетворення, що й під час навчання моделі в ЛР1."""
    df = df.copy()
    df["task_completion_rate"] = df["task_completion_rate"].clip(upper=100)
    df["non_work_screen_time"] = df["screen_time_hours"] - df["work_hours"]
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def predict(df):
    X = build_features(df)
    return model.predict(preprocessor.transform(X), verbose=0).ravel()


def init_state():
    """Початкові значення: із параметрів URL (?work_hours=11&...) або типові."""
    if "initialized" in st.session_state:
        return
    qp = st.query_params
    for key, default in DEFAULTS.items():
        value = qp.get(key, default)
        st.session_state[key] = type(default)(value) if key != "day_type" else str(value)
    st.session_state["initialized"] = True


def apply_example():
    example = EXAMPLES[st.session_state["example"]]
    if example:
        for key, value in example.items():
            st.session_state[key] = value


model, preprocessor = load_model_and_resources()
init_state()

# ---------------- Заголовок ----------------
st.title("🔥 Прогноз професійного вигорання")
st.markdown(
    "Застосунок оцінює рівень професійного вигорання (**burnout_score**) працівника, "
    "який працює з дому, за показниками одного робочого дня."
)

st.selectbox("Заповнити приклад", list(EXAMPLES), key="example", on_change=apply_example)

col_in, col_out = st.columns([1, 1], gap="large")

with col_in:
    st.subheader("Вхідні дані")
    st.radio("Тип дня", ["Weekday", "Weekend"], key="day_type", horizontal=True,
             format_func=lambda v: "Будній" if v == "Weekday" else "Вихідний")
    st.slider("Робочі години за день", 0.0, 16.0, step=0.1, key="work_hours")
    st.slider("Загальний екранний час, год", 0.0, 20.0, step=0.1, key="screen_time_hours",
              help="Час перед екраном за добу, включно з робочим.")
    st.number_input("Кількість зустрічей", min_value=0, max_value=20, step=1, key="meetings_count")
    st.slider("Виконано задач, %", 0.0, 100.0, step=1.0, key="task_completion_rate")

values = {k: st.session_state[k] for k in DEFAULTS}
input_df = pd.DataFrame({k: [v] for k, v in values.items()})
features = build_features(input_df)
non_work = features["non_work_screen_time"].iloc[0]

with col_out:
    st.subheader("Результат")
    if non_work < 0:
        st.error("Екранний час не може бути меншим за робочі години: під час роботи з дому "
                 "працівник працює за комп'ютером.")
    else:
        score = float(predict(input_df)[0])
        risk = burnout_category(score)
        label, color, advice = RISK[risk]

        m1, m2 = st.columns(2)
        m1.metric("Рівень вигорання (burnout_score)", f"{score:.1f}")
        m2.markdown(
            f"<div style='padding-top:6px'>Категорія ризику<br>"
            f"<span style='font-size:1.9rem;font-weight:700;color:{color}'>{label}</span></div>",
            unsafe_allow_html=True,
        )
        st.progress(min(max(score / 150, 0.0), 1.0), text="Шкала 0–150 балів")
        {"Low": st.success, "Medium": st.warning, "High": st.error}[risk](advice)

        out_of_range = [
            name for name, col in [("робочі години", "work_hours"),
                                   ("екранний час поза роботою", "non_work_screen_time"),
                                   ("кількість зустрічей", "meetings_count"),
                                   ("виконано задач", "task_completion_rate")]
            if not RANGES[col][0] <= features[col].iloc[0] <= RANGES[col][1]
        ]
        if out_of_range:
            st.info("Значення поза діапазоном навчальних даних (" + ", ".join(out_of_range) +
                    "). Прогноз є екстраполяцією і може бути менш точним.")

if non_work >= 0:
    st.subheader("Як змінюється прогноз при зміні одного показника")
    sweep = {
        "Виконано задач, %": ("task_completion_rate", np.arange(40, 100.5, 1.0)),
        "Робочі години за день": ("work_hours", np.arange(3.0, 12.25, 0.25)),
        "Кількість зустрічей": ("meetings_count", np.arange(0, 11, 1)),
        "Екранний час поза роботою, год": ("non_work_screen_time", np.arange(1.5, 4.05, 0.1)),
    }
    sweep_label = st.selectbox("Показник", list(sweep))
    sweep_col, sweep_values = sweep[sweep_label]
    grid = pd.DataFrame({k: [v] * len(sweep_values) for k, v in values.items()})
    if sweep_col == "work_hours":
        grid["work_hours"] = sweep_values
        grid["screen_time_hours"] = sweep_values + non_work   # екранний час поза роботою не змінюється
    elif sweep_col == "non_work_screen_time":
        grid["screen_time_hours"] = values["work_hours"] + sweep_values
    else:
        grid[sweep_col] = sweep_values
    grid["x"] = sweep_values
    grid["burnout_score"] = predict(grid)
    current_x = non_work if sweep_col == "non_work_screen_time" else values[sweep_col]

    bands = pd.DataFrame({"from": [0, 70, 110], "to": [70, 110, 160],
                          "Ризик": ["Низький", "Середній", "Високий"]})
    band_chart = alt.Chart(bands).mark_rect(opacity=0.12).encode(
        y="from:Q", y2="to:Q",
        color=alt.Color("Ризик:N", scale=alt.Scale(domain=["Низький", "Середній", "Високий"],
                                                   range=["#2e7d32", "#ef6c00", "#c62828"])))
    line = alt.Chart(grid).mark_line(color="#1f77b4", strokeWidth=3).encode(
        x=alt.X("x:Q", title=sweep_label),
        y=alt.Y("burnout_score:Q", title="burnout_score", scale=alt.Scale(domain=[0, 160])))
    point = alt.Chart(pd.DataFrame({"x": [current_x], "burnout_score": [score]})) \
        .mark_point(size=160, filled=True, color="black").encode(x="x:Q", y="burnout_score:Q")
    st.altair_chart(band_chart + line + point, use_container_width=True)
    st.caption("Інші показники зафіксовано на введених значеннях; чорна точка — поточний прогноз.")

    with st.expander("Ознаки, передані в модель"):
        st.markdown("Вхідні ознаки після інженерії ознак (як у ЛР1):")
        st.dataframe(features, hide_index=True)
        st.markdown("Після масштабування (StandardScaler) та кодування (OneHotEncoder):")
        st.dataframe(pd.DataFrame(preprocessor.transform(features),
                                  columns=preprocessor.get_feature_names_out()).round(3), hide_index=True)
