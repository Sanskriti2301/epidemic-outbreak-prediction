import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np
from pathlib import Path

# First Streamlit call (required)
st.set_page_config(page_title="Epidemic Dashboard", layout="wide")

# -----------------------------
# PATHS
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
FEATURED_DATA_PATH = DATA_DIR / "processed" / "featured_data.csv"
JHU_CONFIRMED_PATH = DATA_DIR / "time_series_covid19_confirmed_global.csv"
MODEL_PATH = REPO_ROOT / "models" / "model.pkl"


def _dedupe_country_date(df: pd.DataFrame) -> pd.DataFrame:
    """featured_data can contain duplicate Country/Region + Date rows; keep last."""
    if df.empty:
        return df
    return df.sort_values("Date").drop_duplicates(subset=["Country/Region", "Date"], keep="last")


# -----------------------------
# LOAD DATA + MODEL
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(FEATURED_DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return _dedupe_country_date(df)


@st.cache_resource
def load_model():
    model_data = joblib.load(MODEL_PATH)
    return model_data["model"], model_data["features"]


@st.cache_data
def load_country_timeseries_data():
    df = pd.read_csv(
        FEATURED_DATA_PATH,
        usecols=["Country/Region", "Date", "Cases", "Daily_Cases"],
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Country/Region", "Date"])
    return _dedupe_country_date(df)


@st.cache_data
def load_jhu_wide_confirmed():
    """
    Optional: JHU time_series_covid19_confirmed_global.csv (wide format).
    Place under data/ to use the same flow as the original second dashboard.
    """
    if not JHU_CONFIRMED_PATH.is_file():
        return None
    df = pd.read_csv(JHU_CONFIRMED_PATH)
    drop_cols = [c for c in ("Lat", "Long", "Province/State") if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    if "Country/Region" not in df.columns:
        return None
    df = df.groupby("Country/Region", as_index=True).sum(numeric_only=True)
    return df


def render_ml_dashboard():
    st.title("🦠 Epidemic Outbreak Prediction Dashboard")
    st.info("This tool predicts outbreak trends using machine learning")
    st.caption("ML-based global insights and per-country forecast.")

    try:
        df = load_data()
    except FileNotFoundError:
        st.error(f"Missing data file at `{FEATURED_DATA_PATH}`.")
        return

    model = None
    features = None
    if MODEL_PATH.exists():
        try:
            model, features = load_model()
        except Exception as e:
            st.warning(f"Model exists but failed to load: `{e}`")
    else:
        st.info(
            f"Model file not found at `{MODEL_PATH}`. "
            "Prediction-based charts and forecasts are limited until you add the trained model."
        )

    tab1, tab2 = st.tabs(["🌍 Global Insights", "📊 Country Analysis"])

    # =========================================================
    # TAB 1: GLOBAL INSIGHTS
    # =========================================================
    with tab1:
        latest_global = df.sort_values("Date").groupby("Country/Region").tail(1).copy()

        if model is not None and features is not None:
            latest_global["Predicted"] = np.expm1(model.predict(latest_global[features]))
        else:
            latest_global["Predicted"] = np.nan

        st.subheader("🌍 Outbreak Spread Over Time")

        df_anim = df.copy()
        df_anim["Date_str"] = df_anim["Date"].dt.strftime("%Y-%m-%d")

        fig_anim = px.scatter_geo(
            df_anim,
            lat="Lat",
            lon="Long",
            size="Cases",
            color="Cases",
            hover_name="Country/Region",
            animation_frame="Date_str",
            title="Global Outbreak Evolution",
            projection="natural earth",
        )

        st.plotly_chart(fig_anim, use_container_width=True)

        st.subheader("🔥 Hotspot Analysis (Growth vs Spread)")

        has_pred = latest_global["Predicted"].notna().any()
        size_col = "Predicted" if has_pred else "Cases"
        color_col = "Predicted" if has_pred else "Cases"

        fig_bubble = px.scatter(
            latest_global,
            x="Growth_Rate",
            y="Rolling_7",
            size=size_col,
            color=color_col,
            hover_name="Country/Region",
            title="Growth vs Cases vs Severity",
            labels={
                "Growth_Rate": "How Fast Cases Are Growing",
                "Rolling_7": "Average Daily Cases (Last 7 Days)",
                "Predicted": "Predicted New Cases",
            },
        )

        st.plotly_chart(fig_bubble, use_container_width=True)

        st.subheader("🔥 Top 10 High-Risk Countries")

        sort_col = "Predicted" if has_pred else "Cases"
        top10 = latest_global.sort_values(sort_col, ascending=False).head(10)

        if sort_col == "Predicted":
            st.dataframe(
                top10[["Country/Region", "Predicted"]].rename(
                    columns={"Predicted": "Predicted Daily Cases"}
                ),
                use_container_width=True,
            )
        else:
            st.dataframe(
                top10[["Country/Region", "Cases"]].rename(columns={"Cases": "Latest Total Cases"}),
                use_container_width=True,
            )

        fig_top10 = px.bar(
            top10,
            x="Country/Region",
            y=sort_col,
            title="Top 10 Countries (by risk proxy)",
        )

        st.plotly_chart(fig_top10, use_container_width=True)

    # =========================================================
    # TAB 2: COUNTRY ANALYSIS
    # =========================================================
    with tab2:
        countries = sorted(df["Country/Region"].unique())
        selected_country = st.selectbox("🌍 Select Country", countries)

        country_df = df[df["Country/Region"] == selected_country].sort_values("Date")

        st.subheader("📈 Historical Cases")

        fig = px.line(
            country_df,
            x="Date",
            y="Cases",
            title=f"Cases Trend - {selected_country}",
        )
        st.plotly_chart(fig, use_container_width=True)

        if model is None or features is None:
            st.info("Model not available; next-day prediction and 7-day forecast are disabled.")
            if "Risk_Level" in country_df.columns:
                st.subheader("🚨 Risk Level (from data)")
                st.write(f"### {country_df['Risk_Level'].iloc[-1]}")
            return

        st.subheader("🔮 Next Day Prediction")

        latest_data = country_df.iloc[-1]
        input_features = latest_data[features].values.reshape(1, -1)

        pred_log = model.predict(input_features)
        prediction = np.expm1(pred_log)[0]

        st.metric("Predicted Daily Cases (Next Day)", int(prediction))

        st.subheader("📅 7-Day Forecast")

        forecast_days = 7

        recent_values = list(country_df["Daily_Cases"].tail(7).values)

        last_row = country_df.iloc[-1]
        mobility_values = last_row[features[4:]].values

        future_predictions = []

        for _ in range(forecast_days):
            lag_1 = recent_values[-1]
            lag_7 = recent_values[0]
            rolling_7 = np.mean(recent_values)
            growth = last_row["Growth_Rate"]

            input_data = [lag_1, lag_7, rolling_7, growth, *mobility_values]

            pred_log = model.predict([input_data])[0]
            pred = np.expm1(pred_log)

            if pred < 1:
                pred = 0

            pred = int(pred)

            future_predictions.append(pred)

            recent_values.pop(0)
            recent_values.append(pred)

        future_dates = pd.date_range(
            start=last_row["Date"] + pd.Timedelta(days=1),
            periods=forecast_days,
        )

        forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Cases": future_predictions})

        fig_forecast = px.line(
            forecast_df,
            x="Date",
            y="Predicted Cases",
            title=f"7-Day Forecast - {selected_country}",
        )

        st.plotly_chart(fig_forecast, use_container_width=True)
        st.dataframe(forecast_df, use_container_width=True)

        st.subheader("🚨 Risk Level")

        if prediction > 10000:
            risk = "HIGH 🔴"
        elif prediction > 1000:
            risk = "MEDIUM 🟠"
        else:
            risk = "LOW 🟢"

        st.write(f"### {risk}")


def render_simple_timeseries_dashboard():
    st.title("AI Epidemic Spread Prediction Dashboard")
    st.info("This tool predicts outbreak trends using machine learning")

    jhu_wide = load_jhu_wide_confirmed()

    if jhu_wide is not None:
        country = st.selectbox("Select Country", jhu_wide.index)
        data = jhu_wide.loc[country]

        st.header("📊 Historical Data")
        st.line_chart(data)

        future_days = 7
        last_value = float(data.values[-1])
        predicted = [last_value + i * 1000 for i in range(1, future_days + 1)]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Cases", int(last_value))
        with col2:
            st.metric("Predicted (7 days)", int(predicted[-1]))

        st.header("🔮 Prediction")
        st.write("### Predicted Cases (Next 7 Days)")
        st.write(predicted)

        last_col = pd.to_datetime(data.index[-1])
        future_dates = pd.date_range(start=last_col, periods=8)[1:]

        pred_df = pd.DataFrame({"Date": future_dates, "Predicted Cases": predicted}).set_index(
            "Date"
        )
        st.line_chart(pred_df)

        st.write("### Risk Level")
        if last_value <= 0:
            st.info("Risk level unavailable (current cases are 0).")
        else:
            growth_rate = (predicted[-1] - last_value) / last_value * 100
            if growth_rate < 5:
                st.success("Low Risk")
            elif growth_rate < 15:
                st.warning("Medium Risk")
            else:
                st.error("High Risk")

        st.markdown("---")
        st.caption("Built for Hackathon | AI Epidemic Prediction System")
        return

    # Fallback: long-format featured_data (no JHU file in repo)
    try:
        df = load_country_timeseries_data()
    except FileNotFoundError:
        st.error(f"Missing data. Add either `{FEATURED_DATA_PATH}` or `{JHU_CONFIRMED_PATH}`.")
        return

    countries = sorted(df["Country/Region"].unique())
    country = st.selectbox("Select Country", countries)
    country_df = df[df["Country/Region"] == country].sort_values("Date")

    st.header("📊 Historical Data")
    st.line_chart(country_df.set_index("Date")[["Cases"]])

    last_value = float(country_df["Cases"].iloc[-1]) if len(country_df) else 0.0

    with st.expander("Advanced (optional)", expanded=False):
        future_days = st.slider("Forecast horizon (days)", min_value=3, max_value=30, value=7, step=1)
        daily_increment = st.number_input(
            "Assumed daily increase (simple baseline)",
            min_value=0,
            value=1000,
            step=100,
        )

    predicted = [last_value + i * float(daily_increment) for i in range(1, future_days + 1)]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Cases", int(last_value))
    with col2:
        st.metric(
            f"Predicted ({future_days} days)",
            int(predicted[-1]) if predicted else int(last_value),
        )

    st.header("🔮 Prediction")
    st.write(f"### Predicted Cases (Next {future_days} Days)")
    st.write([int(x) for x in predicted])

    last_date = pd.to_datetime(country_df["Date"].iloc[-1]) if len(country_df) else pd.Timestamp.today()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

    pred_df = pd.DataFrame({"Date": future_dates, "Predicted Cases": predicted}).set_index("Date")
    st.line_chart(pred_df)

    st.write("### Risk Level")
    if last_value <= 0:
        st.info("Risk level unavailable (current cases are 0).")
    else:
        growth_rate = (predicted[-1] - last_value) / last_value * 100
        if growth_rate < 5:
            st.success("Low Risk")
        elif growth_rate < 15:
            st.warning("Medium Risk")
        else:
            st.error("High Risk")

    st.markdown("---")
    st.caption(
        f"Using `{FEATURED_DATA_PATH.name}` (long format). "
        f"Drop JHU `time_series_covid19_confirmed_global.csv` into `data/` for the classic wide-format view."
    )
    st.caption("Built for Hackathon | AI Epidemic Prediction System")


page = st.sidebar.radio(
    "Dashboard",
    ["ML Forecast Dashboard", "Country Time-Series (Simple Projection)"],
)

if page == "ML Forecast Dashboard":
    render_ml_dashboard()
else:
    render_simple_timeseries_dashboard()
