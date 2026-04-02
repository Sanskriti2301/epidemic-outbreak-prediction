import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

# First Streamlit call (required)
st.set_page_config(
    page_title="Epidemic Intelligence",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# PATHS
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
FEATURED_DATA_PATH = DATA_DIR / "processed" / "featured_data.csv"
JHU_CONFIRMED_PATH = DATA_DIR / "time_series_covid19_confirmed_global.csv"
MODEL_PATH = REPO_ROOT / "models" / "model.pkl"

# Brand palette (health-tech / clarity)
C_PRIMARY = "#0d9488"
C_DEEP = "#0f172a"
C_ACCENT = "#f97316"
SEQ = px.colors.sequential.Teal


def _dedupe_country_date(df: pd.DataFrame) -> pd.DataFrame:
    """featured_data can contain duplicate Country/Region + Date rows; keep last."""
    if df.empty:
        return df
    return df.sort_values("Date").drop_duplicates(subset=["Country/Region", "Date"], keep="last")


def inject_global_css() -> None:
    st.markdown(
        f"""
        <style>
        .block-container {{ padding-top: 1.25rem !important; max-width: 1280px; }}
        [data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0f172a 0%, #134e4a 100%); }}
        [data-testid="stSidebar"] * {{ color: #e2e8f0 !important; }}
        [data-testid="stSidebar"] .stMarkdown strong {{ color: #5eead4 !important; }}
        .hero-box {{
            background: linear-gradient(125deg, {C_DEEP} 0%, #134e4a 45%, {C_PRIMARY} 100%);
            padding: 1.6rem 1.75rem;
            border-radius: 14px;
            margin-bottom: 1.25rem;
            box-shadow: 0 12px 40px rgba(15, 23, 42, 0.25);
        }}
        .hero-box h1 {{
            color: #f8fafc !important;
            font-size: 1.75rem !important;
            font-weight: 700 !important;
            margin: 0 !important;
            letter-spacing: -0.02em;
        }}
        .hero-box p {{
            color: rgba(248, 250, 252, 0.88) !important;
            margin: 0.6rem 0 0 0 !important;
            font-size: 1.02rem;
            line-height: 1.45;
        }}
        .badge {{
            display: inline-block;
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-top: 0.75rem;
        }}
        .badge-ok {{ background: rgba(16, 185, 129, 0.25); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.4); }}
        .badge-warn {{ background: rgba(251, 191, 36, 0.15); color: #fcd34d; border: 1px solid rgba(251,191,36,0.35); }}
        div[data-testid="stMetric"] {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 0.75rem 1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_plotly(fig, title: str | None = None) -> None:
    layout = dict(
        template="plotly_white",
        font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=12, color="#334155"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
        margin=dict(l=48, r=24, t=56, b=48),
        hovermode="closest",
    )
    if title:
        layout["title"] = dict(text=title, font=dict(size=16, color=C_DEEP))
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor="#e2e8f0", zeroline=False)
    fig.update_yaxes(gridcolor="#e2e8f0", zeroline=False)


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
    model_ready = model is not None and features is not None

    d_min, d_max = df["Date"].min(), df["Date"].max()
    n_countries = df["Country/Region"].nunique()

    badge = (
        '<span class="badge badge-ok">Model online · forecasts active</span>'
        if model_ready
        else '<span class="badge badge-warn">Demo mode · add model.pkl for full ML</span>'
    )
    st.markdown(
        f"""
        <div class="hero-box">
            <h1>Epidemic Intelligence</h1>
            <p>Geospatial spread, hotspot dynamics, and country-level forecasts — built for decision speed.</p>
            {badge}
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Countries", f"{n_countries:,}")
    with k2:
        st.metric("Series start", d_min.strftime("%Y-%m-%d"))
    with k3:
        st.metric("Latest date", d_max.strftime("%Y-%m-%d"))
    with k4:
        st.metric("ML pipeline", "Live" if model_ready else "Partial")

    st.divider()

    tab1, tab2 = st.tabs(["Global intelligence", "Country deep-dive"])

    with tab1:
        latest_global = df.sort_values("Date").groupby("Country/Region").tail(1).copy()

        if model_ready:
            latest_global["Predicted"] = np.expm1(model.predict(latest_global[features]))
        else:
            latest_global["Predicted"] = np.nan

        st.subheader("Outbreak spread over time")
        st.caption("Animated globe — size and color reflect cumulative reported cases.")

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
            title="Global evolution",
            projection="natural earth",
            color_continuous_scale=SEQ,
        )
        style_plotly(fig_anim, str(fig_anim.layout.title.text or "Global evolution"))
        st.plotly_chart(fig_anim, use_container_width=True)

        st.subheader("Hotspot dynamics")
        st.caption("Growth vs rolling burden — bubble size reflects predicted load when the model is available.")

        has_pred = latest_global["Predicted"].notna().any()
        size_col = "Predicted" if has_pred else "Cases"

        fig_bubble = px.scatter(
            latest_global,
            x="Growth_Rate",
            y="Rolling_7",
            size=size_col,
            color=size_col,
            hover_name="Country/Region",
            title="Growth vs burden",
            labels={
                "Growth_Rate": "Growth rate",
                "Rolling_7": "7-day rolling avg (daily)",
                "Predicted": "Predicted new cases",
                "Cases": "Reported cases",
            },
            color_continuous_scale=SEQ,
        )
        style_plotly(fig_bubble, str(fig_bubble.layout.title.text or "Hotspots"))
        st.plotly_chart(fig_bubble, use_container_width=True)

        st.subheader("Priority watchlist")
        st.caption("Top 10 jurisdictions by predicted daily load or, in demo mode, by latest cumulative cases.")

        sort_col = "Predicted" if has_pred else "Cases"
        top10 = latest_global.sort_values(sort_col, ascending=False).head(10)

        if sort_col == "Predicted":
            st.dataframe(
                top10[["Country/Region", "Predicted"]].rename(
                    columns={"Predicted": "Predicted daily cases"}
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.dataframe(
                top10[["Country/Region", "Cases"]].rename(columns={"Cases": "Latest cumulative cases"}),
                use_container_width=True,
                hide_index=True,
            )

        fig_top10 = px.bar(
            top10,
            x="Country/Region",
            y=sort_col,
            title="Top 10",
            color=sort_col,
            color_continuous_scale=SEQ,
        )
        style_plotly(fig_top10, str(fig_top10.layout.title.text or "Top 10"))
        fig_top10.update_layout(showlegend=False)
        st.plotly_chart(fig_top10, use_container_width=True)

    with tab2:
        countries = sorted(df["Country/Region"].unique())
        selected_country = st.selectbox("Select country", countries, key="ml_country")

        country_df = df[df["Country/Region"] == selected_country].sort_values("Date")

        st.subheader("Cumulative cases")
        fig = px.area(
            country_df,
            x="Date",
            y="Cases",
            title=f"{selected_country}",
            color_discrete_sequence=[C_PRIMARY],
        )
        style_plotly(fig, str(fig.layout.title.text or selected_country))
        fig.update_traces(line=dict(width=2))
        st.plotly_chart(fig, use_container_width=True)

        if not model_ready:
            st.info("Train and add `models/model.pkl` to unlock next-day and 7-day ML forecasts.")
            if "Risk_Level" in country_df.columns:
                st.subheader("Risk label (from dataset)")
                st.write(f"### {country_df['Risk_Level'].iloc[-1]}")
        else:
            st.subheader("Next-day prediction")
            latest_data = country_df.iloc[-1]
            input_features = latest_data[features].values.reshape(1, -1)
            pred_log = model.predict(input_features)
            prediction = np.expm1(pred_log)[0]
            st.metric("Predicted daily cases (next day)", f"{int(prediction):,}")

            st.subheader("7-day rolling forecast")
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

            fig_ff = go.Figure()
            fig_ff.add_trace(
                go.Scatter(
                    x=forecast_df["Date"],
                    y=forecast_df["Predicted Cases"],
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(color=C_ACCENT, width=3),
                    fill="tozeroy",
                    fillcolor="rgba(249, 115, 22, 0.12)",
                )
            )
            style_plotly(fig_ff, f"7-day outlook — {selected_country}")
            st.plotly_chart(fig_ff, use_container_width=True)
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)

            st.subheader("Risk band")
            if prediction > 10000:
                risk, hue = "HIGH", "🔴"
            elif prediction > 1000:
                risk, hue = "MEDIUM", "🟠"
            else:
                risk, hue = "LOW", "🟢"
            st.markdown(f"### {risk} {hue}")


def render_simple_timeseries_dashboard():
    st.markdown(
        """
        <div class="hero-box">
            <h1>Rapid scenario lab</h1>
            <p>Baseline trajectory view — explore a country curve and stress-test a simple growth assumption.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    jhu_wide = load_jhu_wide_confirmed()

    if jhu_wide is not None:
        country = st.selectbox("Country", jhu_wide.index, key="ts_jhu_country")
        data = jhu_wide.loc[country]

        st.subheader("Confirmed cases (cumulative)")
        hist_df = pd.DataFrame({"Date": pd.to_datetime(data.index), "Cases": data.values})
        fig_h = px.line(
            hist_df,
            x="Date",
            y="Cases",
            title=f"{country} — historical",
            color_discrete_sequence=[C_PRIMARY],
        )
        style_plotly(fig_h, str(fig_h.layout.title.text or country))
        fig_h.update_traces(line=dict(width=2.5))
        st.plotly_chart(fig_h, use_container_width=True)

        future_days = 7
        last_value = float(data.values[-1])
        predicted = [last_value + i * 1000 for i in range(1, future_days + 1)]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Latest cumulative", f"{int(last_value):,}")
        with c2:
            st.metric("7-day projection (baseline)", f"{int(predicted[-1]):,}")
        with c3:
            delta_pct = ((predicted[-1] - last_value) / last_value * 100) if last_value else 0.0
            st.metric("Implied growth vs today", f"{delta_pct:.1f}%")

        st.subheader("Baseline projection")
        last_col = pd.to_datetime(data.index[-1])
        future_dates = pd.date_range(start=last_col, periods=8)[1:]
        pred_df = pd.DataFrame({"Date": future_dates, "Projected": predicted})
        fig_p = px.area(
            pred_df,
            x="Date",
            y="Projected",
            title="Next 7 days (linear ramp)",
            color_discrete_sequence=[C_ACCENT],
        )
        style_plotly(fig_p, str(fig_p.layout.title.text or "Projection"))
        fig_p.update_traces(line=dict(width=2))
        st.plotly_chart(fig_p, use_container_width=True)

        st.subheader("Risk signal")
        if last_value <= 0:
            st.info("Insufficient data for risk ratio.")
        else:
            growth_rate = (predicted[-1] - last_value) / last_value * 100
            if growth_rate < 5:
                st.success("Low")
            elif growth_rate < 15:
                st.warning("Medium")
            else:
                st.error("High")

        st.caption("Epidemic Intelligence · scenario prototype")
        return

    try:
        df = load_country_timeseries_data()
    except FileNotFoundError:
        st.error(f"Missing data. Add `{FEATURED_DATA_PATH}` or `{JHU_CONFIRMED_PATH}`.")
        return

    countries = sorted(df["Country/Region"].unique())
    country = st.selectbox("Country", countries, key="ts_long_country")
    country_df = df[df["Country/Region"] == country].sort_values("Date")

    st.subheader("Confirmed cases (cumulative)")
    fig_hist = px.area(
        country_df,
        x="Date",
        y="Cases",
        title=f"{country}",
        color_discrete_sequence=[C_PRIMARY],
    )
    style_plotly(fig_hist, str(fig_hist.layout.title.text or country))
    st.plotly_chart(fig_hist, use_container_width=True)

    last_value = float(country_df["Cases"].iloc[-1]) if len(country_df) else 0.0

    with st.expander("Scenario controls", expanded=False):
        future_days = st.slider("Horizon (days)", 3, 30, 7, 1)
        daily_increment = st.number_input("Daily increment (baseline)", min_value=0, value=1000, step=100)

    predicted = [last_value + i * float(daily_increment) for i in range(1, future_days + 1)]

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Latest cumulative", f"{int(last_value):,}")
    with m2:
        st.metric(f"End of horizon ({future_days}d)", f"{int(predicted[-1]):,}")
    with m3:
        if last_value > 0:
            st.metric(
                "Implied growth",
                f"{((predicted[-1] - last_value) / last_value * 100):.1f}%",
            )
        else:
            st.metric("Implied growth", "—")

    st.subheader("Projected curve")
    last_date = pd.to_datetime(country_df["Date"].iloc[-1]) if len(country_df) else pd.Timestamp.today()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
    pred_df = pd.DataFrame({"Date": future_dates, "Projected": predicted})
    fig_proj = px.line(
        pred_df,
        x="Date",
        y="Projected",
        title="Baseline projection",
        color_discrete_sequence=[C_ACCENT],
    )
    style_plotly(fig_proj, str(fig_proj.layout.title.text or "Projection"))
    fig_proj.update_traces(line=dict(width=3), fill="tozeroy", fillcolor="rgba(249, 115, 22, 0.08)")
    st.plotly_chart(fig_proj, use_container_width=True)

    st.subheader("Risk signal")
    if last_value <= 0:
        st.info("Insufficient data for risk ratio.")
    else:
        growth_rate = (predicted[-1] - last_value) / last_value * 100
        if growth_rate < 5:
            st.success("Low")
        elif growth_rate < 15:
            st.warning("Medium")
        else:
            st.error("High")

    st.caption(
        f"Using `{FEATURED_DATA_PATH.name}` · Add JHU `time_series_covid19_confirmed_global.csv` under `data/` for wide-format import."
    )


def main():
    inject_global_css()
    st.sidebar.markdown("### Epidemic IQ")
    st.sidebar.markdown(
        "<span style='opacity:0.85;font-size:0.9rem;'>Geo risk · ML forecasts · scenario lab</span>",
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigate",
        [
            "Intelligence (ML)",
            "Scenario lab",
        ],
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.caption("Epidemic Intelligence prototype")

    if page == "Intelligence (ML)":
        render_ml_dashboard()
    else:
        render_simple_timeseries_dashboard()


main()
