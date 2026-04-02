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

# Design system — professional health / analytics
C_PRIMARY = "#0d9488"
C_PRIMARY_LIGHT = "#14b8a6"
C_DEEP = "#0f172a"
C_SLATE = "#334155"
C_BODY = "#1e293b"
C_MUTED = "#475569"
C_ACCENT = "#ea580c"
C_SURFACE = "#ffffff"
C_PAGE_TOP = "#ecfdf5"
C_PAGE_MID = "#f0fdfa"
C_PAGE_BOTTOM = "#f8fafc"
C_BORDER = "#e2e8f0"
# Custom Plotly sequential (teal depth — readable on white)
CSCALE = [
    [0.0, "#ccfbf1"],
    [0.25, "#99f6e4"],
    [0.5, "#2dd4bf"],
    [0.75, "#0d9488"],
    [1.0, "#115e59"],
]


def _dedupe_country_date(df: pd.DataFrame) -> pd.DataFrame:
    """featured_data can contain duplicate Country/Region + Date rows; keep last."""
    if df.empty:
        return df
    return df.sort_values("Date").drop_duplicates(subset=["Country/Region", "Date"], keep="last")


def inject_global_css() -> None:
    st.markdown(
        f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap" rel="stylesheet">
        <style>
        html, body, [class*="css"] {{
            font-family: 'DM Sans', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
        }}
        .stApp {{
            background: linear-gradient(165deg, {C_PAGE_TOP} 0%, {C_PAGE_MID} 22%, {C_PAGE_BOTTOM} 55%, #f1f5f9 100%) !important;
        }}
        .main .block-container {{
            padding-top: 1.5rem !important;
            padding-bottom: 3rem !important;
            max-width: 1200px;
        }}
        /* Headings */
        .main h1, .main h2, .main h3 {{
            color: {C_DEEP} !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em !important;
        }}
        .main [data-testid="stHeader"] {{
            background: rgba(255,255,255,0.85);
            backdrop-filter: blur(8px);
            border-bottom: 1px solid {C_BORDER};
        }}
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background: linear-gradient(175deg, #0f172a 0%, #1e3a3a 48%, #134e4a 100%) !important;
            border-right: 1px solid rgba(255,255,255,0.08) !important;
        }}
        [data-testid="stSidebar"] .stMarkdown {{
            color: #e2e8f0 !important;
        }}
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {{
            color: #cbd5e1 !important;
        }}
        [data-testid="stSidebar"] h3 {{
            color: #f8fafc !important;
            font-size: 1.15rem !important;
            font-weight: 700 !important;
            margin-bottom: 0.25rem !important;
        }}
        [data-testid="stSidebar"] [data-baseweb="radio"] label {{
            color: #f1f5f9 !important;
        }}
        [data-testid="stSidebar"] .stRadio > label {{
            font-size: 0.72rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.12em !important;
            color: #94a3b8 !important;
            font-weight: 600 !important;
        }}
        /* Tabs — pill style */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 6px;
            background: rgba(255,255,255,0.65);
            padding: 6px;
            border-radius: 12px;
            border: 1px solid {C_BORDER};
            box-shadow: 0 1px 3px rgba(15,23,42,0.06);
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 10px;
            padding: 0.45rem 1.1rem;
            font-weight: 600;
            color: {C_BODY} !important;
            opacity: 1 !important;
        }}
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, {C_PRIMARY} 0%, {C_PRIMARY_LIGHT} 100%) !important;
            color: #ffffff !important;
        }}
        /* Dividers */
        hr, [data-testid="stHorizontalRule"] {{
            border: none;
            border-top: 1px solid {C_BORDER};
            margin: 1.25rem 0;
        }}
        /* Select & inputs */
        .stSelectbox label, .stSlider label, .stNumberInput label {{
            font-weight: 500 !important;
            color: {C_SLATE} !important;
            font-size: 0.88rem !important;
        }}
        div[data-baseweb="select"] > div {{
            border-radius: 10px !important;
            border-color: #cbd5e1 !important;
            background: {C_SURFACE} !important;
        }}
        .main div[data-baseweb="select"] span,
        .main div[data-baseweb="select"] div[role="button"] {{
            color: {C_BODY} !important;
        }}
        .main div[data-baseweb="input"] input {{
            color: {C_BODY} !important;
            -webkit-text-fill-color: {C_BODY} !important;
        }}
        .stSlider [data-baseweb="slider"] {{
            padding-top: 0.5rem;
        }}
        /* Alerts */
        div[data-baseweb="notification"], .stAlert {{
            border-radius: 10px !important;
            border-left-width: 4px !important;
        }}
        /* Hero */
        .hero-box {{
            background: linear-gradient(125deg, {C_DEEP} 0%, #1e3a5f 38%, #134e4a 72%, {C_PRIMARY} 110%);
            padding: 1.85rem 2rem;
            border-radius: 16px;
            margin-bottom: 1.35rem;
            box-shadow: 0 16px 48px rgba(15, 23, 42, 0.22), 0 0 0 1px rgba(255,255,255,0.06) inset;
        }}
        .hero-box h1 {{
            color: #f8fafc !important;
            font-size: 1.85rem !important;
            font-weight: 700 !important;
            margin: 0 !important;
            letter-spacing: -0.03em;
            line-height: 1.2 !important;
        }}
        .hero-box p {{
            color: rgba(248, 250, 252, 0.9) !important;
            margin: 0.65rem 0 0 0 !important;
            font-size: 1.05rem;
            line-height: 1.5;
            font-weight: 400;
        }}
        .badge {{
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-top: 0.85rem;
        }}
        .badge-ok {{
            background: rgba(16, 185, 129, 0.22);
            color: #d1fae5 !important;
            border: 1px solid rgba(16,185,129,0.45);
        }}
        .badge-warn {{
            background: rgba(251, 191, 36, 0.12);
            color: #fef3c7 !important;
            border: 1px solid rgba(251,191,36,0.35);
        }}
        /* Metrics */
        div[data-testid="stMetric"] {{
            background: {C_SURFACE};
            border: 1px solid {C_BORDER};
            border-radius: 12px;
            padding: 1rem 1.15rem;
            box-shadow: 0 2px 8px rgba(15,23,42,0.04);
        }}
        div[data-testid="stMetric"] label {{
            color: {C_MUTED} !important;
            font-size: 0.8rem !important;
            font-weight: 500 !important;
        }}
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: {C_DEEP} !important;
            font-weight: 700 !important;
        }}
        /* Dataframe */
        [data-testid="stDataFrame"] {{
            border: 1px solid {C_BORDER};
            border-radius: 10px;
            overflow: hidden;
        }}
        /* Body copy on light background — readable contrast */
        .main [data-testid="stMarkdownContainer"] {{
            color: {C_BODY} !important;
        }}
        .main [data-testid="stMarkdownContainer"] p,
        .main [data-testid="stMarkdownContainer"] li,
        .main [data-testid="stMarkdownContainer"] td {{
            color: {C_BODY} !important;
        }}
        .main [data-testid="stMarkdownContainer"] strong {{
            color: #0f172a !important;
        }}
        /* Captions */
        .main .stCaption, [data-testid="stCaption"] {{
            color: {C_MUTED} !important;
            opacity: 1 !important;
        }}
        /* Inline alerts / notifications on main */
        .main [data-baseweb="notification"] {{
            color: {C_BODY} !important;
        }}
        .main [data-baseweb="notification"] p,
        .main [data-baseweb="notification"] span,
        .main div[data-testid="stAlert"] p,
        .main div[data-testid="stAlert"] div[data-testid="stMarkdownContainer"] p {{
            color: {C_BODY} !important;
        }}
        /* Plot containers */
        [data-testid="stPlotlyChart"] {{
            border: 1px solid {C_BORDER};
            border-radius: 14px;
            background: {C_SURFACE};
            padding: 10px 8px 8px;
            box-shadow: 0 4px 20px rgba(15, 23, 42, 0.06);
        }}
        /* Expander */
        .streamlit-expanderHeader {{
            font-weight: 600 !important;
            color: {C_BODY} !important;
            background: rgba(255,255,255,0.6);
            border-radius: 10px;
        }}
        .streamlit-expanderContent {{
            color: {C_BODY} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_plotly(fig, title: str | None = None) -> None:
    layout = dict(
        template="plotly_white",
        font=dict(family="'DM Sans', sans-serif", size=12, color=C_SLATE),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.92)",
        margin=dict(l=52, r=28, t=62, b=52),
        hovermode="closest",
        hoverlabel=dict(bgcolor=C_DEEP, font_size=12, font_family="'DM Sans', sans-serif"),
    )
    if title:
        layout["title"] = dict(
            text=title,
            font=dict(size=17, color=C_DEEP, family="'DM Sans', sans-serif"),
            x=0.02,
            xanchor="left",
        )
    fig.update_layout(**layout)
    fig.update_xaxes(
        gridcolor="rgba(148, 163, 184, 0.28)",
        zeroline=False,
        linecolor=C_BORDER,
        tickfont=dict(color=C_MUTED, size=11),
    )
    fig.update_yaxes(
        gridcolor="rgba(148, 163, 184, 0.28)",
        zeroline=False,
        linecolor=C_BORDER,
        tickfont=dict(color=C_MUTED, size=11),
    )


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
            color_continuous_scale=CSCALE,
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
            color_continuous_scale=CSCALE,
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
            color_continuous_scale=CSCALE,
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
                    fillcolor="rgba(234, 88, 12, 0.14)",
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
    st.sidebar.markdown(
        """
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            <span style="width:10px;height:10px;border-radius:50%;background:linear-gradient(135deg,#5eead4,#0d9488);box-shadow:0 0 12px rgba(45,212,191,0.5);"></span>
            <span style="font-size:1.05rem;font-weight:700;color:#f8fafc;letter-spacing:-0.02em;">Epidemic IQ</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "<span style='opacity:0.88;font-size:0.875rem;line-height:1.4;display:block;'>Geo risk · ML forecasts · scenario lab</span>",
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
    st.sidebar.markdown(
        "<div style='margin-top:1.5rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,0.12);font-size:0.75rem;opacity:0.65;color:#94a3b8;'>Epidemic Intelligence</div>",
        unsafe_allow_html=True,
    )

    if page == "Intelligence (ML)":
        render_ml_dashboard()
    else:
        render_simple_timeseries_dashboard()


main()
