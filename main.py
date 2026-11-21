"""
EU ETS Speculative Trading Anomaly Monitor
===========================================
Streamlit dashboard that loads pre-trained Isolation Forest + Autoencoder
models to score weekly EU ETS (carbon market) data for speculative /
anomalous trading behaviour, validates against three known episodes
(2006 crash, 2010-11 VAT-fraud, 2018 reform bubble), and links anomaly
scores to market volatility.

Expected model files (place in the same folder as this script, or point
to them in the sidebar):
    isolation_forest.joblib   -> sklearn.ensemble.IsolationForest
    autoencoder_mlp.joblib    -> sklearn.neural_network.MLPRegressor
                                 trained as an autoencoder (X -> X)
    scaler.joblib             -> a fitted sklearn scaler (StandardScaler /
                                 MinMaxScaler) applied to the 6 features
                                 below before they reach the models

Expected input CSV columns (weekly frequency, 2008-present):
    date                              (YYYY-MM-DD)
    eua_price                         (optional, for context plots)
    eua_volume                        (optional, for context plots)
    non_compliance_volume_ratio
    price_fundamental_deviation
    volume_spike_ratio
    price_momentum
    off_deadline_trading_intensity
    options_futures_ratio

If your training used a different column order, re-map it in the sidebar
"Feature order" widget so it matches exactly what scaler.joblib expects.

Run with:
    streamlit run app.py
"""

import io
import warnings
from datetime import date

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Page config & constants
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="EU ETS Speculative Trading Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Theme — warm terracotta / cream / deep green palette
# --------------------------------------------------------------------------
THEME = {
    "cream": "#FBF7EF",
    "cream_dark": "#EDE7DC",
    "terracotta": "#C67D4C",
    "terracotta_light": "#E7B98C",
    "dark_green": "#2F4A3C",
    "sage_green": "#4F6F52",
    "charcoal": "#2B2620",
    "muted": "#9A9184",
    "gold": "#D99A3F",
    "input_bg": "#F1EAE0",
    "alert_red": "#B5573C",
}

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    h1, h2, h3, .stTabs [data-baseweb="tab"] p {{
        font-family: 'Poppins', sans-serif !important;
        color: {THEME['charcoal']};
    }}

    .stApp {{
        background: {THEME['cream']};
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {THEME['dark_green']} 0%, #24382E 100%);
    }}
    section[data-testid="stSidebar"] * {{
        color: {THEME['cream']} !important;
    }}
    section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {{
        color: {THEME['terracotta_light']};
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.15);
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: {THEME['input_bg']};
        padding: 6px;
        border-radius: 16px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 12px;
        padding: 8px 16px;
    }}
    .stTabs [aria-selected="true"] {{
        background: {THEME['dark_green']} !important;
    }}
    .stTabs [aria-selected="true"] p {{
        color: {THEME['cream']} !important;
    }}

    div[data-testid="stMetric"] {{
        background: {THEME['input_bg']};
        border-radius: 18px;
        padding: 14px 16px;
        border: 1px solid rgba(43, 38, 32, 0.06);
    }}
    div[data-testid="stMetricLabel"] {{
        color: {THEME['muted']} !important;
    }}
    div[data-testid="stMetricValue"] {{
        color: {THEME['charcoal']} !important;
    }}

    .stButton > button, .stDownloadButton > button {{
        background: {THEME['sage_green']};
        color: {THEME['cream']};
        border: none;
        border-radius: 14px;
        font-weight: 600;
        letter-spacing: 0.04em;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background: {THEME['dark_green']};
        color: {THEME['cream']};
    }}

    div[data-testid="stDataFrame"] {{
        border-radius: 16px;
        overflow: hidden;
    }}

    .hero-banner {{
        display: flex;
        align-items: center;
        gap: 16px;
        background: linear-gradient(120deg, {THEME['terracotta_light']} 0%, {THEME['terracotta']} 100%);
        border-radius: 24px;
        padding: 22px 26px;
        margin-bottom: 18px;
    }}
    .hero-icon {{
        width: 46px; height: 46px; border-radius: 14px;
        background: {THEME['dark_green']};
        display: flex; align-items: center; justify-content: center;
        font-family: 'Poppins', sans-serif; font-weight: 700;
        font-size: 12px; letter-spacing: 0.03em; color: {THEME['cream']};
        flex-shrink: 0;
    }}
    .hero-title {{
        color: white; font-family: 'Poppins', sans-serif;
        font-size: 22px; font-weight: 600; margin: 0;
    }}
    .hero-subtitle {{
        color: rgba(255,255,255,0.9); font-size: 13px; margin: 2px 0 0 0;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_FEATURES = [
    "non_compliance_volume_ratio",
    "price_fundamental_deviation",
    "volume_spike_ratio",
    "price_momentum",
    "off_deadline_trading_intensity",
    "options_futures_ratio",
]

FEATURE_LABELS = {
    "non_compliance_volume_ratio": "Non-compliance Volume Ratio",
    "price_fundamental_deviation": "Price–Fundamental Deviation",
    "volume_spike_ratio": "Volume Spike Ratio",
    "price_momentum": "Price Momentum",
    "off_deadline_trading_intensity": "Off-deadline Trading Intensity",
    "options_futures_ratio": "Options-to-Futures Ratio",
}

KNOWN_EPISODES = [
    {"name": "2006 Crash", "start": "2006-04-01", "end": "2006-07-31",
     "color": "rgba(239, 85, 59, 0.18)"},
    {"name": "2010–11 VAT Fraud", "start": "2010-06-01", "end": "2011-06-30",
     "color": "rgba(99, 110, 250, 0.18)"},
    {"name": "2018 Reform Bubble", "start": "2018-01-01", "end": "2018-12-31",
     "color": "rgba(0, 204, 150, 0.18)"},
]


# --------------------------------------------------------------------------
# Caching helpers
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(file_or_path):
    """Load a joblib model from an uploaded file-like object or a path."""
    return joblib.load(file_or_path)


@st.cache_data(show_spinner=False)
def load_csv(file_or_path):
    df = pd.read_csv(file_or_path)
    date_col = None
    for c in df.columns:
        if c.lower() in ("date", "week", "week_start", "timestamp"):
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date"}).sort_values("date").reset_index(drop=True)
    return df


def generate_demo_data():
    """Synthetic weekly demo series (2008 to today) so the dashboard is
    explorable even before real data / models are supplied."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2008-01-07", pd.Timestamp.today().normalize(), freq="W-MON")
    n = len(dates)

    price = 20 + np.cumsum(rng.normal(0, 0.6, n))
    price = np.clip(price, 3, 100)
    volume = np.abs(rng.normal(5_000_000, 800_000, n))

    df = pd.DataFrame({"date": dates, "eua_price": price, "eua_volume": volume})

    for col in DEFAULT_FEATURES:
        df[col] = rng.normal(0, 1, n)

    def boost(mask, cols, scale=3.0):
        for c in cols:
            df.loc[mask, c] += rng.normal(scale, 1.0, mask.sum())

    m1 = (df["date"] >= "2006-04-01") & (df["date"] <= "2006-07-31")
    m2 = (df["date"] >= "2010-06-01") & (df["date"] <= "2011-06-30")
    m3 = (df["date"] >= "2018-01-01") & (df["date"] <= "2018-12-31")
    boost(m2, ["non_compliance_volume_ratio", "off_deadline_trading_intensity"], 3.5)
    boost(m3, ["price_momentum", "volume_spike_ratio", "options_futures_ratio"], 3.0)
    boost(m1, ["price_fundamental_deviation", "volume_spike_ratio"], 3.0)

    df["realized_volatility"] = (
        pd.Series(price).pct_change().rolling(4).std().fillna(0).values * 100
    )
    return df


def minmax(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def score_isolation_forest(model, X):
    """Return anomaly scores in [0, 1], higher = more anomalous."""
    raw = -model.decision_function(X)  # sklearn: lower decision_function = more abnormal
    return minmax(raw), raw


def score_autoencoder(model, X):
    """Reconstruction-error based anomaly score, higher = more anomalous."""
    recon = model.predict(X)
    per_feature_sq_err = (X - recon) ** 2
    mse = per_feature_sq_err.mean(axis=1)
    return minmax(mse), mse, per_feature_sq_err


@st.cache_data(show_spinner=False)
def compute_shap_values(_model, X_sample, background_size=100):
    """Best-effort SHAP explanation for the Isolation Forest. Falls back
    to a permutation-based approximation if TreeExplainer isn't
    compatible with the installed shap/sklearn versions."""
    import shap

    bg = X_sample[: min(background_size, len(X_sample))]
    try:
        explainer = shap.TreeExplainer(_model)
        values = explainer.shap_values(X_sample)
    except Exception:
        explainer = shap.Explainer(_model.decision_function, bg)
        values = explainer(X_sample).values
    return np.asarray(values)


def forecast_series(dates, values, horizon):
    """Project a weekly series `horizon` steps into the future.

    Tries Holt's damped-trend exponential smoothing first; falls back to
    a plain linear-trend extrapolation if that fails to fit (e.g. too
    few points, or a degenerate/flat series). Returns future dates, the
    point forecast, and a simple growing uncertainty band derived from
    in-sample residuals -- this is a transparent statistical projection
    of the historical trend, not a validated predictive model.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    step = dates.iloc[-1] - dates.iloc[-2] if n >= 2 else pd.Timedelta(weeks=1)
    future_dates = pd.date_range(dates.iloc[-1] + step, periods=horizon, freq=step)

    method = "Holt's damped-trend exponential smoothing"
    try:
        if n < 8:
            raise ValueError("series too short for exponential smoothing")
        model = ExponentialSmoothing(
            values, trend="add", damped_trend=True, initialization_method="estimated"
        )
        fit = model.fit(optimized=True)
        fitted = np.asarray(fit.fittedvalues)
        point = np.asarray(fit.forecast(horizon))
    except Exception:
        x = np.arange(n)
        coef = np.polyfit(x, values, 1)
        fitted = np.polyval(coef, x)
        point = np.polyval(coef, np.arange(n, n + horizon))
        method = "linear trend extrapolation (fallback)"

    resid_std = float(np.nanstd(values - fitted)) if n > 1 else 0.0
    growth = 1.96 * resid_std * np.sqrt(np.arange(1, horizon + 1))
    lower, upper = point - growth, point + growth
    return future_dates, point, lower, upper, method


# --------------------------------------------------------------------------
# Sidebar — configuration only (models & data load silently in the background)
# --------------------------------------------------------------------------
st.sidebar.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
        <div style="width:34px;height:34px;border-radius:10px;background:{THEME['terracotta']};
                    display:flex;align-items:center;justify-content:center;font-size:13px;
                    font-family:'Poppins',sans-serif;font-weight:700;color:{THEME['cream']};">EM</div>
        <div style="font-family:'Poppins',sans-serif;font-weight:600;font-size:17px;">ETS Monitor</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.caption("Speculative-activity anomaly monitoring")
st.sidebar.markdown("---")

# Model files & dataset are loaded automatically from the working directory —
# no upload controls are shown in the UI. Place the following next to app.py:
#   isolation_forest.joblib, autoencoder_mlp.joblib, scaler.joblib
# and (optionally) a weekly CSV named one of: eu_ets_weekly.csv, data.csv,
# eu_ets_data.csv — otherwise synthetic demo data is used automatically.
LOCAL_DATA_CANDIDATES = ["eu_ets_weekly.csv", "eu_ets_data.csv", "data.csv", "weekly_data.csv"]


def _resolve_model(local_name):
    try:
        return load_model(local_name), f"local: {local_name}"
    except Exception:
        return None, None


iso_model, iso_src = _resolve_model("isolation_forest.joblib")
ae_model, ae_src = _resolve_model("autoencoder_mlp.joblib")
scaler, scaler_src = _resolve_model("scaler.joblib")

models_ready = all(m is not None for m in (iso_model, ae_model, scaler))

with st.sidebar.expander("Model status", expanded=not models_ready):
    st.write("Isolation Forest:", "Loaded" if iso_model is not None else "Missing")
    st.write("Autoencoder (MLP):", "Loaded" if ae_model is not None else "Missing")
    st.write("Scaler:", "Loaded" if scaler is not None else "Missing")
    if not models_ready:
        st.warning(
            "Place isolation_forest.joblib, autoencoder_mlp.joblib and "
            "scaler.joblib next to app.py to enable live scoring. Showing "
            "demo data without model scores until then."
        )

# --------------------------------------------------------------------------
# Load data — auto-detect a local CSV, otherwise fall back to demo data
# --------------------------------------------------------------------------
use_demo = True
df = None
for candidate in LOCAL_DATA_CANDIDATES:
    try:
        df = load_csv(candidate)
        use_demo = False
        st.sidebar.caption(f"Data source: {candidate}")
        break
    except Exception:
        continue
if df is None:
    df = generate_demo_data()
    st.sidebar.caption("Data source: synthetic demo data")

missing = [c for c in DEFAULT_FEATURES if c not in df.columns]
if missing and not use_demo:
    st.sidebar.error(f"Missing expected feature columns: {missing}")

feature_cols = [c for c in DEFAULT_FEATURES if c in df.columns]

st.sidebar.markdown("---")
st.sidebar.subheader("Composite score weighting")
w_iso = st.sidebar.slider("Isolation Forest weight", 0.0, 1.0, 0.5, 0.05)
w_ae = 1.0 - w_iso
st.sidebar.caption(f"Autoencoder weight: {w_ae:.2f}")
alert_threshold = st.sidebar.slider("Alert threshold (composite score)", 0.0, 1.0, 0.75, 0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Forecast")
forecast_horizon = st.sidebar.slider("Forecast horizon (weeks ahead)", 4, 52, 12, 1)

with st.sidebar.expander("Advanced: feature order"):
    st.caption("Must match the column order used when scaler.joblib was fit.")
    feature_cols = st.multiselect(
        "Feature columns (in order)",
        options=[c for c in df.columns if c != "date"],
        default=feature_cols,
    )

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-icon">ETS</div>
        <div>
            <p class="hero-title">EU ETS Speculative Trading Monitor</p>
            <p class="hero-subtitle">Isolation Forest + Autoencoder ensemble · validated against 2006, 2010–11 and 2018 episodes</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

min_d, max_d = df["date"].min().date(), df["date"].max().date()
c1, c2 = st.columns(2)
start_date = c1.date_input("Start date", value=min_d, min_value=min_d, max_value=max_d)
end_date = c2.date_input("End date", value=max_d, min_value=min_d, max_value=max_d)
mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
view = df.loc[mask].reset_index(drop=True)

# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
scored = view.copy()
shap_values = None

if models_ready and len(feature_cols) > 0 and not missing:
    X_raw = view[feature_cols].to_numpy(dtype=float)
    try:
        X_scaled = scaler.transform(X_raw)
    except Exception as e:
        st.error(
            f"Scaler could not transform the selected features "
            f"(shape mismatch?). Details: {e}"
        )
        X_scaled = None

    if X_scaled is not None:
        iso_score, iso_raw = score_isolation_forest(iso_model, X_scaled)
        ae_score, ae_mse, ae_feat_err = score_autoencoder(ae_model, X_scaled)
        composite = w_iso * iso_score + w_ae * ae_score

        scored["iso_forest_score"] = iso_score
        scored["autoencoder_score"] = ae_score
        scored["composite_score"] = composite
        scored["is_alert"] = composite >= alert_threshold

        for i, col in enumerate(feature_cols):
            scored[f"recon_err__{col}"] = ae_feat_err[:, i]
else:
    st.info(
        "Place the three model files next to app.py and a matching CSV "
        "in the working directory to compute live anomaly scores. Showing "
        "raw series only for now."
    )

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_overview, tab_episodes, tab_explain, tab_volatility, tab_forecast, tab_data = st.tabs(
    ["Overview", "Episode Validation", "Explainability (SHAP)", "Volatility Link", "Forecast", "Data"]
)

# ---- Overview -------------------------------------------------------------
with tab_overview:
    if "composite_score" in scored.columns:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Weeks analysed", f"{len(scored):,}")
        k2.metric("Alerts flagged", f"{int(scored['is_alert'].sum()):,}")
        k3.metric("Mean composite score", f"{scored['composite_score'].mean():.3f}")
        k4.metric(
            "Peak composite score",
            f"{scored['composite_score'].max():.3f}",
            help=f"on {scored.loc[scored['composite_score'].idxmax(), 'date'].date()}",
        )

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if "eua_price" in scored.columns:
            fig.add_trace(
                go.Scatter(x=scored["date"], y=scored["eua_price"], name="EUA price",
                           line=dict(color=THEME["dark_green"], width=1.5)),
                secondary_y=False,
            )
        fig.add_trace(
            go.Scatter(x=scored["date"], y=scored["composite_score"], name="Composite anomaly score",
                       line=dict(color=THEME["terracotta"], width=2)),
            secondary_y=True,
        )
        alerts = scored[scored["is_alert"]]
        fig.add_trace(
            go.Scatter(x=alerts["date"], y=alerts["composite_score"], mode="markers",
                       name="Alert", marker=dict(color=THEME["alert_red"], size=8, symbol="x")),
            secondary_y=True,
        )
        episode_fill_colors = [
            "rgba(198, 125, 76, 0.16)",   # terracotta tint
            "rgba(47, 74, 60, 0.14)",     # dark green tint
            "rgba(217, 154, 63, 0.18)",   # gold tint
        ]
        for ep, fill in zip(KNOWN_EPISODES, episode_fill_colors):
            fig.add_vrect(x0=ep["start"], x1=ep["end"], fillcolor=fill,
                          line_width=0, annotation_text=ep["name"], annotation_position="top left")
        fig.update_layout(
            height=480, hovermode="x unified",
            legend=dict(orientation="h", y=1.08),
            margin=dict(t=60, b=10),
            plot_bgcolor=THEME["cream"], paper_bgcolor=THEME["cream"],
            font=dict(color=THEME["charcoal"]),
        )
        fig.update_yaxes(title_text="EUA price (€)", secondary_y=False)
        fig.update_yaxes(title_text="Anomaly score", range=[0, 1], secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Model agreement")
        agree_fig = px.scatter(
            scored, x="iso_forest_score", y="autoencoder_score", color="composite_score",
            color_continuous_scale=[THEME["dark_green"], THEME["terracotta_light"], THEME["gold"]],
            hover_data=["date"],
            labels={"iso_forest_score": "Isolation Forest score",
                    "autoencoder_score": "Autoencoder score"},
        )
        agree_fig.update_layout(
            height=380, plot_bgcolor=THEME["cream"], paper_bgcolor=THEME["cream"],
            font=dict(color=THEME["charcoal"]),
        )
        st.plotly_chart(agree_fig, use_container_width=True)

        st.subheader("Top anomalous weeks")
        top_cols = ["date", "composite_score", "iso_forest_score", "autoencoder_score"]
        top_n = scored.sort_values("composite_score", ascending=False).head(10)[top_cols].copy()
        top_n["date"] = top_n["date"].dt.date
        st.dataframe(
            top_n.style.format({
                "composite_score": "{:.3f}",
                "iso_forest_score": "{:.3f}",
                "autoencoder_score": "{:.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        fig = px.line(view, x="date", y="eua_price" if "eua_price" in view.columns else feature_cols[0],
                      title="EUA price / selected series")
        st.plotly_chart(fig, use_container_width=True)

# ---- Episode validation -----------------------------------------------------
with tab_episodes:
    st.subheader("Validation against documented speculative episodes")
    if "composite_score" in scored.columns:
        rows = []
        for ep in KNOWN_EPISODES:
            m = (scored["date"] >= ep["start"]) & (scored["date"] <= ep["end"])
            other = ~m
            if m.sum() == 0:
                continue
            rows.append({
                "Episode": ep["name"],
                "Weeks in window": int(m.sum()),
                "Mean score (in-window)": scored.loc[m, "composite_score"].mean(),
                "Mean score (baseline)": scored.loc[other, "composite_score"].mean(),
                "Peak score": scored.loc[m, "composite_score"].max(),
                "Alerts triggered": int(scored.loc[m, "is_alert"].sum()),
            })
        ep_df = pd.DataFrame(rows)
        if not ep_df.empty:
            ep_df["Score lift vs baseline"] = (
                ep_df["Mean score (in-window)"] - ep_df["Mean score (baseline)"]
            )
            st.dataframe(
                ep_df.style.format({
                    "Mean score (in-window)": "{:.3f}",
                    "Mean score (baseline)": "{:.3f}",
                    "Peak score": "{:.3f}",
                    "Score lift vs baseline": "{:+.3f}",
                }),
                use_container_width=True,
            )
            bar = px.bar(
                ep_df, x="Episode", y=["Mean score (in-window)", "Mean score (baseline)"],
                barmode="group", title="In-window vs. baseline anomaly score",
                color_discrete_sequence=[THEME["terracotta"], THEME["muted"]],
            )
            bar.update_layout(plot_bgcolor=THEME["cream"], paper_bgcolor=THEME["cream"],
                              font=dict(color=THEME["charcoal"]))
            st.plotly_chart(bar, use_container_width=True)
        else:
            st.info("None of the three known episodes fall within the selected date range.")
    else:
        st.info("Run scoring (load models + matching data) to see episode validation.")

# ---- Explainability ---------------------------------------------------------
with tab_explain:
    st.subheader("Feature-level explainability")
    if "composite_score" in scored.columns:
        st.markdown("**Global feature importance — average |reconstruction error| (Autoencoder)**")
        err_cols = [f"recon_err__{c}" for c in feature_cols]
        avg_err = scored[err_cols].mean().rename(lambda x: x.replace("recon_err__", ""))
        avg_err.index = [FEATURE_LABELS.get(i, i) for i in avg_err.index]
        fig_err = px.bar(avg_err.sort_values(ascending=True), orientation="h",
                         labels={"value": "Mean squared reconstruction error", "index": "Feature"},
                         color_discrete_sequence=[THEME["dark_green"]])
        fig_err.update_layout(plot_bgcolor=THEME["cream"], paper_bgcolor=THEME["cream"],
                              font=dict(color=THEME["charcoal"]), showlegend=False)
        st.plotly_chart(fig_err, use_container_width=True)

        st.markdown("**SHAP feature attribution — Isolation Forest**")
        try:
            X_scaled_full = scaler.transform(view[feature_cols].to_numpy(dtype=float))
            sample_n = min(300, len(X_scaled_full))
            idx = np.linspace(0, len(X_scaled_full) - 1, sample_n).astype(int)
            with st.spinner("Computing SHAP values (sampled for speed)..."):
                sv = compute_shap_values(iso_model, X_scaled_full[idx])
            sv_mean_abs = np.abs(sv).mean(axis=0)
            shap_df = pd.DataFrame({
                "feature": [FEATURE_LABELS.get(c, c) for c in feature_cols],
                "mean_abs_shap": sv_mean_abs,
            }).sort_values("mean_abs_shap")
            fig_shap = px.bar(shap_df, x="mean_abs_shap", y="feature", orientation="h",
                              labels={"mean_abs_shap": "Mean |SHAP value|", "feature": ""},
                              color_discrete_sequence=[THEME["terracotta"]])
            fig_shap.update_layout(plot_bgcolor=THEME["cream"], paper_bgcolor=THEME["cream"],
                                   font=dict(color=THEME["charcoal"]), showlegend=False)
            st.plotly_chart(fig_shap, use_container_width=True)
        except Exception as e:
            st.warning(f"SHAP computation unavailable in this environment: {e}")

        st.markdown("**Inspect a single week**")
        sel_date = st.select_slider("Select week", options=list(scored["date"]),
                                    value=scored.loc[scored["composite_score"].idxmax(), "date"])
        row = scored.loc[scored["date"] == sel_date].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Composite score", f"{row['composite_score']:.3f}")
        c2.metric("Isolation Forest", f"{row['iso_forest_score']:.3f}")
        c3.metric("Autoencoder", f"{row['autoencoder_score']:.3f}")
        row_err = pd.Series(
            {FEATURE_LABELS.get(c, c): row[f"recon_err__{c}"] for c in feature_cols}
        ).sort_values(ascending=True)
        fig_row = px.bar(row_err, orientation="h",
                         labels={"value": "Squared reconstruction error", "index": ""},
                         title=f"Feature-level reconstruction error — week of {pd.Timestamp(sel_date).date()}",
                         color_discrete_sequence=[THEME["gold"]])
        fig_row.update_layout(plot_bgcolor=THEME["cream"], paper_bgcolor=THEME["cream"],
                              font=dict(color=THEME["charcoal"]), showlegend=False)
        st.plotly_chart(fig_row, use_container_width=True)
    else:
        st.info("Run scoring (load models + matching data) to see explainability charts.")

# ---- Volatility link ---------------------------------------------------------
with tab_volatility:
    st.subheader("Anomaly score vs market volatility")
    vol_col = None
    for cand in ["realized_volatility", "volatility", "rv"]:
        if cand in scored.columns:
            vol_col = cand
            break
    if vol_col is None and "eua_price" in scored.columns:
        scored["realized_volatility"] = (
            scored["eua_price"].pct_change().rolling(4).std().fillna(0) * 100
        )
        vol_col = "realized_volatility"

    if "composite_score" in scored.columns and vol_col is not None:
        corr = scored[["composite_score", vol_col]].corr().iloc[0, 1]
        st.metric("Correlation (composite score vs. volatility)", f"{corr:.3f}")

        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(x=scored["date"], y=scored[vol_col], name="Realized volatility",
                                  line=dict(color=THEME["dark_green"])), secondary_y=False)
        fig2.add_trace(go.Scatter(x=scored["date"], y=scored["composite_score"], name="Composite score",
                                  line=dict(color=THEME["terracotta"])), secondary_y=True)
        fig2.update_layout(
            height=420, hovermode="x unified",
            plot_bgcolor=THEME["cream"], paper_bgcolor=THEME["cream"],
            font=dict(color=THEME["charcoal"]),
        )
        fig2.update_yaxes(title_text="Realized volatility (%)", secondary_y=False)
        fig2.update_yaxes(title_text="Anomaly score", range=[0, 1], secondary_y=True)
        st.plotly_chart(fig2, use_container_width=True)

        scatter = px.scatter(
            scored, x="composite_score", y=vol_col, trendline="ols",
            labels={"composite_score": "Composite anomaly score", vol_col: "Realized volatility (%)"},
            color_discrete_sequence=[THEME["terracotta"]],
        )
        scatter.update_layout(plot_bgcolor=THEME["cream"], paper_bgcolor=THEME["cream"],
                              font=dict(color=THEME["charcoal"]))
        scatter.update_traces(selector=dict(mode="lines"), line=dict(color=THEME["dark_green"]))
        st.plotly_chart(scatter, use_container_width=True)
    else:
        st.info(
            "Provide an `eua_price` or `realized_volatility` column, and run "
            "scoring, to see the volatility linkage."
        )

# ---- Forecast ---------------------------------------------------------------
with tab_forecast:
    st.subheader(f"{forecast_horizon}-week outlook")
    st.caption(
        "A simple statistical projection of the historical trend (Holt's "
        "damped-trend exponential smoothing, with a linear-trend fallback "
        "for short series). This extrapolates past behaviour forward -- it "
        "is not a validated predictive model and should not replace the "
        "retrospective anomaly detection above as an early-warning signal."
    )

    if len(scored) < 8:
        st.info("Select a wider date range (at least 8 weeks) to generate a forecast.")
    else:
        if "eua_price" in scored.columns:
            f_dates, f_point, f_lo, f_hi, f_method = forecast_series(
                scored["date"], scored["eua_price"], forecast_horizon
            )
            st.markdown(f"**EUA price forecast** &nbsp;·&nbsp; method: {f_method}")
            price_fig = go.Figure()
            price_fig.add_trace(go.Scatter(
                x=scored["date"], y=scored["eua_price"], name="Historical price",
                line=dict(color=THEME["dark_green"], width=1.5),
            ))
            price_fig.add_trace(go.Scatter(
                x=f_dates, y=f_point, name="Forecast", mode="lines",
                line=dict(color=THEME["terracotta"], width=2, dash="dash"),
            ))
            price_fig.add_trace(go.Scatter(
                x=list(f_dates) + list(f_dates[::-1]), y=list(f_hi) + list(f_lo[::-1]),
                fill="toself", fillcolor="rgba(198, 125, 76, 0.15)",
                line=dict(width=0), name="95% band", hoverinfo="skip",
            ))
            price_fig.update_layout(
                height=380, hovermode="x unified",
                plot_bgcolor=THEME["cream"], paper_bgcolor=THEME["cream"],
                font=dict(color=THEME["charcoal"]),
                yaxis_title="EUA price (EUR)", margin=dict(t=20),
            )
            st.plotly_chart(price_fig, use_container_width=True)

        if "composite_score" in scored.columns:
            s_dates, s_point, s_lo, s_hi, s_method = forecast_series(
                scored["date"], scored["composite_score"], forecast_horizon
            )
            s_point_c = np.clip(s_point, 0, 1)
            s_lo_c = np.clip(s_lo, 0, 1)
            s_hi_c = np.clip(s_hi, 0, 1)

            st.markdown(f"**Composite anomaly score outlook** &nbsp;·&nbsp; method: {s_method}")
            score_fig = go.Figure()
            score_fig.add_trace(go.Scatter(
                x=scored["date"], y=scored["composite_score"], name="Historical score",
                line=dict(color=THEME["terracotta"], width=1.5),
            ))
            score_fig.add_trace(go.Scatter(
                x=s_dates, y=s_point_c, name="Projected score", mode="lines",
                line=dict(color=THEME["dark_green"], width=2, dash="dash"),
            ))
            score_fig.add_trace(go.Scatter(
                x=list(s_dates) + list(s_dates[::-1]), y=list(s_hi_c) + list(s_lo_c[::-1]),
                fill="toself", fillcolor="rgba(47, 74, 60, 0.15)",
                line=dict(width=0), name="95% band", hoverinfo="skip",
            ))
            score_fig.add_hline(y=alert_threshold, line_dash="dot", line_color=THEME["alert_red"],
                                annotation_text="Alert threshold")
            score_fig.update_layout(
                height=380, hovermode="x unified",
                plot_bgcolor=THEME["cream"], paper_bgcolor=THEME["cream"],
                font=dict(color=THEME["charcoal"]),
                yaxis_title="Composite anomaly score", yaxis_range=[0, 1], margin=dict(t=20),
            )
            st.plotly_chart(score_fig, use_container_width=True)

            at_risk = pd.DataFrame({"date": s_dates, "projected_score": s_point_c})
            at_risk = at_risk[at_risk["projected_score"] >= alert_threshold]
            if len(at_risk) > 0:
                st.warning(
                    f"{len(at_risk)} of the next {forecast_horizon} weeks are projected to "
                    f"cross the alert threshold ({alert_threshold:.2f}), starting "
                    f"{at_risk.iloc[0]['date'].date()}. Treat this as a trend signal to "
                    "watch, not a confirmed forecast."
                )
            else:
                st.success(
                    f"No week in the next {forecast_horizon} weeks is projected to cross "
                    f"the alert threshold ({alert_threshold:.2f}) based on the current trend."
                )
        else:
            st.info("Run scoring (load models + matching data) to forecast the anomaly score.")

# ---- Data tab ---------------------------------------------------------------
with tab_data:
    st.subheader("Underlying data")
    st.dataframe(scored, use_container_width=True, height=420)
    csv_buf = io.StringIO()
    scored.to_csv(csv_buf, index=False)
    st.download_button(
        "Download scored data as CSV",
        data=csv_buf.getvalue(),
        file_name=f"eu_ets_anomaly_scores_{date.today().isoformat()}.csv",
        mime="text/csv",
    )

st.sidebar.markdown("---")
st.sidebar.caption(
    "Framework designed for extension to emerging carbon markets "
    "(UK ETS, India CCTS, China ETS) once equivalent feature data is available."
)