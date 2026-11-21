import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import joblib
import io

from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import shap

st.set_page_config(
    page_title="CarbonWatch | Speculative Trading Detector",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CREAM = "#F6F1E6"
CARD = "#FFFFFF"
SAGE = "#6D9773"
SAGE_DARK = "#4F7A58"
DARKGREEN = "#0C3B2E"
TAN = "#B88A52"
GOLD = "#E3A008"
MUTED = "#8C8878"
BORDER = "#EAE3D3"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Poppins', sans-serif;
}}

.stApp {{
    background: {CREAM};
    color: {DARKGREEN};
}}

h1, h2, h3, h4 {{
    color: {DARKGREEN} !important;
    font-weight: 700 !important;
}}

p, span, label, div {{
    color: {DARKGREEN};
}}

.topbar {{
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 6px 4px 22px 4px;
}}

.logo-circle {{
    width: 46px;
    height: 46px;
    border-radius: 16px;
    background: {SAGE};
    color: {CREAM};
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 16px;
    box-shadow: 0 6px 14px rgba(109,151,115,0.35);
}}

.app-title {{
    font-size: 22px;
    font-weight: 800;
    color: {DARKGREEN};
    line-height: 1.1;
}}

.app-sub {{
    font-size: 13px;
    color: {MUTED};
    margin-top: 2px;
}}

.card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 24px;
    padding: 26px 28px;
    margin-bottom: 22px;
    box-shadow: 0 10px 26px rgba(12,59,46,0.06);
}}

.card-title {{
    font-size: 18px;
    font-weight: 700;
    color: {DARKGREEN};
    margin-bottom: 2px;
}}

.card-subtitle {{
    font-size: 13px;
    color: {MUTED};
    margin-bottom: 18px;
}}

.field-label {{
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {MUTED};
    font-weight: 700;
    margin-bottom: 4px;
}}

.metric-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 20px 22px;
    box-shadow: 0 8px 20px rgba(12,59,46,0.05);
    height: 100%;
}}

.metric-label {{
    font-size: 11px;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: {MUTED};
    font-weight: 700;
}}

.metric-value {{
    font-size: 28px;
    font-weight: 800;
    color: {SAGE_DARK};
    margin-top: 6px;
}}

.metric-delta {{
    font-size: 12px;
    color: {TAN};
    margin-top: 4px;
}}

.status-pill {{
    display: inline-block;
    padding: 7px 16px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 12px;
    margin-top: 4px;
}}

.status-loaded {{
    background: rgba(109,151,115,0.14);
    color: {SAGE_DARK};
}}

.status-trained {{
    background: rgba(227,160,8,0.14);
    color: {GOLD};
}}

.episode-card {{
    border-radius: 20px;
    padding: 20px;
    background: {CARD};
    border: 1px solid {BORDER};
    border-left: 4px solid {SAGE};
    margin-bottom: 14px;
    box-shadow: 0 8px 20px rgba(12,59,46,0.05);
}}

.episode-title {{
    font-weight: 700;
    font-size: 16px;
    color: {DARKGREEN};
}}

.episode-sub {{
    font-size: 12px;
    color: {MUTED};
}}

div.stButton > button {{
    background: {SAGE};
    color: {CREAM};
    border: none;
    border-radius: 999px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-size: 13px;
    padding: 12px 30px;
    transition: 0.2s;
    box-shadow: 0 8px 18px rgba(109,151,115,0.35);
}}

div.stButton > button:hover {{
    background: {SAGE_DARK};
    transform: translateY(-1px);
}}

div[data-testid="stRadio"] > div {{
    gap: 8px;
}}

div[data-testid="stRadio"] label {{
    background: {CREAM};
    border: 1px solid {BORDER};
    border-radius: 999px;
    padding: 8px 18px;
    margin-right: 6px;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    border-bottom: 1px solid {BORDER};
}}

.stTabs [data-baseweb="tab"] {{
    background: transparent;
    border-radius: 0;
    padding: 10px 16px;
    color: {MUTED};
    font-weight: 600;
}}

.stTabs [aria-selected="true"] {{
    color: {SAGE_DARK} !important;
    border-bottom: 2px solid {SAGE};
}}

thead tr th {{
    background-color: {CREAM} !important;
    color: {DARKGREEN} !important;
}}

tbody tr td {{
    color: {DARKGREEN} !important;
}}

section[data-testid="stFileUploaderDropzone"] {{
    background: {CREAM};
    border: 1px dashed {BORDER};
    border-radius: 16px;
}}

.footer-tag {{
    text-align: center;
    color: {MUTED};
    font-size: 12px;
    margin-top: 30px;
    opacity: 0.8;
}}
</style>
""", unsafe_allow_html=True)

RANDOM_STATE = 42

# Features that only need price/volume data.
PRICE_ONLY_FEATURES = [
    "volume_spike_ratio",
    "price_momentum_4w",
    "off_deadline_trading_intensity",
    "realized_vol_4w",
]

# Features that additionally require compliance data (annual CSV or weekly manual entry).
COMPLIANCE_FEATURES = [
    "non_compliance_ratio",
    "price_fundamental_deviation",
]

# Features unlocked by the extra fundamental / derivatives columns in Manual Input mode.
FUNDAMENTAL_FEATURES = [
    "gas_price_deviation",
    "coal_price_deviation",
    "power_price_deviation",
    "weather_deviation",
    "options_futures_ratio",
]

FEATURE_LABELS = {
    "non_compliance_ratio": "Non-Compliance Volume Ratio",
    "price_fundamental_deviation": "Price-Fundamental Deviation",
    "volume_spike_ratio": "Volume Spike Ratio",
    "price_momentum_4w": "Price Momentum (4w)",
    "off_deadline_trading_intensity": "Off-Deadline Trading Intensity",
    "realized_vol_4w": "Realized Volatility (4w)",
    "gas_price_deviation": "Natural Gas Price Deviation",
    "coal_price_deviation": "Coal Price Deviation",
    "power_price_deviation": "Electricity Price Deviation",
    "weather_deviation": "Weather Deviation (z-score)",
    "options_futures_ratio": "Options-to-Futures Ratio",
}

DOCUMENTED_EPISODES = {
    "2006 Crash": ("2006-01-01", "2006-12-31", "Over-allocation revealed, EUA price collapsed ~60%"),
    "2010-11 VAT Fraud": ("2010-01-01", "2011-12-31", "Carousel VAT fraud drove abnormal volume spikes"),
    "2018 Reform Bubble": ("2018-01-01", "2018-12-31", "Market Stability Reserve announcement triggered rally"),
}

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
IF_PATH = MODEL_DIR / "isolation_forest.joblib"
AE_PATH = MODEL_DIR / "autoencoder_mlp.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
FEATURES_PATH = MODEL_DIR / "feature_cols.joblib"

PLOT_FONT = dict(family="Poppins, sans-serif", color=DARKGREEN)
GRID_COLOR = "rgba(12,59,46,0.08)"

# Reference EUR->INR rate. FX moves daily, so this is only a starting default —
# it's exposed as an editable input in the UI rather than hardcoded everywhere.
DEFAULT_EUR_INR_RATE = 110.0

# Manual Input table column names, kept as constants so column_config and the
# parsing code below can never drift out of sync with each other.
MCOL_DATE = "Date"
MCOL_PRICE = "EUA Price (INR/tonne)"
MCOL_VOLUME = "EUA Volume"
MCOL_NONCOMPLIANCE = "EUTL Non-Compliance (%)"
MCOL_GAS = "Natural Gas Price (INR/MWh)"
MCOL_COAL = "Coal Price (INR/tonne)"
MCOL_POWER = "Electricity Price (INR/MWh)"
MCOL_WEATHER = "Weather Index"
MCOL_OPTIONS_OI = "Options Open Interest"
MCOL_FUTURES_OI = "Futures Open Interest/Volume"

MANUAL_COLUMNS = [
    MCOL_DATE, MCOL_PRICE, MCOL_VOLUME, MCOL_NONCOMPLIANCE, MCOL_GAS,
    MCOL_COAL, MCOL_POWER, MCOL_WEATHER, MCOL_OPTIONS_OI, MCOL_FUTURES_OI,
]


def style_fig(fig, height=420, legend=True):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=PLOT_FONT,
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=legend,
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    return fig


def style_indicator_fig(fig, height=280):
    # Indicator ("gauge") figures have no cartesian axes, so they get a
    # lighter styling pass than style_fig.
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=PLOT_FONT,
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def format_inr(value, decimals=2):
    """Formats a number using the Indian digit-grouping convention with a
    rupee sign, e.g. 1234567.8 -> '₹12,34,567.80'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "₹0.00" if decimals else "₹0"
    neg = value < 0
    value = abs(float(value))
    s = f"{value:,.{decimals}f}" if decimals else f"{value:,.0f}"
    if "." in s:
        whole, dec = s.split(".")
    else:
        whole, dec = s, None
    whole = whole.replace(",", "")
    if len(whole) > 3:
        last3 = whole[-3:]
        rest = whole[:-3]
        groups = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        whole = ",".join(groups) + "," + last3
    formatted = f"₹{whole}" + (f".{dec}" if dec is not None else "")
    return f"-{formatted}" if neg else formatted


def saved_model_files_exist():
    return IF_PATH.exists() and AE_PATH.exists() and SCALER_PATH.exists() and FEATURES_PATH.exists()


def parse_vol(v):
    if pd.isna(v):
        return np.nan
    v = str(v).strip().upper()
    mult = 1.0
    if v.endswith("K"):
        mult, v = 1_000.0, v[:-1]
    elif v.endswith("M"):
        mult, v = 1_000_000.0, v[:-1]
    try:
        return float(v) * mult
    except ValueError:
        return np.nan


def load_eu_ets_compliance(file_obj):
    df = pd.read_csv(file_obj)
    required_cols = {"year", "main_activity_code", "citl_information", "value"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"EU-ETS compliance CSV is missing required column(s): {', '.join(sorted(missing))}"
        )
    years = [str(y) for y in range(2005, 2026)]
    d = df[df["year"].astype(str).isin(years)].copy()
    d["year"] = d["year"].astype(int)
    d = d[d["main_activity_code"] == "20-99"]
    metrics = {
        "total_allocated": "1. Total allocated allowances (EUA or EUAA)",
        "verified_emissions": "2.1 EU-ETS Verified Emission",
        "total_surrendered": "4. Total surrendered units",
    }
    panel = {}
    for name, label in metrics.items():
        panel[name] = d[d["citl_information"] == label].groupby("year")["value"].sum()
    panel = pd.DataFrame(panel).sort_index()
    panel["non_compliance_ratio"] = (
        (panel["verified_emissions"] - panel["total_surrendered"]) / panel["verified_emissions"]
    )
    panel["allocation_surplus_ratio"] = (
        (panel["total_allocated"] - panel["verified_emissions"]) / panel["verified_emissions"]
    )
    return panel


def load_price_series(daily_file=None, monthly_file=None, eur_inr_rate=1.0):
    """Builds a weekly price/volume panel from whichever price file(s) are
    supplied. At least one of daily_file / monthly_file must be provided.
    Source prices are EUR-denominated (Investing.com convention) and are
    converted to INR using eur_inr_rate."""
    frames = []
    for f in (daily_file, monthly_file):
        if f is None:
            continue
        d = pd.read_csv(f, encoding="utf-8-sig")
        required_cols = {"Date", "Price", "Vol."}
        missing = required_cols - set(d.columns)
        if missing:
            raise ValueError(
                f"Price CSV is missing required column(s): {', '.join(sorted(missing))}"
            )
        d["Date"] = pd.to_datetime(d["Date"], format="%m/%d/%Y", errors="coerce")
        d["Price"] = pd.to_numeric(d["Price"], errors="coerce") * eur_inr_rate
        d["Vol."] = d["Vol."].apply(parse_vol)
        frames.append(d[["Date", "Price", "Vol."]])

    if not frames:
        raise ValueError("At least one price file (Daily or Monthly EUA Futures CSV) is required.")

    combined = pd.concat(frames).drop_duplicates(subset="Date").sort_values("Date").set_index("Date")
    weekly = combined.resample("W-FRI").agg({"Price": "last", "Vol.": "sum"})
    weekly = weekly.rename(columns={"Price": "price", "Vol.": "volume"})
    weekly = weekly.dropna(subset=["price"])

    if len(weekly) == 0:
        raise ValueError(
            "No valid weekly price observations could be built from the supplied file(s). "
            "Check the Date/Price columns and date format (MM/DD/YYYY)."
        )
    return weekly


def engineer_features(weekly_price, compliance=None, weekly_non_compliance=None,
                       gas_price=None, coal_price=None, power_price=None,
                       weather=None, options_oi=None, futures_oi=None):
    """Builds the weekly feature panel. weekly_price is required; every other
    argument is optional. Only the features that can actually be computed from
    what's supplied are returned as (dataframe, feature_cols_used)."""
    feat = weekly_price.copy()
    feat["year"] = feat.index.year

    available = []

    # --- Price-only features (always computable) ---
    feat["price_momentum_4w"] = feat["price"].pct_change(4)
    feat["volume_spike_ratio"] = feat["volume"] / feat["volume"].rolling(8, min_periods=4).mean()
    feat["weekly_return"] = feat["price"].pct_change()
    feat["realized_vol_4w"] = feat["weekly_return"].rolling(4, min_periods=2).std()

    deadline_week = feat.index.to_series().apply(
        lambda d: abs((d - pd.Timestamp(year=d.year, month=4, day=30)).days) <= 28
    )
    feat["near_compliance_deadline"] = deadline_week.astype(int)
    yearly_avg_vol = feat.groupby("year")["volume"].transform("mean")
    feat["off_deadline_trading_intensity"] = np.where(
        feat["near_compliance_deadline"] == 0,
        feat["volume"] / yearly_avg_vol,
        np.nan,
    )
    feat["off_deadline_trading_intensity"] = feat["off_deadline_trading_intensity"].fillna(
        feat["off_deadline_trading_intensity"].rolling(8, min_periods=1).mean()
    )
    available += PRICE_ONLY_FEATURES

    # --- Compliance features: prefer an explicit weekly series (Manual Input),
    # otherwise fall back to the annual EU-ETS compliance CSV joined by year ---
    if weekly_non_compliance is not None and len(weekly_non_compliance) > 0:
        feat = feat.join(weekly_non_compliance.rename("non_compliance_ratio"), how="left")
        available.append("non_compliance_ratio")
    elif compliance is not None and len(compliance) > 0:
        feat = feat.join(compliance, on="year")
        feat["price_fundamental_deviation"] = (
            feat["weekly_return"] - feat["allocation_surplus_ratio"].diff().fillna(0)
        )
        available += COMPLIANCE_FEATURES

    # --- Optional fundamental-factor deviations (Manual Input only, for now) ---
    if gas_price is not None and len(gas_price) > 0:
        feat = feat.join(gas_price.rename("gas_price"), how="left")
        feat["gas_price_deviation"] = feat["weekly_return"] - feat["gas_price"].pct_change()
        available.append("gas_price_deviation")

    if coal_price is not None and len(coal_price) > 0:
        feat = feat.join(coal_price.rename("coal_price"), how="left")
        feat["coal_price_deviation"] = feat["weekly_return"] - feat["coal_price"].pct_change()
        available.append("coal_price_deviation")

    if power_price is not None and len(power_price) > 0:
        feat = feat.join(power_price.rename("power_price"), how="left")
        feat["power_price_deviation"] = feat["weekly_return"] - feat["power_price"].pct_change()
        available.append("power_price_deviation")

    if weather is not None and len(weather) > 0:
        feat = feat.join(weather.rename("weather_raw"), how="left")
        w_mean = feat["weather_raw"].rolling(8, min_periods=2).mean()
        w_std = feat["weather_raw"].rolling(8, min_periods=2).std()
        feat["weather_deviation"] = (feat["weather_raw"] - w_mean) / (w_std + 1e-9)
        available.append("weather_deviation")

    if options_oi is not None and futures_oi is not None and len(options_oi) > 0 and len(futures_oi) > 0:
        feat = feat.join(options_oi.rename("options_oi"), how="left")
        feat = feat.join(futures_oi.rename("futures_oi"), how="left")
        feat["options_futures_ratio"] = feat["options_oi"] / (feat["futures_oi"] + 1e-9)
        available.append("options_futures_ratio")

    out = feat[["price", "volume"] + available].dropna()
    return out, available


def generate_demo_data(eur_inr_rate=1.0):
    rng = np.random.default_rng(RANDOM_STATE)
    dates = pd.date_range("2008-01-04", "2023-12-29", freq="W-FRI")
    n = len(dates)
    trend = np.linspace(20, 85, n)
    noise = rng.normal(0, 3.5, n).cumsum() * 0.05
    seasonal = 4 * np.sin(np.linspace(0, 40, n))
    price = np.clip(trend + noise + seasonal, 3, None) * eur_inr_rate

    vat_mask = (dates >= "2010-06-01") & (dates <= "2011-12-31")
    reform_mask = (dates >= "2018-01-01") & (dates <= "2018-12-31")
    price[reform_mask] *= 1.35

    volume = rng.gamma(shape=5, scale=1_500_000, size=n)
    volume[vat_mask] *= rng.uniform(2.2, 3.5, vat_mask.sum())
    volume[reform_mask] *= rng.uniform(1.6, 2.4, reform_mask.sum())

    weekly_price = pd.DataFrame({"price": price, "volume": volume}, index=dates)

    years = np.arange(2005, 2026)
    verified = 2000 + rng.normal(0, 40, len(years)).cumsum() * 0.3 + 2050
    allocated = verified * rng.uniform(0.9, 1.15, len(years))
    surrendered = verified * rng.uniform(0.85, 1.0, len(years))
    compliance = pd.DataFrame({
        "total_allocated": allocated,
        "verified_emissions": verified,
        "total_surrendered": surrendered,
    }, index=years)
    compliance["non_compliance_ratio"] = (
        (compliance["verified_emissions"] - compliance["total_surrendered"]) / compliance["verified_emissions"]
    )
    compliance["allocation_surplus_ratio"] = (
        (compliance["total_allocated"] - compliance["verified_emissions"]) / compliance["verified_emissions"]
    )
    features_df, feature_cols = engineer_features(weekly_price, compliance)
    return features_df, feature_cols


def normalize(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-9)


def train_models(features_df, feature_cols):
    X_raw = features_df[feature_cols].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    if_model = IsolationForest(n_estimators=300, contamination=0.1, random_state=RANDOM_STATE)
    if_model.fit(X)
    if_score = -if_model.decision_function(X)

    n_features = X.shape[1]
    bottleneck = max(2, n_features // 2)
    ae_model = MLPRegressor(
        hidden_layer_sizes=(n_features * 2, bottleneck, n_features * 2),
        activation="tanh",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    ae_model.fit(X, X)
    reconstruction = ae_model.predict(X)
    ae_score = np.mean((X - reconstruction) ** 2, axis=1)

    features_df = features_df.copy()
    features_df["if_score"] = if_score
    features_df["ae_score"] = ae_score
    features_df["composite_anomaly_score"] = 0.5 * normalize(if_score) + 0.5 * normalize(ae_score)
    return features_df, if_model, ae_model, scaler, X


def score_with_models(features_df, if_model, ae_model, scaler, feature_cols):
    X_raw = features_df[feature_cols].values
    X = scaler.transform(X_raw)
    if_score = -if_model.decision_function(X)
    reconstruction = ae_model.predict(X)
    ae_score = np.mean((X - reconstruction) ** 2, axis=1)
    features_df = features_df.copy()
    features_df["if_score"] = if_score
    features_df["ae_score"] = ae_score
    features_df["composite_anomaly_score"] = 0.5 * normalize(if_score) + 0.5 * normalize(ae_score)
    return features_df, X


def score_cell_style(val, vmin, vmax):
    if pd.isna(val):
        return ""
    t = (val - vmin) / (vmax - vmin + 1e-9)
    t = min(max(t, 0.0), 1.0)
    c1 = (246, 241, 230)
    c2 = (79, 122, 88)
    r = int(c1[0] + (c2[0] - c1[0]) * t)
    g = int(c1[1] + (c2[1] - c1[1]) * t)
    b = int(c1[2] + (c2[2] - c1[2]) * t)
    text_color = "#0C3B2E" if t < 0.55 else "#FFFFFF"
    return f"background-color: rgb({r},{g},{b}); color: {text_color}"


def style_score_column(dataframe, column):
    vmin = dataframe[column].min()
    vmax = dataframe[column].max()
    styler = dataframe.style
    if hasattr(styler, "map"):
        return styler.map(lambda v: score_cell_style(v, vmin, vmax), subset=[column])
    return styler.applymap(lambda v: score_cell_style(v, vmin, vmax), subset=[column])


def empty_manual_table(n_rows=8):
    return pd.DataFrame({
        MCOL_DATE: pd.Series([pd.NaT] * n_rows, dtype="datetime64[ns]"),
        MCOL_PRICE: pd.Series([np.nan] * n_rows, dtype="float64"),
        MCOL_VOLUME: pd.Series([np.nan] * n_rows, dtype="float64"),
        MCOL_NONCOMPLIANCE: pd.Series([np.nan] * n_rows, dtype="float64"),
        MCOL_GAS: pd.Series([np.nan] * n_rows, dtype="float64"),
        MCOL_COAL: pd.Series([np.nan] * n_rows, dtype="float64"),
        MCOL_POWER: pd.Series([np.nan] * n_rows, dtype="float64"),
        MCOL_WEATHER: pd.Series([np.nan] * n_rows, dtype="float64"),
        MCOL_OPTIONS_OI: pd.Series([np.nan] * n_rows, dtype="float64"),
        MCOL_FUTURES_OI: pd.Series([np.nan] * n_rows, dtype="float64"),
    })


def build_from_manual_table(manual_df):
    """Turns the edited Manual Input table into (weekly_price, feature_cols
    kwargs) ready for engineer_features. Raises ValueError on unusable input."""
    rows = manual_df.dropna(subset=[MCOL_DATE, MCOL_PRICE]).copy()
    if len(rows) == 0:
        raise ValueError("Please fill in at least Date and EUA Price for one or more rows.")

    rows[MCOL_DATE] = pd.to_datetime(rows[MCOL_DATE])
    rows = rows.drop_duplicates(subset=MCOL_DATE).sort_values(MCOL_DATE).set_index(MCOL_DATE)

    weekly_price = rows[[MCOL_PRICE, MCOL_VOLUME]].rename(
        columns={MCOL_PRICE: "price", MCOL_VOLUME: "volume"}
    )
    weekly_price["volume"] = weekly_price["volume"].fillna(0)

    def _series_or_none(col, scale=1.0):
        s = rows[col].dropna()
        if len(s) == 0:
            return None
        return s * scale

    weekly_non_compliance = _series_or_none(MCOL_NONCOMPLIANCE, scale=0.01)  # % -> ratio
    gas_price = _series_or_none(MCOL_GAS)
    coal_price = _series_or_none(MCOL_COAL)
    power_price = _series_or_none(MCOL_POWER)
    weather = _series_or_none(MCOL_WEATHER)
    options_oi = _series_or_none(MCOL_OPTIONS_OI)
    futures_oi = _series_or_none(MCOL_FUTURES_OI)

    return dict(
        weekly_price=weekly_price,
        weekly_non_compliance=weekly_non_compliance,
        gas_price=gas_price,
        coal_price=coal_price,
        power_price=power_price,
        weather=weather,
        options_oi=options_oi,
        futures_oi=futures_oi,
    )


if "features_df" not in st.session_state:
    st.session_state.features_df = None
    st.session_state.if_model = None
    st.session_state.ae_model = None
    st.session_state.scaler = None
    st.session_state.X = None
    st.session_state.feature_cols = None
    st.session_state.model_status = None

st.markdown(f"""
<div class="topbar">
    <div class="logo-circle">CW</div>
    <div>
        <div class="app-title">CarbonWatch</div>
        <div class="app-sub">EU ETS speculative trading monitor &middot; values shown in &#8377; (INR)</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">Data &amp; Pipeline Setup</div>', unsafe_allow_html=True)
st.markdown('<div class="card-subtitle">Choose how the weekly panel should be built, then run the pipeline.</div>', unsafe_allow_html=True)

data_mode = st.radio(
    "Data input method",
    ["Demo Dataset", "Upload CSV Files", "Manual Input"],
    horizontal=True,
    label_visibility="collapsed",
)

eu_ets_file = daily_file = monthly_file = None
manual_edited_df = None

if data_mode == "Upload CSV Files":
    st.caption(
        "Provide at least one price file (Daily or Monthly). The EU-ETS compliance file is "
        "optional — without it, the two compliance-linked features are skipped and the model "
        "runs on price/volume features only. Source prices are treated as EUR and converted to "
        "INR below."
    )
    with st.expander("Currency conversion", expanded=False):
        eur_inr_rate = st.number_input(
            "EUR → INR rate", min_value=1.0, value=DEFAULT_EUR_INR_RATE, step=0.5,
            key="eur_inr_rate_upload",
            help="Reference rate used to convert EUR-denominated source prices to INR for display and modeling.",
        )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="field-label">EU-ETS Compliance CSV (optional)</div>', unsafe_allow_html=True)
        eu_ets_file = st.file_uploader("EU-ETS compliance CSV", type="csv", key="up_eu_ets", label_visibility="collapsed")
    with col2:
        st.markdown('<div class="field-label">Daily EUA Futures CSV</div>', unsafe_allow_html=True)
        daily_file = st.file_uploader("Daily EUA futures CSV", type="csv", key="up_daily", label_visibility="collapsed")
    with col3:
        st.markdown('<div class="field-label">Monthly EUA Futures CSV</div>', unsafe_allow_html=True)
        monthly_file = st.file_uploader("Monthly EUA futures CSV", type="csv", key="up_monthly", label_visibility="collapsed")

elif data_mode == "Manual Input":
    st.caption(
        "Add one row per week you want to analyze. Date and EUA Price are required on every row; "
        "everything else is optional — leave a column blank if you don't have that data, and the "
        "pipeline uses whichever columns are filled in. Enter all prices directly in ₹ (INR)."
    )
    manual_edited_df = st.data_editor(
        empty_manual_table(),
        num_rows="dynamic",
        width='stretch',
        key="manual_input_editor",
        column_config={
            MCOL_DATE: st.column_config.DateColumn(
                "Analysis Date / Week", help="Week or date range this row represents", required=True,
            ),
            MCOL_PRICE: st.column_config.NumberColumn(
                "EUA Price (₹/tonne)", help="Carbon allowance price", required=True, min_value=0.0,
            ),
            MCOL_VOLUME: st.column_config.NumberColumn(
                "EUA Trading Volume", help="Number of contracts/tonnes traded", min_value=0.0,
            ),
            MCOL_NONCOMPLIANCE: st.column_config.NumberColumn(
                "EUTL Account Data — Non-Compliance %", help="Compliance/non-compliance account activity, 0-100",
                min_value=0.0, max_value=100.0,
            ),
            MCOL_GAS: st.column_config.NumberColumn(
                "Natural Gas Price (₹/MWh)", help="Fundamental energy-market factor", min_value=0.0,
            ),
            MCOL_COAL: st.column_config.NumberColumn(
                "Coal Price (₹/tonne)", help="Fundamental factor", min_value=0.0,
            ),
            MCOL_POWER: st.column_config.NumberColumn(
                "Electricity Price (₹/MWh)", help="Fundamental factor", min_value=0.0,
            ),
            MCOL_WEATHER: st.column_config.NumberColumn(
                "Weather Data", help="Temperature / heating- or cooling-demand indicator",
            ),
            MCOL_OPTIONS_OI: st.column_config.NumberColumn(
                "Options Open Interest", help="Number of open option contracts — derivatives activity", min_value=0.0,
            ),
            MCOL_FUTURES_OI: st.column_config.NumberColumn(
                "Futures Open Interest / Volume", help="Number of futures contracts — used with options data", min_value=0.0,
            ),
        },
    )

st.markdown("<br>", unsafe_allow_html=True)

if saved_model_files_exist():
    st.markdown('<span class="status-pill status-loaded">Saved models detected — will be reused if the feature set matches</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="status-pill status-trained">No saved models found — new models will be trained and saved automatically</span>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
run_btn = st.button("Run Pipeline", width='stretch')
st.markdown('</div>', unsafe_allow_html=True)

if run_btn:
    try:
        with st.spinner("Processing weekly features and anomaly scores..."):
            if data_mode == "Demo Dataset":
                base_features, feature_cols = generate_demo_data(eur_inr_rate=DEFAULT_EUR_INR_RATE)

            elif data_mode == "Upload CSV Files":
                if not (daily_file or monthly_file):
                    st.error(
                        "Please upload at least one price file (Daily or Monthly EUA Futures CSV). "
                        "Price data is required to build the weekly panel."
                    )
                    st.stop()
                compliance = load_eu_ets_compliance(eu_ets_file) if eu_ets_file else None
                weekly_price = load_price_series(daily_file, monthly_file, eur_inr_rate=eur_inr_rate)
                base_features, feature_cols = engineer_features(weekly_price, compliance)

            else:  # Manual Input
                try:
                    manual_kwargs = build_from_manual_table(manual_edited_df)
                except ValueError as e:
                    st.error(str(e))
                    st.stop()
                base_features, feature_cols = engineer_features(**manual_kwargs)

            if len(feature_cols) == 0:
                st.error("No usable features could be engineered from the supplied data.")
                st.stop()
            if len(base_features) == 0:
                st.error(
                    "After computing rolling-window features (momentum, volatility, volume spikes) "
                    "no complete weeks remained. These need a few weeks of history before and after "
                    "each point — please add more rows/weeks and try again."
                )
                st.stop()
            if len(base_features) < 20:
                st.warning(
                    f"Only {len(base_features)} usable week(s) were found after feature "
                    "engineering. Results may be unstable with so few observations."
                )

            models_ready = saved_model_files_exist()
            reuse_saved = False
            if models_ready:
                try:
                    saved_cols = joblib.load(FEATURES_PATH)
                    reuse_saved = (saved_cols == feature_cols)
                except Exception:
                    reuse_saved = False

            if reuse_saved:
                if_model = joblib.load(IF_PATH)
                ae_model = joblib.load(AE_PATH)
                scaler = joblib.load(SCALER_PATH)
                scored_df, X = score_with_models(base_features, if_model, ae_model, scaler, feature_cols)
                st.session_state.model_status = "Loaded saved models from disk"
            else:
                scored_df, if_model, ae_model, scaler, X = train_models(base_features, feature_cols)
                joblib.dump(if_model, IF_PATH)
                joblib.dump(ae_model, AE_PATH)
                joblib.dump(scaler, SCALER_PATH)
                joblib.dump(feature_cols, FEATURES_PATH)
                if models_ready:
                    st.session_state.model_status = "Feature set changed — trained new models and saved to disk"
                else:
                    st.session_state.model_status = "Trained new models and saved to disk"

            st.session_state.features_df = scored_df
            st.session_state.if_model = if_model
            st.session_state.ae_model = ae_model
            st.session_state.scaler = scaler
            st.session_state.X = X
            st.session_state.feature_cols = feature_cols

        used_labels = ", ".join(FEATURE_LABELS.get(c, c) for c in feature_cols)
        st.success(f"Pipeline complete — {st.session_state.model_status}. Using {len(feature_cols)} feature(s): {used_labels}")
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.stop()

if st.session_state.features_df is None:
    st.info("Choose a data input method above, then click Run Pipeline to activate the dashboard.")
    st.stop()

df = st.session_state.features_df
if_model = st.session_state.if_model
ae_model = st.session_state.ae_model
scaler = st.session_state.scaler
X = st.session_state.X
feature_cols = st.session_state.feature_cols

threshold = df["composite_anomaly_score"].quantile(0.9)
n_flagged = (df["composite_anomaly_score"] >= threshold).sum()
latest = df.iloc[-1]

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Weeks Analyzed</div>
    <div class="metric-value">{len(df):,}</div>
    <div class="metric-delta">{df.index.min().date()} to {df.index.max().date()}</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Flagged Anomalies</div>
    <div class="metric-value">{n_flagged}</div>
    <div class="metric-delta">Top 10% composite score</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Latest Composite Score</div>
    <div class="metric-value">{latest['composite_anomaly_score']:.3f}</div>
    <div class="metric-delta">Week of {df.index[-1].date()}</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Latest EUA Price</div>
    <div class="metric-value">{format_inr(latest['price'])}</div>
    <div class="metric-delta">Volume {latest['volume']:,.0f}</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Anomaly Timeline", "Top Anomalies", "Feature Explorer", "SHAP Explainability",
    "Episode Validation", "Manual Scenario"
])

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["price"], name="EUA Price (INR)",
        line=dict(color=SAGE_DARK, width=2.5), yaxis="y1"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["composite_anomaly_score"], name="Composite Anomaly Score",
        line=dict(color=GOLD, width=2), yaxis="y2", fill="tozeroy",
        fillcolor="rgba(227,160,8,0.10)"
    ))
    top_flags = df[df["composite_anomaly_score"] >= threshold]
    fig.add_trace(go.Scatter(
        x=top_flags.index, y=top_flags["price"], mode="markers", name="Flagged Week",
        marker=dict(color=TAN, size=9, symbol="diamond", line=dict(color=DARKGREEN, width=1)),
        yaxis="y1"
    ))
    fig.update_layout(
        legend=dict(orientation="h", y=1.1, x=0),
        yaxis=dict(title="EUA Price (₹)", side="left"),
        yaxis2=dict(title="Anomaly Score", overlaying="y", side="right", range=[0, 1]),
    )
    st.plotly_chart(style_fig(fig, height=500), width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)

    colA, colB = st.columns(2)
    with colA:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Volume vs Anomaly Score</div>', unsafe_allow_html=True)
        fig2 = px.scatter(
            df, x="volume", y="composite_anomaly_score", color="composite_anomaly_score",
            color_continuous_scale=[CREAM, SAGE, DARKGREEN]
        )
        st.plotly_chart(style_fig(fig2, height=360, legend=False), width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)
    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Score Distribution</div>', unsafe_allow_html=True)
        fig3 = px.histogram(df, x="composite_anomaly_score", nbins=40,
                             color_discrete_sequence=[SAGE])
        fig3.add_vline(x=threshold, line_dash="dash", line_color=TAN)
        st.plotly_chart(style_fig(fig3, height=360, legend=False), width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Top 15 Most Anomalous Weeks</div>', unsafe_allow_html=True)
    top15 = df.sort_values("composite_anomaly_score", ascending=False).head(15)
    display_df = top15[["price", "volume", "if_score", "ae_score", "composite_anomaly_score"]].copy()
    display_df.index = display_df.index.date
    display_df["price"] = display_df["price"].apply(format_inr)
    st.dataframe(style_score_column(display_df, "composite_anomaly_score"), width='stretch', height=460)
    csv_buf = io.StringIO()
    df.to_csv(csv_buf)
    st.download_button("Download Full Scored Dataset (CSV)", csv_buf.getvalue(),
                        file_name="weekly_features_and_scores.csv", mime="text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Behavioural Feature Correlation</div>', unsafe_allow_html=True)
    corr = df[feature_cols + ["composite_anomaly_score"]].corr()
    fig4 = px.imshow(
        corr, text_auto=".2f", color_continuous_scale=[CREAM, SAGE, DARKGREEN], aspect="auto"
    )
    st.plotly_chart(style_fig(fig4, height=500, legend=False), width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    sel_feature = st.selectbox("Explore feature over time", feature_cols,
                                format_func=lambda x: FEATURE_LABELS.get(x, x))
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=df.index, y=df[sel_feature], line=dict(color=SAGE_DARK, width=2), name=sel_feature))
    fig5.add_trace(go.Scatter(
        x=top_flags.index, y=top_flags[sel_feature], mode="markers",
        marker=dict(color=GOLD, size=8, symbol="star"), name="Flagged Week"
    ))
    fig5.update_layout(title=FEATURE_LABELS.get(sel_feature, sel_feature))
    st.plotly_chart(style_fig(fig5, height=400), width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Global Feature Importance (SHAP, Isolation Forest)</div>', unsafe_allow_html=True)
    with st.spinner("Computing SHAP values..."):
        X_df = pd.DataFrame(X, columns=feature_cols, index=df.index)
        explainer = shap.TreeExplainer(if_model)
        shap_values = explainer.shap_values(X_df)
    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_summary = pd.DataFrame({
        "feature": [FEATURE_LABELS.get(c, c) for c in feature_cols],
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=True)
    fig6 = px.bar(
        shap_summary, x="mean_abs_shap", y="feature", orientation="h",
        color="mean_abs_shap", color_continuous_scale=[CREAM, SAGE, DARKGREEN]
    )
    fig6.update_layout(xaxis_title="Mean |SHAP value|", yaxis_title="")
    st.plotly_chart(style_fig(fig6, height=400, legend=False), width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Explain a Specific Week</div>', unsafe_allow_html=True)
    sel_date = st.selectbox("Select week", df.index.date[::-1])
    idx = list(df.index.date).index(sel_date)
    week_shap = shap_values[idx]
    week_df = pd.DataFrame({
        "feature": [FEATURE_LABELS.get(c, c) for c in feature_cols],
        "shap_value": week_shap
    }).sort_values("shap_value")
    fig7 = px.bar(
        week_df, x="shap_value", y="feature", orientation="h",
        color="shap_value", color_continuous_scale=[SAGE, CREAM, GOLD]
    )
    fig7.update_layout(xaxis_title="SHAP contribution", yaxis_title="")
    st.plotly_chart(style_fig(fig7, height=360, legend=False), width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)

with tab5:
    st.markdown('<div class="card-title" style="margin-bottom:16px;">Validation Against Documented Speculative Episodes</div>', unsafe_allow_html=True)
    baseline = df["composite_anomaly_score"].mean()
    cols = st.columns(3)
    for i, (name, (start, end, desc)) in enumerate(DOCUMENTED_EPISODES.items()):
        window = df[(df.index >= start) & (df.index <= end)]
        with cols[i]:
            if len(window) == 0:
                st.markdown(f"""<div class="episode-card">
                <div class="episode-title">{name}</div>
                <div class="episode-sub">{desc}</div>
                <div class="episode-sub" style="margin-top:8px;">No data coverage in this window.</div>
                </div>""", unsafe_allow_html=True)
            else:
                avg_score = window["composite_anomaly_score"].mean()
                lift = (avg_score / baseline - 1) * 100
                st.markdown(f"""<div class="episode-card">
                <div class="episode-title">{name}</div>
                <div class="episode-sub">{desc}</div>
                <div style="font-size:24px;font-weight:800;color:{SAGE_DARK};margin-top:10px;">{avg_score:.3f}</div>
                <div class="episode-sub">avg score, {lift:+.1f}% vs baseline, {len(window)} weeks</div>
                </div>""", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    ep_fig = go.Figure()
    ep_fig.add_trace(go.Scatter(x=df.index, y=df["composite_anomaly_score"],
                                 line=dict(color=SAGE_DARK, width=2), name="Composite Score"))
    colors = [TAN, GOLD, "#C0694F"]
    for i, (name, (start, end, desc)) in enumerate(DOCUMENTED_EPISODES.items()):
        ep_fig.add_vrect(x0=start, x1=end, fillcolor=colors[i % 3], opacity=0.14,
                          line_width=0, annotation_text=name, annotation_position="top left",
                          annotation_font_color=DARKGREEN)
    ep_fig.update_layout(title="Anomaly Score with Documented Episode Windows")
    st.plotly_chart(style_fig(ep_fig, height=440), width='stretch')
    st.markdown('</div>', unsafe_allow_html=True)

with tab6:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Manual Scenario Builder</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-subtitle">Enter feature values by hand — with sliders or typed '
        'numbers — to score a hypothetical week against the current models. The result appears '
        'below once you click Analyze.</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"This scenario uses the {len(feature_cols)} feature(s) the current model was trained on: "
        + ", ".join(FEATURE_LABELS.get(c, c) for c in feature_cols)
    )

    input_mode = st.radio(
        "Input mode",
        ["Sliders", "Number Fields"],
        horizontal=True,
        key="manual_input_mode",
    )

    manual_values = {}
    input_cols = st.columns(2)
    for i, feat in enumerate(feature_cols):
        col_min = float(df[feat].min())
        col_max = float(df[feat].max())
        col_mean = float(df[feat].mean())
        if col_min == col_max:
            col_max = col_min + 1.0
        label = FEATURE_LABELS.get(feat, feat)
        with input_cols[i % 2]:
            st.markdown(f'<div class="field-label">{label}</div>', unsafe_allow_html=True)
            if input_mode == "Sliders":
                manual_values[feat] = st.slider(
                    label, min_value=col_min, max_value=col_max, value=col_mean,
                    key=f"manual_slider_{feat}", label_visibility="collapsed"
                )
            else:
                manual_values[feat] = st.number_input(
                    label, min_value=col_min, max_value=col_max, value=col_mean,
                    key=f"manual_number_{feat}", label_visibility="collapsed"
                )

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("Analyze Scenario", width='stretch', key="analyze_manual")
    st.markdown('</div>', unsafe_allow_html=True)

    if analyze_btn:
        manual_row = pd.DataFrame([manual_values])[feature_cols]
        X_manual = scaler.transform(manual_row.values)

        if_score_manual = -if_model.decision_function(X_manual)[0]
        reconstruction_manual = ae_model.predict(X_manual)
        ae_score_manual = float(np.mean((X_manual - reconstruction_manual) ** 2))

        if_min, if_max = df["if_score"].min(), df["if_score"].max()
        ae_min, ae_max = df["ae_score"].min(), df["ae_score"].max()
        if_norm = min(max((if_score_manual - if_min) / (if_max - if_min + 1e-9), 0.0), 1.0)
        ae_norm = min(max((ae_score_manual - ae_min) / (ae_max - ae_min + 1e-9), 0.0), 1.0)
        composite_manual = 0.5 * if_norm + 0.5 * ae_norm

        percentile = (df["composite_anomaly_score"] < composite_manual).mean() * 100

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Scenario Result</div>', unsafe_allow_html=True)

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Composite Anomaly Score</div>
            <div class="metric-value">{composite_manual:.3f}</div>
            <div class="metric-delta">{'Above' if composite_manual >= threshold else 'Below'} flag threshold ({threshold:.3f})</div></div>""", unsafe_allow_html=True)
        with rc2:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Historical Percentile</div>
            <div class="metric-value">{percentile:.1f}%</div>
            <div class="metric-delta">vs {len(df):,} observed weeks</div></div>""", unsafe_allow_html=True)
        with rc3:
            verdict = "Flagged as Speculative" if composite_manual >= threshold else "Within Normal Range"
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Verdict</div>
            <div class="metric-value" style="font-size:20px;">{verdict}</div>
            <div class="metric-delta">Isolation Forest {if_norm:.2f} / Autoencoder {ae_norm:.2f}</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=composite_manual,
            number={"valueformat": ".3f"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": SAGE_DARK},
                "steps": [
                    {"range": [0, float(threshold)], "color": "rgba(109,151,115,0.18)"},
                    {"range": [float(threshold), 1], "color": "rgba(227,160,8,0.25)"},
                ],
                "threshold": {
                    "line": {"color": TAN, "width": 3},
                    "value": float(threshold),
                },
            },
        ))
        st.plotly_chart(style_indicator_fig(gauge, height=280), width='stretch')

        manual_shap = explainer.shap_values(pd.DataFrame(X_manual, columns=feature_cols))[0]
        manual_shap_df = pd.DataFrame({
            "feature": [FEATURE_LABELS.get(c, c) for c in feature_cols],
            "shap_value": manual_shap
        }).sort_values("shap_value")
        fig_manual_shap = px.bar(
            manual_shap_df, x="shap_value", y="feature", orientation="h",
            color="shap_value", color_continuous_scale=[SAGE, CREAM, GOLD]
        )
        fig_manual_shap.update_layout(xaxis_title="SHAP contribution to anomaly score", yaxis_title="")
        st.plotly_chart(style_fig(fig_manual_shap, height=320, legend=False), width='stretch')

        st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footer-tag">CarbonWatch, Isolation Forest + Autoencoder Ensemble &middot; prices in &#8377; (INR) '
    '&middot; extendable to UK / India / China ETS &middot; supports partial data uploads and manual data entry</div>',
    unsafe_allow_html=True,
)