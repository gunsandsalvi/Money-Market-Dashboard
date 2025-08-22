# streamlit_money_market_dashboard_v4.py
"""
Funding / Credit / Liquidity Stress Dashboard (v4)
- Public-data-only (FRED + CSV uploads / CSV URLs)
- No synthetic data
- Friendly labels, recent-focused horizons, gauges + fallback metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List

# Defensive imports for visualization / optional libs
PLOTLY = False
ALTAIR = False
FREDAPI = False
SKLEARN = False
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY = True
except Exception:
    PLOTLY = False

try:
    import altair as alt
    ALTAIR = True
except Exception:
    ALTAIR = False

try:
    from fredapi import Fred
    FREDAPI = True
except Exception:
    FREDAPI = False

try:
    from sklearn.decomposition import PCA
    SKLEARN = True
except Exception:
    SKLEARN = False

st.set_page_config(page_title="Funding/Credit/Liquidity Stress Dashboard", layout="wide")

# --------------------
# Helpers: cleaning & IO
# --------------------
def clean_series(s: pd.Series, name: str) -> Optional[pd.Series]:
    if s is None:
        return None
    try:
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index, errors="coerce")
        s = s[~s.index.isna()].copy()
        s = s.sort_index()
        s = s[~s.index.duplicated(keep="last")]
        s = pd.to_numeric(s, errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        s.name = name
        # minimal winsorization only for very long series (optional)
        if s.size >= 365:
            lo, hi = s.quantile(0.001), s.quantile(0.999)
            s = s.clip(lower=lo, upper=hi)
        return s
    except Exception:
        return None

def read_csv_series(uploaded, name: str) -> Optional[pd.Series]:
    try:
        df = pd.read_csv(uploaded)
        df.columns = [c.strip().lower() for c in df.columns]
        if 'date' not in df.columns or 'value' not in df.columns:
            return None
        s = pd.Series(df['value'].values, index=pd.to_datetime(df['date'], errors='coerce'), name=name)
        return clean_series(s, name)
    except Exception:
        return None

def fetch_public_csv(url: str, name: str) -> Optional[pd.Series]:
    try:
        df = pd.read_csv(url)
        df.columns = [c.strip().lower() for c in df.columns]
        if 'date' not in df.columns or 'value' not in df.columns:
            if len(df.columns) >= 2:
                df = df.rename(columns={df.columns[0]:'date', df.columns[1]:'value'})
            else:
                return None
        s = pd.Series(df['value'].values, index=pd.to_datetime(df['date'], errors='coerce'), name=name)
        return clean_series(s, name)
    except Exception:
        return None

@st.cache_data(ttl=900)
def load_fred_series(series_id: str, api_key: str) -> Optional[pd.Series]:
    if not (FREDAPI and api_key):
        return None
    try:
        fred = Fred(api_key=api_key)
        raw = fred.get_series(series_id)
        s = pd.Series(raw.values, index=pd.to_datetime(raw.index), name=series_id)
        return clean_series(s, series_id)
    except Exception:
        return None

# --------------------
# Friendly names and FRED mapping
# --------------------
FRIENDLY = {
    'sofr': 'SOFR (Overnight Secured Financing Rate)',
    'fedfunds': 'Effective Fed Funds Rate',
    'dtb1': '1M Treasury Bill Yield',
    'dtb3': '3M Treasury Bill Yield',
    'dtb6': '6M Treasury Bill Yield',
    'dtb12': '12M Treasury Bill Yield',
    'tedrate': 'TED Spread (Libor - T-Bill)',
    'totalcp': 'Total Commercial Paper Outstanding',
    'cpn3m': 'Commercial Paper 3M (Non-Financial)',
    'cpf3m': 'Commercial Paper 3M (Financial)',
    'repo_spread': 'Repo Spread (user CSV or URL)',
    'fra_ois': 'FRA - OIS Spread (user CSV or URL)',
    'mmf_flows': 'Money Market Fund Net Flows (user CSV or URL)',
    'ig_oas': 'IG Option-Adjusted Spread (broad)',
    'hy_oas': 'HY Option-Adjusted Spread (broad)',
}

FRED_SERIES = {
    'sofr': 'SOFR',
    'fedfunds': 'FEDFUNDS',
    'dtb1': 'DTB1',
    'dtb3': 'DTB3',
    'dtb6': 'DTB6',
    'dtb12': 'DTB12',
    'tedrate': 'TEDRATE',
    'totalcp': 'TOTALCP',
    'cpn3m': 'CPN3M',
    'cpf3m': 'CPF3M',
    # credit spreads (may or may not be present depending on FRED collection)
    'ig_oas': 'BAMLC0A0CM',
    'hy_oas': 'BAMLH0A0HYM2',
}

# --------------------
# Sidebar inputs
# --------------------
st.sidebar.title("Inputs & Settings")
st.sidebar.write("Provide FRED API key (secrets/env/sidebar), or upload CSVs / paste public CSV URLs.")

# FRED key resolution: st.secrets -> env -> input
fred_key = None
if hasattr(st, 'secrets') and isinstance(st.secrets, dict) and st.secrets.get("FRED_API_KEY"):
    fred_key = st.secrets.get("FRED_API_KEY")
fred_key = fred_key or os.environ.get("FRED_API_KEY")
fk = st.sidebar.text_input("FRED API key (optional)", value=fred_key or "")
fred_key = fk or fred_key

# public CSV URLs
st.sidebar.subheader("Public CSV URLs (direct .csv)")
url_repo = st.sidebar.text_input("Repo spread CSV URL")
url_fra = st.sidebar.text_input("FRA-OIS CSV URL")
url_cp = st.sidebar.text_input("CP spread CSV URL")
url_mmf = st.sidebar.text_input("MMF flows CSV URL")

st.sidebar.subheader("Upload CSVs (date,value)")
up_sofr = st.sidebar.file_uploader("SOFR CSV (date,value)", type=['csv'])
up_dtb1 = st.sidebar.file_uploader("1M T-Bill CSV (date,value)", type=['csv'])
up_fedfunds = st.sidebar.file_uploader("Fed Funds CSV (date,value)", type=['csv'])
up_repo = st.sidebar.file_uploader("Repo spread CSV (date,value)", type=['csv'])
up_fra = st.sidebar.file_uploader("FRA-OIS CSV (date,value)", type=['csv'])
up_cp = st.sidebar.file_uploader("CP spread CSV (date,value)", type=['csv'])
up_mmf = st.sidebar.file_uploader("MMF flows CSV (date,value)", type=['csv'])

st.sidebar.markdown("---")
horizon = st.sidebar.radio("Horizon (charts & recent stats)", options=["6M", "1Y", "2Y", "5Y"], index=1)
roll_window = st.sidebar.slider("Rolling window for z-scores (days)", min_value=20, max_value=120, value=60)
compute_pca = st.sidebar.checkbox("Compute PCA composite (optional)", value=True)

# compute horizon days
HORIZON_DAYS = {"6M": 182, "1Y": 365, "2Y": 365*2, "5Y": 365*5}
h_days = HORIZON_DAYS.get(horizon, 365)

# --------------------
# Data ingestion
# --------------------
series: Dict[str, pd.Series] = {}

# 1) load FRED series if key provided
if fred_key:
    for n, s_id in FRED_SERIES.items():
        s = load_fred_series(s_id, fred_key)
        if s is not None:
            s.name = n
            series[n] = s

# 2) fetch public CSV URLs if given
url_map = {'repo_spread': url_repo, 'fra_ois': url_fra, 'cp_spread': url_cp, 'mmf_flows': url_mmf}
for name, url in url_map.items():
    if url and isinstance(url, str) and url.strip():
        s = fetch_public_csv(url.strip(), name)
        if s is not None:
            series[name] = s

# 3) uploaded CSVs (uploads override fetched)
upload_map = [
    ('sofr', up_sofr), ('dtb1', up_dtb1), ('fedfunds', up_fedfunds),
    ('repo_spread', up_repo), ('fra_ois', up_fra), ('cp_spread', up_cp), ('mmf_flows', up_mmf)
]
for name, up in upload_map:
    if up is not None:
        s = read_csv_series(up, name)
        if s is not None:
            series[name] = s

# If nothing loaded, show help and stop
if not series:
    st.title("Funding / Credit / Liquidity Stress Dashboard")
    st.info("No series loaded. Provide a FRED API key (sidebar) or upload/paste CSVs for the indicators.")
    st.stop()

# Align to business-day index over union of available data
all_dates = sorted({d for s in series.values() for d in s.index})
start, end = all_dates[0], all_dates[-1]
bidx = pd.bdate_range(start=start, end=end)
for k in list(series.keys()):
    series[k] = series[k].reindex(bidx)

df = pd.DataFrame(series)

# --------------------
# Compute recent window trimmed df for plotting & stats
# --------------------
end_dt = df.index.max()
start_dt = end_dt - pd.Timedelta(days=h_days)
df_recent = df.loc[df.index >= start_dt].copy()

# --------------------
# Z-scores (rolling)
# --------------------
z = pd.DataFrame(index=df.index)
for col in df.columns:
    z[col] = df[col].rolling(window=roll_window, min_periods=max(5, roll_window//3)).apply(
        lambda x: (x[-1] - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) != 0 else np.nan)
    )

# sign-adjust mmf_flows (large negative outflows -> positive stress)
if 'mmf_flows' in z.columns:
    z['mmf_flows'] = -z['mmf_flows']

# --------------------
# Scoring (Funding, Credit, Liquidity)
# --------------------
def mean_abs_z(cols: List[str]) -> float:
    """Return mean absolute z of the latest available indicators among cols."""
    present = [c for c in cols if c in z.columns]
    if not present:
        return np.nan
    latest = z[present].iloc[-1].abs().dropna()
    if latest.empty:
        return np.nan
    return latest.mean()

# Category definitions
FUNDING_COLS = ['sofr', 'repo_spread', 'fra_ois', 'fedfunds', 'dtb1', 'dtb3']
CREDIT_COLS = ['ig_oas', 'hy_oas']
LIQUIDITY_COLS = ['mmf_flows', 'totalcp', 'cpn3m', 'cpf3m']

fund_z = mean_abs_z(FUNDING_COLS)
cred_z = mean_abs_z(CREDIT_COLS)
liq_z = mean_abs_z(LIQUIDITY_COLS)

def z_to_score(zval: float) -> int:
    # map absolute z = 0 -> 0, z = 3 -> 100 (cap)
    if pd.isna(zval):
        return 0
    return int(np.clip((zval / 3.0) * 100, 0, 100))

fund_score = z_to_score(fund_z)
cred_score = z_to_score(cred_z)
liq_score = z_to_score(liq_z)

# Combined weighted: 40% funding, 40% credit, 20% liquidity
combined_score = int(np.clip(0.4 * fund_score + 0.4 * cred_score + 0.2 * liq_score, 0, 100))

# 1-month delta helper (compare mean abs z now vs 21 bdays ago)
def delta_1m(cols: List[str]) -> Optional[float]:
    idx = z.index
    if len(idx) < 22:
        return None
    now = z[cols].iloc[-1].abs().dropna().mean() if any(c in z.columns for c in cols) else None
    past_idx = idx.get_loc(idx[-1] - pd.Timedelta(days=21), method='nearest') if len(idx) > 21 else 0
    past = z[cols].iloc[past_idx].abs().dropna().mean() if any(c in z.columns for c in cols) else None
    if now is None or past is None:
        return None
    return now - past

fund_delta = delta_1m(FUNDING_COLS)
cred_delta = delta_1m(CREDIT_COLS)
liq_delta = delta_1m(LIQUIDITY_COLS)

# --------------------
# UI: Top summary (three meters + combined)
# --------------------
st.title("Funding / Credit / Liquidity Stress Dashboard")
st.markdown(f"**Horizon:** last {horizon} (charts & recent stats). Rolling z-window = {roll_window} business days.")

# color helper
def color_for_score(s:int)->str:
    if s <= 40: return "green"
    if s <= 70: return "orange"
    return "red"

# Layout: 4 cards across (responsive)
c1, c2, c3, c4 = st.columns([1,1,1,1])

def show_gauge(col, label, score, delta=None):
    color = color_for_score(score)
    if PLOTLY:
        # simple radial gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x':[0,1],'y':[0,1]},
            title={'text': label},
            gauge={'axis': {'range':[0,100]},
                   'bar': {'color': color},
                   'steps': [{'range':[0,40],'color':'#d4f7dc'}, {'range':[40,70],'color':'#fff0c2'}, {'range':[70,100],'color':'#ffd6d6'}]}
        ))
        fig.update_layout(height=200, margin=dict(l=10,r=10,t=30,b=0))
        col.plotly_chart(fig, use_container_width=True)
        if delta is not None:
            col.caption(f"1M Δ (mean |z|): {delta:+.2f}" if delta is not None else "")
    else:
        # fallback: big metric + colored background
        col.markdown(f"**{label}**")
        col.metric("", f"{score}", delta=f"{delta:+.2f}" if delta is not None and not pd.isna(delta) else "")
        col.markdown(f"<div style='width:100%;height:8px;background:{color};border-radius:4px'></div>", unsafe_allow_html=True)

show_gauge(c1, "Funding Stress (0-100)", fund_score, fund_delta)
show_gauge(c2, "Credit Stress (0-100)", cred_score, cred_delta)
show_gauge(c3, "Liquidity Stress (0-100)", liq_score, liq_delta)
show_gauge(c4, "Combined Stress Index (0-100)", combined_score, None)

st.markdown("---")

# --------------------
# Recent charts (clipped to horizon)
# --------------------
st.header("Recent Indicators (focused view)")
left_col, right_col = st.columns([2,1])

with left_col:
    st.subheader("Funding Indicators")
    funding_plot_cols = [c for c in FUNDING_COLS if c in df_recent.columns]
    # rename friendly
    df_plot = df_recent[funding_plot_cols].rename(columns=lambda x: FRIENDLY.get(x, x))
    plot_timeseries(df_plot, list(df_plot.columns), title=f"Funding Rates & Spreads — last {horizon}")

    st.subheader("Credit Indicators")
    cred_plot_cols = [c for c in CREDIT_COLS if c in df_recent.columns]
    df_cred = df_recent[cred_plot_cols].rename(columns=lambda x: FRIENDLY.get(x, x))
    plot_timeseries(df_cred, list(df_cred.columns), title=f"Credit Spreads — last {horizon}")

with right_col:
    st.subheader("Liquidity Indicators")
    liq_plot_cols = [c for c in LIQUIDITY_COLS if c in df_recent.columns]
    df_liq = df_recent[liq_plot_cols].rename(columns=lambda x: FRIENDLY.get(x, x))
    plot_timeseries(df_liq, list(df_liq.columns), title=f"Liquidity Measures — last {horizon}")

# --------------------
# Heatmap of z-scores (recent 6 months by default)
# --------------------
st.markdown("---")
st.header("Recent Z-score Heatmap (grouped)")
# take last 120 business days (~6 months)
heat_window = min(len(z), 120)
z_recent = z.tail(heat_window)
# order rows by category
rows = []
for cat in [FUNDING_COLS, CREDIT_COLS, LIQUIDITY_COLS]:
    rows.extend([c for c in cat if c in z_recent.columns])
if rows:
    heat_df = z_recent[rows].T
    heat_df.index = [FRIENDLY.get(r, r) for r in heat_df.index]
    heatmap(heat_df, title="Z-scores (recent) — rows grouped by category")
else:
    st.info("Not enough z-score series to build heatmap.")

# --------------------
# Alerts panel (recent breaches)
# --------------------
st.markdown("---")
st.header("Alerts — Recent breaches (|z| ≥ 2)")
alerts = []
if not z.empty:
    latest_z = z.iloc[-1]
    for c in latest_z.index:
        val = latest_z[c]
        if pd.notna(val) and abs(val) >= 2.0:
            cat = ("Funding" if c in FUNDING_COLS else "Credit" if c in CREDIT_COLS else "Liquidity" if c in LIQUIDITY_COLS else "Other")
            alerts.append((c, val, cat))
if alerts:
    for (c, val, cat) in alerts:
        label = FRIENDLY.get(c, c)
        if abs(val) >= 3.0:
            emoji = "🔴"
        else:
            emoji = "⚠️"
        st.error(f"{emoji} {label} | {cat} | z = {val:.2f}")
else:
    st.success("No recent z-score breaches (|z| ≥ 2).")

# --------------------
# Exports & downloads
# --------------------
st.markdown("---")
st.subheader("Export")
if st.button("Download latest raw indicators CSV"):
    tmp = df.reset_index().rename(columns={'index':'date'})
    st.download_button("Download indicators CSV", tmp.to_csv(index=False).encode('utf-8'), file_name='indicators_latest.csv', mime='text/csv')
if st.button("Download latest z-scores CSV"):
    tmp2 = z.reset_index().rename(columns={'index':'date'})
    st.download_button("Download z-scores CSV", tmp2.to_csv(index=False).encode('utf-8'), file_name='z_scores.csv', mime='text/csv')

st.caption("Notes: All series are public (FRED or user-provided CSVs/URLs). No synthetic data is used. Use the sidebar to add/upload series or change horizon/rolling window.")
