"""
Funding & Credit Market Stress Dashboard — Public-data + Clean ingestion (v3)

- No synthetic data.
- Uses public FRED series when you provide a FRED API key (or uploads / public CSV URLs).
- Cleans all input series before use.
- Optional Plotly/Altair/Scikit-learn support (app runs without them, with graceful fallbacks).
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Optional, Dict, List

# ---------- Optional libraries (defensive imports) ----------
PLOTLY_AVAILABLE = False
ALTAIR_AVAILABLE = False
FRED_AVAILABLE = False
SKLEARN_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

try:
    import altair as alt
    ALTAIR_AVAILABLE = True
except Exception:
    ALTAIR_AVAILABLE = False

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except Exception:
    FRED_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ---------- Config ----------
st.set_page_config(page_title="Funding & Credit Stress — Public Data", layout="centered", initial_sidebar_state="expanded")

# Optionally hard-code a FRED key here (NOT recommended for public repos)
HARDCODED_FRED_KEY = ""  # <-- put your key here if you insist; better: use st.secrets["FRED_API_KEY"]

# ---------- Helpers: data cleaning ----------
def clean_series(s: pd.Series, name: str) -> Optional[pd.Series]:
    """Standardize a timeseries: parse to datetime index, numeric values, sort, dedupe, drop non-finite, and optionally winsorize."""
    if s is None:
        return None
    try:
        # Ensure index is datetime and sorted
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index, errors='coerce')
        s = s[~s.index.isna()].copy()
        s = s.sort_index()
        # Deduplicate by index (keep last)
        s = s[~s.index.duplicated(keep='last')]
        # Coerce values to numeric
        s = pd.to_numeric(s, errors='coerce')
        # Drop non-finite
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        s.name = name
        # Optional gentle winsorization to cap extreme one-off spikes
        if s.size >= 30:
            q_low, q_hi = s.quantile(0.001), s.quantile(0.999)
            s = s.clip(lower=q_low, upper=q_hi)
        return s
    except Exception:
        return None

def read_csv_series(file_or_bytes, name: str, date_col: str = 'date', value_col: str = 'value') -> Optional[pd.Series]:
    try:
        df = pd.read_csv(file_or_bytes)
        df.columns = [c.strip().lower() for c in df.columns]
        if date_col not in df.columns or value_col not in df.columns:
            return None
        s = pd.Series(df[value_col].values, index=pd.to_datetime(df[date_col], errors='coerce'), name=name)
        return clean_series(s, name)
    except Exception:
        return None

def fetch_public_csv(url: str, name: str, date_col: str = 'date', value_col: str = 'value') -> Optional[pd.Series]:
    try:
        df = pd.read_csv(url)
        df.columns = [c.strip().lower() for c in df.columns]
        if date_col not in df.columns or value_col not in df.columns:
            # try first two columns as (date,value)
            if len(df.columns) >= 2:
                df = df.rename(columns={df.columns[0]: 'date', df.columns[1]: 'value'})
            else:
                return None
        s = pd.Series(df['value'].values, index=pd.to_datetime(df['date'], errors='coerce'), name=name)
        return clean_series(s, name)
    except Exception:
        return None

# ---------- Plot helpers ----------
def plot_timeseries(df: pd.DataFrame, cols: List[str], title: str, height: int = 300):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        st.info(f"No data available for: {title}")
        return
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        for c in cols:
            fig.add_trace(go.Scatter(x=df.index, y=df[c], name=c))
        fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        return
    if ALTAIR_AVAILABLE:
        m = df[cols].reset_index().melt(id_vars='index', var_name='series', value_name='value').rename(columns={'index':'date'})
        chart = alt.Chart(m).mark_line().encode(x='date:T', y='value:Q', color='series:N', tooltip=['date','series','value'])
        st.altair_chart(chart.properties(title=title, height=height), use_container_width=True)
        return
    st.line_chart(df[cols])

def heatmap(df: pd.DataFrame, title: str, height: int = 320):
    if df.empty:
        st.info("No data for heatmap")
        return
    if PLOTLY_AVAILABLE:
        fig = px.imshow(df, aspect='auto', labels=dict(x='Date', y='Series', color='z'))
        fig.update_layout(title=title, height=height)
        st.plotly_chart(fig, use_container_width=True)
        return
    if ALTAIR_AVAILABLE:
        m = df.reset_index().melt(id_vars='index', var_name='series', value_name='z').rename(columns={'index':'date'})
        chart = alt.Chart(m).mark_rect().encode(x='date:T', y='series:N', color='z:Q')
        st.altair_chart(chart.properties(title=title, height=height), use_container_width=True)
        return
    st.dataframe(df)

# ---------- Sidebar: inputs ----------
st.sidebar.title("Data Sources (Public)")
st.sidebar.caption("This app only uses public data. Provide a FRED key or CSV/URLs.")

# FRED API key priority: secrets -> env -> hardcoded -> sidebar input
fred_key = None
fred_key = fred_key or (st.secrets.get('FRED_API_KEY') if hasattr(st, 'secrets') else None)
fred_key = fred_key or os.environ.get('FRED_API_KEY')
fred_key = fred_key or (HARDCODED_FRED_KEY if HARDCODED_FRED_KEY else None)
fk_input = st.sidebar.text_input("FRED API key (optional)", value=fred_key or "")
fred_key = fk_input or fred_key

# Public CSV URLs (user-provided). These must expose at least two columns: date,value (case-insensitive)
st.sidebar.subheader("Public CSV URLs (optional)")
url_repo = st.sidebar.text_input("Repo spread CSV URL")
url_fra = st.sidebar.text_input("FRA-OIS CSV URL")
url_cp = st.sidebar.text_input("CP spread CSV URL")
url_mmf = st.sidebar.text_input("MMF flows CSV URL")

st.sidebar.subheader("File uploads (CSV)")
up_sofr = st.sidebar.file_uploader("SOFR CSV (date,value)")
up_dtb1 = st.sidebar.file_uploader("1M T-Bill CSV (date,value)")
up_fedfunds = st.sidebar.file_uploader("Fed Funds CSV (date,value)")
up_repo = st.sidebar.file_uploader("Repo spread CSV (date,value)")
up_fra = st.sidebar.file_uploader("FRA-OIS CSV (date,value)")
up_cp = st.sidebar.file_uploader("CP spread CSV (date,value)")
up_mmf = st.sidebar.file_uploader("MMF flows CSV (date,value)")

st.sidebar.markdown("---")
st.sidebar.subheader("Model")
roll_win = st.sidebar.number_input("Rolling window (business days)", min_value=10, max_value=252, value=60)
compute_pca = st.sidebar.checkbox("Compute PCA (requires scikit-learn)", value=True)

# ---------- FRED mapping (public series) ----------
FRED_SERIES: Dict[str, str] = {
    # Funding core
    'sofr': 'SOFR',
    'fedfunds': 'FEDFUNDS',
    'dtb1': 'DTB1',  # 1M T-bill
    'dtb3': 'DTB3',  # 3M T-bill
    'dtb6': 'DTB6',  # 6M T-bill
    'dtb12': 'DTB12',  # 12M T-bill
    'tedrate': 'TEDRATE',  # TED spread
    'totalcp': 'TOTALCP',  # Commercial Paper Outstanding
    'cpn3m': 'CPN3M',     # 3M CP rate, nonfinancial
    'cpf3m': 'CPF3M',     # 3M CP rate, financial
    # Credit spreads (ICE BofA via FRED; tickers may vary)
    'ig_oas': 'BAMLC0A0CM',   # IG OAS (broad)
    'hy_oas': 'BAMLH0A0HYM2', # HY OAS
}

# ---------- Load series ----------
@st.cache_data(ttl=900)
def load_fred_series(series_id: str, api_key: str) -> Optional[pd.Series]:
    if not (FRED_AVAILABLE and api_key):
        return None
    try:
        fred = Fred(api_key=api_key)
        s = fred.get_series(series_id)
        s = pd.Series(s.values, index=pd.to_datetime(s.index), name=series_id)
        return clean_series(s, series_id)
    except Exception:
        return None

series: Dict[str, pd.Series] = {}

# 1) FRED-backed
for name, sid in FRED_SERIES.items():
    s = load_fred_series(sid, fred_key) if fred_key else None
    if s is not None:
        s.name = name
        series[name] = s

# 2) Public CSV URLs (user-specified)
url_map = {
    'repo_spread': url_repo,
    'fra_ois': url_fra,
    'cp_spread': url_cp,
    'mmf_flows': url_mmf,
}
for name, url in url_map.items():
    if url:
        s = fetch_public_csv(url, name)
        if s is not None:
            series[name] = s

# 3) File uploads
iomap = [
    ('sofr', up_sofr), ('dtb1', up_dtb1), ('fedfunds', up_fedfunds),
    ('repo_spread', up_repo), ('fra_ois', up_fra), ('cp_spread', up_cp), ('mmf_flows', up_mmf)
]
for name, up in iomap:
    if up is not None:
        s = read_csv_series(up, name)
        if s is not None:
            series[name] = s

# Build a common business-day index only over the range where *any* data exists
if series:
    all_dates = sorted({ts for s in series.values() for ts in s.index})
    if all_dates:
        start, end = all_dates[0], all_dates[-1]
        bidx = pd.bdate_range(start=start, end=end)
        for k in list(series.keys()):
            series[k] = series[k].reindex(bidx)

# Compose DataFrame
if series:
    df = pd.DataFrame(series)
else:
    df = pd.DataFrame()

# ---------- UI ----------
st.title("Funding & Credit Stress — Public Data (v3)")
if not fred_key:
    st.info("Tip: add a FRED API key in the sidebar (or via Secrets/ENV) to auto-fetch public series.")

if df.empty:
    st.warning("No data loaded yet. Provide a FRED key and/or CSVs/URLs. Nothing will be fabricated.")
    st.stop()

# ---------- Modeling (only on available data) ----------
# Rolling z-scores
z = pd.DataFrame(index=df.index)
for col in df.columns:
    z[col] = df[col].rolling(window=roll_win, min_periods=max(5, roll_win//2)).apply(
        lambda x: (x[-1] - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) != 0 else np.nan)
    )

# Invert MMF flows sign if present
if 'mmf_flows' in z.columns:
    z['mmf_flows'] = -z['mmf_flows']

# PCA composite
pc1 = pd.Series(index=df.index, data=np.nan, name='pc1')
if compute_pca and SKLEARN_AVAILABLE:
    valid = [c for c in z.columns if z[c].notna().sum() > roll_win]
    if len(valid) >= 2:
        xin = z[valid].dropna()
        try:
            p = PCA(n_components=1)
            comp = p.fit_transform(xin.fillna(0))
            pc1.loc[xin.index] = comp[:,0]
        except Exception:
            pc1[:] = np.nan

# Simple composite = mean abs z of latest
avail_cols = [c for c in z.columns if z[c].notna().sum() > 0]
if avail_cols:
    composite_simple = z[avail_cols].abs().iloc[-1].fillna(0).mean()
else:
    composite_simple = np.nan

# Scale to 0-100 heuristics (only if not nan)
comp_scaled = int(np.clip((composite_simple / 3.0) * 100, 0, 100)) if pd.notna(composite_simple) else 0
pc1_nonempty = pc1.dropna()
pc1_scaled = 50
if not pc1_nonempty.empty and np.std(pc1_nonempty) != 0:
    last_val = float(pc1_nonempty.iloc[-1])
    pc1_z = (last_val - np.mean(pc1_nonempty)) / np.std(pc1_nonempty)
    pc1_scaled = int(np.clip(pc1_z * 20 + 50, 0, 100))

# Mahalanobis (requires at least 2 series with recent data)
maha_score = 0
try:
    Xw = z.dropna(axis=1, how='all').tail(roll_win)
    Xw = Xw.dropna()
    if Xw.shape[1] >= 2 and Xw.shape[0] >= Xw.shape[1] + 2:
        mu = Xw.mean()
        cov = np.cov(Xw.T)
        cov += np.eye(cov.shape[0]) * 1e-8
        invcov = np.linalg.pinv(cov)
        last = Xw.iloc[-1].values - mu.values
        md = float(np.sqrt(np.dot(np.dot(last.T, invcov), last)))
        maha_score = int(np.clip((md / 3.0) * 100, 0, 100))
except Exception:
    maha_score = 0

final_score = int(np.clip(0.5 * comp_scaled + 0.4 * pc1_scaled + 0.1 * maha_score, 0, 100))

# ---------- Display ----------
cols = st.columns(4)
cols[0].metric("Composite (simple)", f"{comp_scaled}")
cols[1].metric("PCA-based", f"{pc1_scaled}")
cols[2].metric("Mahalanobis", f"{maha_score}")
cols[3].metric("Final stress 0-100", f"{final_score}")

st.markdown("---")

# Funding block
fund_cols = [c for c in ['sofr','fedfunds','dtb1','dtb3','dtb6','dtb12','tedrate','totalcp','cpn3m','cpf3m','repo_spread','fra_ois','mmf_flows'] if c in df.columns]
plot_timeseries(df, fund_cols, title="Funding & CP indicators")

# Credit block (if any)
cred_cols = [c for c in ['ig_oas','hy_oas'] if c in df.columns]
if cred_cols:
    plot_timeseries(df, cred_cols, title="Credit OAS (public via FRED)")

# Heatmap of z-scores (last ~120 bdays)
heat = z[avail_cols].tail(120).T if avail_cols else pd.DataFrame()
heatmap(heat, title="Z-score heatmap (recent)")

st.markdown("---")

# Alerts
st.subheader("Alerts")
alerts = []
if comp_scaled >= 80: alerts.append(f"Composite simple high: {comp_scaled}")
if pc1_scaled >= 80: alerts.append(f"PCA-based score high: {pc1_scaled}")
if maha_score >= 70: alerts.append(f"Mahalanobis anomaly high: {maha_score}")
for c in avail_cols:
    val = z[c].iloc[-1]
    if pd.notna(val) and abs(val) >= 2.0:
        alerts.append(f"{c} | |z| >= 2.0: {val:.2f}")

if alerts:
    for a in alerts: st.error(a)
else:
    st.success("No immediate high-stress alerts.")

st.markdown("---")

# Exports
st.subheader("Export")
if st.button("Download latest raw indicators CSV"):
    csv = df.reset_index().rename(columns={'index':'date'}).to_csv(index=False).encode('utf-8')
    st.download_button(label='Download CSV', data=csv, file_name='indicators_latest.csv', mime='text/csv')

if st.button("Download z-scores CSV"):
    csv2 = z.reset_index().rename(columns={'index':'date'}).to_csv(index=False).encode('utf-8')
    st.download_button(label='Download z-scores', data=csv2, file_name='z_scores.csv', mime='text/csv')

st.caption("Note: All data is public. FRED series are fetched with your API key. Repo/FRA/CP/MMF series require your CSV/URL unless available on FRED.")
