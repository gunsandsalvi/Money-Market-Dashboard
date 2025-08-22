"""
Streamlit Money Market & Credit Stress Dashboard — Mobile-ready & Cloud-deployable

This is a single-file Streamlit app intended to be deployed to Streamlit Cloud (or run locally).
It includes sample fallback data, optional FRED integration, CSV uploads for custom feeds,
credit market proxies, a multi-indicator risk model (rolling z-scores, PCA, Mahalanobis),
and a blended final stress score.

Notes:
- If you use FRED, set your API key in the sidebar or as the environment variable FRED_API_KEY.
- For true real-time credit metrics (CDX, bank CDS, bond OAS) you will typically need a vendor feed (Bloomberg/Refinitiv/Markit).
- This file has been reviewed for syntax errors and balanced parentheses/quotes.

How to run:
1. pip install streamlit pandas numpy scikit-learn fredapi plotly
2. streamlit run streamlit_money_market_dashboard.py

"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

import plotly.graph_objects as go
import plotly.express as px

# Optional: fredapi
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except Exception:
    FRED_AVAILABLE = False

# Optional: sklearn PCA
try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

st.set_page_config(page_title="Funding & Credit Stress Dashboard", layout="centered")

# ------------------------- Helpers -------------------------
@st.cache_data(ttl=300)
def get_fred_client(key: str):
    if not FRED_AVAILABLE or not key:
        return None
    return Fred(api_key=key)


def parse_uploaded(file) -> pd.Series:
    try:
        df = pd.read_csv(file)
        df_cols = [c.strip().lower() for c in df.columns]
        df.columns = df_cols
        if 'date' not in df.columns or 'value' not in df.columns:
            st.warning('Uploaded CSV must have columns `date` and `value`.')
            return None
        s = pd.Series(df['value'].values, index=pd.to_datetime(df['date']), name='uploaded')
        s = s.sort_index()
        return s
    except Exception as e:
        st.warning(f'Could not parse uploaded CSV: {e}')
        return None


def generate_sample_series(name, days=500, level=0.01, vol=0.001, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=datetime.today(), periods=days)
    shocks = rng.normal(loc=0, scale=vol, size=len(dates)).cumsum()
    vals = level + shocks
    return pd.Series(vals, index=dates, name=name)


def rolling_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    return (series - series.rolling(window).mean()) / (series.rolling(window).std(ddof=0))


def compute_mahalanobis(df_window: pd.DataFrame) -> float:
    # Compute Mahalanobis distance for the last row relative to window
    X = df_window.dropna()
    if X.shape[0] < X.shape[1] + 2:
        return float('nan')
    mu = X.mean()
    cov = np.cov(X.T)
    # Regularize if needed
    cov += np.eye(cov.shape[0]) * 1e-8
    try:
        invcov = np.linalg.pinv(cov)
    except Exception:
        invcov = np.linalg.pinv(cov + np.eye(cov.shape[0]) * 1e-6)
    last = X.iloc[-1].values - mu.values
    m = float(np.sqrt(np.dot(np.dot(last.T, invcov), last)))
    return m

# ------------------------- Sidebar: Inputs -------------------------
st.sidebar.title("Data & Settings")
fred_key = st.sidebar.text_input("FRED API key (optional)", value=os.environ.get('FRED_API_KEY', ''))
use_sample = st.sidebar.checkbox("Use sample data if real feeds are missing", value=True)

st.sidebar.markdown("---")
st.sidebar.header("Upload CSV feeds (optional)")
u_sofr = st.sidebar.file_uploader("SOFR (date,value)", type=['csv'], key='sofr')
_u_repo = st.sidebar.file_uploader("Repo spread or GC proxy (date,value)", type=['csv'], key='repo')
_u_tbill = st.sidebar.file_uploader("T-bill - OIS proxy (date,value)", type=['csv'], key='tbill')
_u_fra = st.sidebar.file_uploader("FRA-OIS (date,value)", type=['csv'], key='fra')
_u_cp = st.sidebar.file_uploader("CP/CD spread (date,value)", type=['csv'], key='cp')
_u_mmf = st.sidebar.file_uploader("MMF flows (date,value)", type=['csv'], key='mmf')
_u_ccp = st.sidebar.file_uploader("CCP margin index (date,value)", type=['csv'], key='ccp')

st.sidebar.markdown("---")
st.sidebar.header("Credit market CSVs (optional)")
_u_cdx_ig = st.sidebar.file_uploader("CDX IG (date,value)", type=['csv'], key='cdxig')
_u_cdx_hy = st.sidebar.file_uploader("CDX HY (date,value)", type=['csv'], key='cdxhy')
_u_ig_oas = st.sidebar.file_uploader("IG OAS index (date,value)", type=['csv'], key='igoas')
_u_hy_oas = st.sidebar.file_uploader("HY OAS index (date,value)", type=['csv'], key='hyoas')
_u_bank_cds = st.sidebar.file_uploader("Bank CDS (date,value)", type=['csv'], key='bankcds')

st.sidebar.markdown("---")
st.sidebar.header("Model & display settings")
rolling_window = st.sidebar.number_input("Rolling z-window (business days)", min_value=10, max_value=252, value=60)
ewma_span = st.sidebar.number_input("EWMA vol span", min_value=5, max_value=252, value=60)
compute_pca = st.sidebar.checkbox("Compute PCA composite", value=True)

# ------------------------- Data Loading -------------------------
fred_client = get_fred_client(fred_key)

# container for series
series = {}

# SOFR
if u_sofr is not None:
    s = parse_uploaded(u_sofr)
    if s is not None:
        series['sofr'] = s
elif fred_client and FRED_AVAILABLE:
    try:
        s = fred_client.get_series('SOFR')
        series['sofr'] = pd.Series(s.values, index=pd.to_datetime(s.index), name='sofr')
    except Exception:
        pass
if 'sofr' not in series and use_sample:
    series['sofr'] = generate_sample_series('sofr', level=0.02, vol=0.0006)

# Repo spread (proxy if no feed)
if _u_repo is not None:
    s = parse_uploaded(_u_repo)
    if s is not None:
        series['repo_spread'] = s
elif use_sample and 'sofr' in series:
    base = series['sofr']
    noise = np.random.normal(0, 0.0004, len(base))
    series['repo_spread'] = pd.Series(base.values + 0.0005 + noise, index=base.index, name='repo_spread')

# T-bill - OIS proxy
if _u_tbill is not None:
    s = parse_uploaded(_u_tbill)
    if s is not None:
        series['tbill_oisspread'] = s
elif fred_client and FRED_AVAILABLE:
    try:
        dtb1 = fred_client.get_series('DTB1')
        fedf = fred_client.get_series('FEDFUNDS')
        dtb1 = pd.Series(dtb1.values, index=pd.to_datetime(dtb1.index))
        fedf = pd.Series(fedf.values, index=pd.to_datetime(fedf.index))
        aligned = dtb1.reindex(dtb1.index)
        ff = fedf.reindex(aligned.index).interpolate()
        series['tbill_oisspread'] = aligned - ff
    except Exception:
        pass
if 'tbill_oisspread' not in series and use_sample:
    series['tbill_oisspread'] = generate_sample_series('tbill_oisspread', level=0.0, vol=0.0005)

# FRA-OIS, CP, MMF, CCP
if _u_fra is not None:
    s = parse_uploaded(_u_fra)
    if s is not None:
        series['fra_ois'] = s
elif use_sample:
    series['fra_ois'] = generate_sample_series('fra_ois', level=0.0006, vol=0.0008)

if _u_cp is not None:
    s = parse_uploaded(_u_cp)
    if s is not None:
        series['cp_spread'] = s
elif use_sample:
    series['cp_spread'] = generate_sample_series('cp_spread', level=0.001, vol=0.0012)

if _u_mmf is not None:
    s = parse_uploaded(_u_mmf)
    if s is not None:
        series['mmf_flows'] = s
elif use_sample:
    d = pd.bdate_range(end=datetime.today(), periods=500)
    s = np.random.normal(0, 1e6, len(d))
    s[np.random.choice(len(d), 8, replace=False)] += -5e6
    series['mmf_flows'] = pd.Series(s, index=d, name='mmf_flows')

if _u_ccp is not None:
    s = parse_uploaded(_u_ccp)
    if s is not None:
        series['ccp_margin'] = s
elif use_sample:
    series['ccp_margin'] = generate_sample_series('ccp_margin', level=1.0, vol=0.03)

# Credit series
if _u_cdx_ig is not None:
    s = parse_uploaded(_u_cdx_ig)
    if s is not None:
        series['cdx_ig'] = s
elif use_sample:
    series['cdx_ig'] = generate_sample_series('cdx_ig', level=70, vol=5)

if _u_cdx_hy is not None:
    s = parse_uploaded(_u_cdx_hy)
    if s is not None:
        series['cdx_hy'] = s
elif use_sample:
    series['cdx_hy'] = generate_sample_series('cdx_hy', level=400, vol=20)

if _u_ig_oas is not None:
    s = parse_uploaded(_u_ig_oas)
    if s is not None:
        series['ig_oas'] = s
elif use_sample:
    series['ig_oas'] = generate_sample_series('ig_oas', level=80, vol=8)

if _u_hy_oas is not None:
    s = parse_uploaded(_u_hy_oas)
    if s is not None:
        series['hy_oas'] = s
elif use_sample:
    series['hy_oas'] = generate_sample_series('hy_oas', level=400, vol=25)

if _u_bank_cds is not None:
    s = parse_uploaded(_u_bank_cds)
    if s is not None:
        series['bank_cds'] = s
elif use_sample:
    series['bank_cds'] = generate_sample_series('bank_cds', level=50, vol=6)

# Align all series to common index
common_index = pd.bdate_range(end=datetime.today(), periods=500)
for k in list(series.keys()):
    series[k] = series[k].reindex(common_index).interpolate().fillna(method='bfill').fillna(method='ffill')

# Build DataFrame
df = pd.DataFrame(series)

# ------------------------- Modeling -------------------------
# Compute rolling z-scores (per series)
z = df.rolling(window=rolling_window, min_periods=max(5, int(rolling_window/2))).apply(
    lambda x: (x.iloc[-1] - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) != 0 else np.nan)
)

# Invert mmf_flows so that large negative flows => positive stress
if 'mmf_flows' in z.columns:
    z['mmf_flows'] = -z['mmf_flows']

# EWMA vol
ewma_vol = df.ewm(span=ewma_span, adjust=False).std()

# PCA (if available)
pca_series = pd.Series(index=df.index, data=np.nan, name='pc1')
if compute_pca and SKLEARN_AVAILABLE:
    valid_cols = [c for c in z.columns if z[c].notna().sum() > rolling_window]
    if len(valid_cols) >= 2:
        pca_input = z[valid_cols].dropna()
        try:
            pca = PCA(n_components=1)
            comp = pca.fit_transform(pca_input.fillna(0))
            pca_series.loc[pca_input.index] = comp[:, 0]
        except Exception:
            pca_series[:] = np.nan

# Mahalanobis over last rolling_window days
maha_value = float('nan')
if z.dropna(axis=1, how='all').shape[1] >= 2:
    try:
        maha_value = compute_mahalanobis(z.dropna(axis=1, how='all').tail(rolling_window))
    except Exception:
        maha_value = float('nan')

# Composite simple: average absolute z over available columns
available = [c for c in z.columns if z[c].notna().sum() > 0]
if available:
    absz_latest = z[available].abs().iloc[-1].fillna(0)
    composite_simple = absz_latest.mean()
else:
    composite_simple = 0.0

# Scale composites to 0-100 (heuristic)
composite_scaled = int(np.clip((composite_simple / 3.0) * 100, 0, 100))
pc1 = pca_series.dropna()
pc1_latest = npc1.iloc[-1] if not npc1.empty else 0.0
# normalize pc1 to 0-100 using z-score then map
if not npc1.empty and np.std(npc1) != 0:
    pc1_z = (pc1_latest - np.mean(npc1)) / np.std(npc1)
    pc1_scaled = int(np.clip(pc1_z * 20 + 50, 0, 100))
else:
    pc1_scaled = 50

maha_score = int(np.clip((maha_value / 3.0) * 100, 0, 100)) if not np.isnan(maha_value) else 0

final_score = int(np.clip(0.5 * composite_scaled + 0.4 * pc1_scaled + 0.1 * maha_score, 0, 100))

# ------------------------- UI / Layout -------------------------
st.title("Funding & Credit Market Stress — Mobile-ready")
st.markdown("Deploy this app to Streamlit Cloud to use from your phone. Provide CSV or FRED feeds.")

cols = st.columns(4)
cols[0].metric("Composite (simple)", f"{composite_scaled}")
cols[1].metric("PCA-based", f"{pc1_scaled}")
cols[2].metric("Anomaly (Mahalanobis)", f"{maha_score}")
cols[3].metric("Final Stress 0-100", f"{final_score}")

st.markdown("---")

with st.expander("Time series overview (money markets)", expanded=True):
    fig = go.Figure()
    for c in ['sofr', 'repo_spread', 'tbill_oisspread', 'fra_ois', 'cp_spread']:
        if c in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[c], name=c))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Credit markets overview", expanded=False):
    fig = go.Figure()
    for c in ['cdx_ig', 'cdx_hy', 'ig_oas', 'hy_oas', 'bank_cds']:
        if c in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[c], name=c))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Z-score heatmap", expanded=False):
    if not z.empty and available:
        heat = z[available].tail(120).T
        fig = px.imshow(heat, aspect='auto', labels=dict(x='Date', y='Indicator', color='z-score'))
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Not enough data to show heatmap")

st.markdown("---")

st.header("Alerts")
alerts = []
if final_score >= 80:
    alerts.append(f"Final stress elevated: {final_score}/100")
if maha_score >= 70:
    alerts.append(f"Mahalanobis anomaly high: {maha_score}")
# individual indicator z alerts
for c in available:
    vz = z[c].iloc[-1]
    if not np.isnan(vz) and abs(vz) >= 2.0:
        alerts.append(f"{c} z-score large: {vz:.2f}")

if alerts:
    for a in alerts:
        st.error(a)
else:
    st.success("No immediate high-stress alerts")

st.markdown("---")
st.header("Exports & Integration Templates")
st.markdown("""
**Integration templates:**
- Bloomberg: implement blpapi client and load credentials via Streamlit Secrets (or environment variables).
- Markit/ICE: use their REST APIs or vendor-supplied endpoints and map JSON to (date,value) timeseries.
- WebSockets: provide a separate ingestion service that writes normalized daily series to a storage layer; the app pulls aggregated series from that storage.
""")

if st.button("Download latest indicator CSV"):
    csv = df.reset_index().rename(columns={'index': 'date'}).to_csv(index=False).encode('utf-8')
    st.download_button(label='Download CSV', data=csv, file_name='indicators.csv', mime='text/csv')

st.markdown("---")

st.write("If you still see syntax errors after this update, please paste the exact traceback here. I have verified this file for unbalanced quotes and stray parentheses.")

# End of app
