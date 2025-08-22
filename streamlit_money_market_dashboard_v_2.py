"""
Streamlit Money & Credit Market Stress Dashboard (v2)

- Redesigned from scratch to be robust, mobile-friendly, and cloud-deployable.
- Single-file Streamlit app. Deploy to Streamlit Cloud by pushing this file to a GitHub repo
  and pointing Streamlit Cloud to run it.

Features:
- Accepts CSV uploads for each indicator (columns: date,value) or uses sample fallback data.
- Optional FRED integration (if fredapi installed and you provide an API key).
- Optional scikit-learn PCA if installed.
- Robust plotting: uses Plotly if available, else Altair, else Streamlit-native charts.
- Indicators included: SOFR, repo spread, T-bill - OIS, FRA-OIS, CP spread, MMF flows, CCP margin,
  CDX IG/HY, IG/HY OAS, bank CDS.
- Risk model: rolling z-scores, EWMA volatility, PCA composite (if available), Mahalanobis anomaly,
  blended final stress score (0-100).
- Mobile-friendly layout and alerting panel.

How to run:
1) (Optional) create requirements.txt with packages you need (streamlit, pandas, numpy, plotly, fredapi, scikit-learn, altair).
2) streamlit run streamlit_money_market_dashboard_v2.py

Notes:
- This file aims to run even when optional libraries are not installed: it falls back gracefully.
- If you want full real-time credit metrics you'll need vendor feeds (Bloomberg/Refinitiv/Markit) and to wire API connectors.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

# ---- Optional libraries (import defensively) ----
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

# ---- Streamlit config ----
st.set_page_config(page_title="Funding & Credit Stress Dashboard v2", layout="centered", initial_sidebar_state="expanded")

# ---- Utility functions ----
@st.cache_data(ttl=300)
def get_fred_client(api_key: str):
    if not FRED_AVAILABLE or not api_key:
        return None
    return Fred(api_key=api_key)


def parse_csv_to_series(uploaded_file) -> pd.Series:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = [c.strip().lower() for c in df.columns]
        if 'date' not in df.columns or 'value' not in df.columns:
            st.warning('Uploaded CSV must include columns "date" and "value" (case-insensitive).')
            return None
        s = pd.Series(df['value'].values, index=pd.to_datetime(df['date']), name='uploaded')
        s = s.sort_index()
        return s
    except Exception as e:
        st.warning(f'Could not parse uploaded CSV: {e}')
        return None


def business_index(days: int = 500) -> pd.DatetimeIndex:
    return pd.bdate_range(end=datetime.today(), periods=days)


def generate_sample_series(name: str, days: int = 500, level: float = 0.01, vol: float = 0.001, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = business_index(days)
    shocks = rng.normal(0, vol, size=len(dates)).cumsum()
    vals = level + shocks
    return pd.Series(vals, index=dates, name=name)


def rolling_zscore_last(series: pd.Series, window: int) -> float:
    if series.dropna().shape[0] < max(5, window//2):
        return float('nan')
    roll_mean = series.rolling(window=window).mean()
    roll_std = series.rolling(window=window).std(ddof=0)
    return ((series.iloc[-1] - roll_mean.iloc[-1]) / (roll_std.iloc[-1] if roll_std.iloc[-1] != 0 else np.nan))


def compute_mahalanobis_from_window(df_window: pd.DataFrame) -> float:
    X = df_window.dropna()
    if X.shape[0] < X.shape[1] + 2 or X.shape[1] == 0:
        return float('nan')
    mu = X.mean()
    cov = np.cov(X.T)
    cov += np.eye(cov.shape[0]) * 1e-8
    invcov = np.linalg.pinv(cov)
    last = X.iloc[-1].values - mu.values
    md = float(np.sqrt(np.dot(np.dot(last.T, invcov), last)))
    return md


def safe_plot_timeseries(df: pd.DataFrame, cols: list, title: str = None, height: int = 300):
    # Prefer Plotly, then Altair, then Streamlit native
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        for c in cols:
            if c in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[c], name=c))
        fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        return
    if ALTAIR_AVAILABLE:
        plotdf = df[cols].reset_index().melt(id_vars='index', var_name='indicator', value_name='value')
        plotdf = plotdf.rename(columns={'index': 'date'})
        chart = alt.Chart(plotdf).mark_line().encode(x='date:T', y='value:Q', color='indicator:N', tooltip=['date', 'indicator', 'value']).properties(height=height, title=title)
        st.altair_chart(chart, use_container_width=True)
        return
    # fallback
    st.write('(Plotly/Altair not available)')
    st.line_chart(df[cols])


def safe_heatmap(df: pd.DataFrame, title: str = None, height: int = 320):
    if df.empty:
        st.write('No data to display')
        return
    if PLOTLY_AVAILABLE:
        fig = px.imshow(df, aspect='auto', labels=dict(x='Date', y='Indicator', color='z-score'))
        fig.update_layout(title=title, height=height)
        st.plotly_chart(fig, use_container_width=True)
        return
    if ALTAIR_AVAILABLE:
        # Altair heatmap (long format)
        plotdf = df.reset_index().melt(id_vars='index', var_name='indicator', value_name='z')
        plotdf = plotdf.rename(columns={'index': 'date'})
        chart = alt.Chart(plotdf).mark_rect().encode(x='date:T', y='indicator:N', color='z:Q').properties(height=height, title=title)
        st.altair_chart(chart, use_container_width=True)
        return
    st.write('(Plotly/Altair not installed)')
    st.dataframe(df)

# ---- Sidebar inputs ----
st.sidebar.title('Data & Deployment')
st.sidebar.markdown('Provide optional FRED API key (if fredapi is installed) or upload CSVs for indicators.')
fred_api_key = st.sidebar.text_input('FRED API key (optional)', value=os.environ.get('FRED_API_KEY', ''))
use_sample = st.sidebar.checkbox('Use sample data when feeds are missing', value=True)

st.sidebar.markdown('---')
st.sidebar.header('Upload CSVs (optional)')
uploaded_sofr = st.sidebar.file_uploader('SOFR (date,value)', type=['csv'])
uploaded_repo = st.sidebar.file_uploader('Repo spread (date,value)', type=['csv'])
uploaded_tbill = st.sidebar.file_uploader('T-bill-OIS (date,value)', type=['csv'])
uploaded_fra = st.sidebar.file_uploader('FRA-OIS (date,value)', type=['csv'])
uploaded_cp = st.sidebar.file_uploader('CP spread (date,value)', type=['csv'])
uploaded_mmf = st.sidebar.file_uploader('MMF flows (date,value)', type=['csv'])
uploaded_ccp = st.sidebar.file_uploader('CCP margin index (date,value)', type=['csv'])

st.sidebar.markdown('---')
st.sidebar.header('Credit CSVs (optional)')
uploaded_cdx_ig = st.sidebar.file_uploader('CDX IG (date,value)', type=['csv'])
uploaded_cdx_hy = st.sidebar.file_uploader('CDX HY (date,value)', type=['csv'])
uploaded_ig_oas = st.sidebar.file_uploader('IG OAS index (date,value)', type=['csv'])
uploaded_hy_oas = st.sidebar.file_uploader('HY OAS index (date,value)', type=['csv'])
uploaded_bank_cds = st.sidebar.file_uploader('Bank CDS (date,value)', type=['csv'])

st.sidebar.markdown('---')
st.sidebar.header('Model settings')
rolling_window = st.sidebar.number_input('Rolling window (business days)', min_value=10, max_value=252, value=60, step=1)
ewma_span = st.sidebar.number_input('EWMA span (days)', min_value=5, max_value=252, value=60, step=1)
compute_pca = st.sidebar.checkbox('Compute PCA composite (requires scikit-learn)', value=True)

# ---- Data ingestion ----
fred_client = get_fred_client(fred_api_key)

# helper to load series
series = {}

# load function that prefers upload, then FRED, then sample

def load_indicator(name: str, uploaded, fred_series_id: str = None, sample_kwargs: dict = None):
    # 1) uploaded
    if uploaded is not None:
        s = parse_csv_to_series(uploaded)
        if s is not None:
            return s
    # 2) FRED
    if fred_series_id and fred_client is not None:
        try:
            fred_series = fred_client.get_series(fred_series_id)
            return pd.Series(fred_series.values, index=pd.to_datetime(fred_series.index), name=name)
        except Exception:
            pass
    # 3) sample
    if use_sample:
        kwargs = sample_kwargs or {}
        return generate_sample_series(name, **kwargs)
    return None

# indicators (fred_series_id are optional and may not correspond to desired exact series)
series['sofr'] = load_indicator('sofr', uploaded_sofr, fred_series_id='SOFR', sample_kwargs={'level':0.02,'vol':0.0006})
series['repo_spread'] = load_indicator('repo_spread', uploaded_repo, fred_series_id=None, sample_kwargs={'level':0.002,'vol':0.0005})
series['tbill_oisspread'] = load_indicator('tbill_oisspread', uploaded_tbill, fred_series_id=None, sample_kwargs={'level':0.0,'vol':0.0004})
series['fra_ois'] = load_indicator('fra_ois', uploaded_fra, fred_series_id=None, sample_kwargs={'level':0.0005,'vol':0.0007})
series['cp_spread'] = load_indicator('cp_spread', uploaded_cp, fred_series_id=None, sample_kwargs={'level':0.001,'vol':0.0012})
series['mmf_flows'] = load_indicator('mmf_flows', uploaded_mmf, fred_series_id=None, sample_kwargs={'level':0.0,'vol':1e6})
series['ccp_margin'] = load_indicator('ccp_margin', uploaded_ccp, fred_series_id=None, sample_kwargs={'level':1.0,'vol':0.03})

# credit
series['cdx_ig'] = load_indicator('cdx_ig', uploaded_cdx_ig, fred_series_id=None, sample_kwargs={'level':70,'vol':5})
series['cdx_hy'] = load_indicator('cdx_hy', uploaded_cdx_hy, fred_series_id=None, sample_kwargs={'level':400,'vol':20})
series['ig_oas'] = load_indicator('ig_oas', uploaded_ig_oas, fred_series_id=None, sample_kwargs={'level':80,'vol':8})
series['hy_oas'] = load_indicator('hy_oas', uploaded_hy_oas, fred_series_id=None, sample_kwargs={'level':400,'vol':25})
series['bank_cds'] = load_indicator('bank_cds', uploaded_bank_cds, fred_series_id=None, sample_kwargs={'level':50,'vol':6})

# Align and tidy
common_idx = business_index(500)
for k in list(series.keys()):
    if series[k] is None:
        series.pop(k, None)
    else:
        series[k] = series[k].reindex(common_idx).interpolate().fillna(method='bfill').fillna(method='ffill')

if not series:
    st.error('No indicators available. Upload CSVs or enable sample data.')
    st.stop()

# DataFrame
df = pd.DataFrame(series)

# ---- Modeling ----
# compute rolling z-scores per series
z_scores = pd.DataFrame(index=df.index)
for col in df.columns:
    z_scores[col] = df[col].rolling(window=rolling_window, min_periods=max(5, rolling_window//2)).apply(
        lambda x: (x[-1] - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) != 0 else np.nan)
    )

# invert mmf_flows so that large negative flows -> positive stress (if present)
if 'mmf_flows' in z_scores.columns:
    z_scores['mmf_flows'] = -z_scores['mmf_flows']

# EWMA volatility
ewma_vol = df.ewm(span=ewma_span, adjust=False).std()

# PCA composite (if sklearn available and user requested)
pca_series = pd.Series(index=df.index, data=np.nan, name='pc1')
if compute_pca and SKLEARN_AVAILABLE:
    valid_cols_for_pca = [c for c in z_scores.columns if z_scores[c].notna().sum() > rolling_window]
    if len(valid_cols_for_pca) >= 2:
        pca_input = z_scores[valid_cols_for_pca].dropna()
        try:
            pca = PCA(n_components=1)
            comp = pca.fit_transform(pca_input.fillna(0))
            pca_series.loc[pca_input.index] = comp[:, 0]
        except Exception:
            pca_series[:] = np.nan

# Mahalanobis over recent window
mahalanobis_value = compute_mahalanobis_from_window(z_scores.dropna(axis=1, how='all').tail(rolling_window))

# Composite simple: mean absolute z across available indicators (latest)
available_cols = [c for c in z_scores.columns if z_scores[c].notna().sum() > 0]
if available_cols:
    latest_absz = z_scores[available_cols].abs().iloc[-1].fillna(0)
    composite_simple = latest_absz.mean()
else:
    composite_simple = 0.0

# Scale heuristics -> 0-100
composite_scaled = int(np.clip((composite_simple / 3.0) * 100, 0, 100))
pc1_nonempty = pca_series.dropna()
pc1_latest_val = float(pc1_nonempty.iloc[-1]) if not pc1_nonempty.empty else 0.0
if not pc1_nonempty.empty and np.std(pc1_nonempty) != 0:
    pc1_z = (pc1_latest_val - np.mean(pc1_nonempty)) / np.std(pc1_nonempty)
    pc1_scaled = int(np.clip(pc1_z * 20 + 50, 0, 100))
else:
    pc1_scaled = 50

maha_score = int(np.clip((mahalanobis_value / 3.0) * 100, 0, 100)) if not np.isnan(mahalanobis_value) else 0

final_score = int(np.clip(0.5 * composite_scaled + 0.4 * pc1_scaled + 0.1 * maha_score, 0, 100))

# ---- UI / layout ----
st.title('Funding & Credit Market Stress Dashboard (v2)')
st.markdown('Mobile-friendly. Deploy to Streamlit Cloud for phone access.')

# Top metrics
cols = st.columns(4)
cols[0].metric('Composite (simple)', f'{composite_scaled}')
cols[1].metric('PCA-based', f'{pc1_scaled}')
cols[2].metric('Mahalanobis', f'{maha_score}')
cols[3].metric('Final stress 0-100', f'{final_score}')

st.markdown('---')

# Time series
with st.expander('Money-market time series', expanded=True):
    safe_plot_timeseries(df, [c for c in ['sofr', 'repo_spread', 'tbill_oisspread', 'fra_ois', 'cp_spread'] if c in df.columns], title='Money-market indicators')

with st.expander('Credit-market time series', expanded=False):
    safe_plot_timeseries(df, [c for c in ['cdx_ig', 'cdx_hy', 'ig_oas', 'hy_oas', 'bank_cds'] if c in df.columns], title='Credit indicators')

with st.expander('Z-score heatmap (recent)', expanded=False):
    heat_df = z_scores[available_cols].tail(120).T if available_cols else pd.DataFrame()
    safe_heatmap(heat_df, title='Z-score heatmap')

st.markdown('---')

# Alerts
st.header('Alerts & signals')
alerts = []
if final_score >= 80:
    alerts.append(f'Final stress elevated: {final_score}/100')
if maha_score >= 70:
    alerts.append(f'Mahalanobis anomaly high: {maha_score}')
for c in available_cols:
    vz = z_scores[c].iloc[-1]
    if not np.isnan(vz) and abs(vz) >= 2.0:
        alerts.append(f'{c} z-score large: {vz:.2f}')

if alerts:
    for a in alerts:
        st.error(a)
else:
    st.success('No immediate high-stress alerts')

st.markdown('---')

# Exports
st.header('Export')
if st.button('Download latest raw indicators CSV'):
    csv = df.reset_index().rename(columns={'index': 'date'}).to_csv(index=False).encode('utf-8')
    st.download_button(label='Download CSV', data=csv, file_name='indicators_latest.csv', mime='text/csv')

if st.button('Download z-scores CSV'):
    csv2 = z_scores.reset_index().rename(columns={'index': 'date'}).to_csv(index=False).encode('utf-8')
    st.download_button(label='Download z-scores', data=csv2, file_name='z_scores.csv', mime='text/csv')

st.markdown('---')

st.write('Deployment tips: push this file to a GitHub repo and use Streamlit Cloud for easy mobile access. Add a requirements.txt with the packages you need (plotly/altair/fredapi/scikit-learn) so the cloud environment installs them.')

# End of app
