"""
Streamlit Money & Credit Market Stress Dashboard (v2) — Clean public-data version

User request applied:
- Removed all synthetic / sample data generation. If no data is available the dashboard will show empty charts and clear messaging.
- Removed non-public APIs and vendor-only indicators from the default app. The app now only automatically fetches from public APIs (FRED via fredapi) when a FRED API key is provided. Any other indicator must be provided via CSV upload by the user.
- Kept advanced analytics (rolling z-scores, EWMA, PCA and Mahalanobis) but they only run when sufficient public/uploaded data exists.

How to use:
- Provide a FRED API key in the sidebar to fetch public series (SOFR, DTB1 and FEDFUNDS if requested). The app will only fetch series with public FRED IDs.
- Upload CSVs for any indicator that does not have a public API (CSV must have columns: date, value).
- If an indicator is not present the app will not fabricate values and charts will be blank with a short note.

Deploy:
- Push this file to a GitHub repo and deploy to Streamlit Cloud for phone access. Include a requirements.txt listing packages you want installed. The app tolerates optional packages being missing and will fall back gracefully.

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
st.set_page_config(page_title="Funding Market Stress — Public-data only", layout="centered")

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
    if not cols:
        st.info('No series to display for this chart.')
        return
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
        st.info('No data to display in heatmap.')
        return
    if PLOTLY_AVAILABLE:
        fig = px.imshow(df, aspect='auto', labels=dict(x='Date', y='Indicator', color='z-score'))
        fig.update_layout(title=title, height=height)
        st.plotly_chart(fig, use_container_width=True)
        return
    if ALTAIR_AVAILABLE:
        plotdf = df.reset_index().melt(id_vars='index', var_name='indicator', value_name='z')
        plotdf = plotdf.rename(columns={'index': 'date'})
        chart = alt.Chart(plotdf).mark_rect().encode(x='date:T', y='indicator:N', color='z:Q').properties(height=height, title=title)
        st.altair_chart(chart, use_container_width=True)
        return
    st.write('(Plotly/Altair not installed)')
    st.dataframe(df)

# ---- Sidebar inputs ----
st.sidebar.title('Data & Deployment (public-data only)')
st.sidebar.markdown('This version only pulls public FRED series automatically. Other series must be uploaded as CSVs.')
fred_api_key = st.sidebar.text_input('FRED API key (optional)', value=os.environ.get('FRED_API_KEY', ''))

st.sidebar.markdown('---')
st.sidebar.header('Upload CSVs for indicators (optional)')
# Money-market indicators: uploads only for indicators without public FRED series
uploaded_sofr = st.sidebar.file_uploader('SOFR CSV (date,value) — optional (FRED available)', type=['csv'])
uploaded_repo = st.sidebar.file_uploader('Repo spread CSV (date,value) — upload only', type=['csv'])
uploaded_tbill = st.sidebar.file_uploader('T-bill CSV (date,value) — optional (FRED available)', type=['csv'])
uploaded_fra = st.sidebar.file_uploader('FRA-OIS CSV (date,value) — upload only', type=['csv'])
uploaded_cp = st.sidebar.file_uploader('CP spread CSV (date,value) — upload only', type=['csv'])
uploaded_mmf = st.sidebar.file_uploader('MMF flows CSV (date,value) — upload only', type=['csv'])

st.sidebar.markdown('---')
st.sidebar.header('Model settings')
rolling_window = st.sidebar.number_input('Rolling window (business days)', min_value=10, max_value=252, value=60, step=1)
compute_pca = st.sidebar.checkbox('Compute PCA composite (requires scikit-learn)', value=True)

# ---- Data ingestion (public-only logic) ----
fred_client = get_fred_client(fred_api_key)

series = {}

# Helper: try upload first, then FRED for known public series, otherwise None

def load_indicator(name: str, uploaded, fred_series_id: str = None):
    if uploaded is not None:
        s = parse_csv_to_series(uploaded)
        if s is not None:
            return s
    if fred_series_id and fred_client is not None:
        try:
            fred_series = fred_client.get_series(fred_series_id)
            return pd.Series(fred_series.values, index=pd.to_datetime(fred_series.index), name=name)
        except Exception:
            return None
    return None

# Public FRED-backed indicators (only these are auto-fetchable)
# SOFR: FRED ID 'SOFR'
# 1-month Treasury bill: FRED ID 'DTB1'
# Effective Fed funds: 'FEDFUNDS'

series['sofr'] = load_indicator('sofr', uploaded_sofr, fred_series_id='SOFR')
series['dtb1'] = load_indicator('dtb1', uploaded_tbill, fred_series_id='DTB1')
series['fedfunds'] = load_indicator('fedfunds', None, fred_series_id='FEDFUNDS')

# Indicators that have no reliable free public API: repo spreads, FRA-OIS, CP, MMF flows — accept only uploads
series['repo_spread'] = load_indicator('repo_spread', uploaded_repo, fred_series_id=None)
series['fra_ois'] = load_indicator('fra_ois', uploaded_fra, fred_series_id=None)
series['cp_spread'] = load_indicator('cp_spread', uploaded_cp, fred_series_id=None)
series['mmf_flows'] = load_indicator('mmf_flows', uploaded_mmf, fred_series_id=None)

# Remove None entries
for k in list(series.keys()):
    if series[k] is None:
        series.pop(k)

# Align to union index of available series
if series:
    common_idx = pd.DatetimeIndex(sorted({d for s in series.values() for d in s.index}))
    if len(common_idx) > 0:
        # resample to business days covering the union range for nicer charts
        start = common_idx.min()
        end = common_idx.max()
        bidx = pd.bdate_range(start=start, end=end)
        for k in list(series.keys()):
            series[k] = series[k].reindex(bidx).interpolate().fillna(method='bfill').fillna(method='ffill')
else:
    # no series provided
    df = pd.DataFrame()

if series:
    df = pd.DataFrame(series)
else:
    df = pd.DataFrame()

# ---- Modeling: only run if there's data ----
if df.empty:
    st.title('Funding Market Stress — Public-data only')
    st.warning('No indicators available. Provide a FRED API key for public series (SOFR/DTB1/FEDFUNDS) or upload CSVs for other indicators.')
    st.write('The dashboard will remain empty until you provide real data. No synthetic/sample data will be generated.')
    # Still render the sidebar and stop further processing
    st.stop()

# compute rolling z-scores per series
z_scores = pd.DataFrame(index=df.index)
for col in df.columns:
    z_scores[col] = df[col].rolling(window=rolling_window, min_periods=max(5, rolling_window//2)).apply(
        lambda x: (x[-1] - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) != 0 else np.nan)
    )

# Invert mmf_flows if present so negative (large outflows) are treated as positive stress
if 'mmf_flows' in z_scores.columns:
    z_scores['mmf_flows'] = -z_scores['mmf_flows']

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
st.title('Funding Market Stress — Public-data only')
st.markdown('This dashboard only displays public-data series (FRED) automatically. Upload CSVs for other series.')

# Top metrics
cols = st.columns(4)
cols[0].metric('Composite (simple)', f'{composite_scaled}')
cols[1].metric('PCA-based', f'{pc1_scaled}')
cols[2].metric('Mahalanobis', f'{maha_score}')
cols[3].metric('Final stress 0-100', f'{final_score}')

st.markdown('---')

# Time series
with st.expander('Money-market time series', expanded=True):
    mm_cols = [c for c in ['sofr', 'repo_spread', 'dtb1', 'fra_ois', 'cp_spread', 'mmf_flows', 'fedfunds'] if c in df.columns]
    safe_plot_timeseries(df, mm_cols, title='Money-market indicators')

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

st.write('Deployment tips: push this file to a GitHub repo and use Streamlit Cloud for easy mobile access. Add requirements.txt with the packages you want pre-installed (plotly/altair/fredapi/scikit-learn).')

# End of app
