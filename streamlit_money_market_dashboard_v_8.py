"""
streamlit_money_market_dashboard_v8.py

Professional Money Market Stress Dashboard (v8)
- Sources: FRED (fredapi) + OFR Short-Term Funding Monitor API (JSON)
- Focus: anomaly-first visualization, robust checks, professional scoring and verification
- Features:
    * Explicit source/status reporting and sample previews
    * Automated series validation (recency, length, frequency)
    * Engineered anomaly series (SOFR-FedFunds, Repo spread, FRA-OIS, TED, HY-IG, CP spreads, MMF flows, RRP)
    * Multiple stress metrics: rolling z, EWMA z, PCA composite, Mahalanobis anomaly
    * Configurable weights and smoothing in sidebar
    * Mobile-friendly cards, clarified labels, export

Run:
    pip install -r requirements.txt
    streamlit run streamlit_money_market_dashboard_v8.py

Notes:
- Provide a FRED API key in the sidebar or via st.secrets['FRED_API_KEY'] / env FRED_API_KEY.
- OFR API is used where available; if unavailable, upload CSV fallbacks are provided.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from datetime import datetime, timedelta
import os
from typing import Dict, Optional, List, Tuple

# Optional libs
PLOTLY = False
FREDAPI = False
SKLEARN = False
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY = True
except Exception:
    PLOTLY = False

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

st.set_page_config(page_title="Money Market Stress Dashboard v8", layout="wide")

# --------------------------- Utilities & I/O ---------------------------
@st.cache_data(ttl=21600)
def ofr_fetch_dataset(dataset_key: str) -> Optional[dict]:
    base = "https://data.financialresearch.gov/v1/series/dataset"
    try:
        r = requests.get(base, params={"dataset": dataset_key}, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

@st.cache_data(ttl=21600)
def fred_get_series(series_id: str, api_key: str) -> Optional[pd.Series]:
    if not (FREDAPI and api_key):
        return None
    try:
        fred = Fred(api_key=api_key)
        raw = fred.get_series(series_id)
        s = pd.Series(raw.values, index=pd.to_datetime(raw.index), name=series_id)
        return _clean_ts(s, series_id)
    except Exception:
        return None

def _clean_ts(s: pd.Series, name: str) -> Optional[pd.Series]:
    if s is None:
        return None
    try:
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index, errors='coerce')
        s = s[~s.index.isna()].copy()
        s = s.sort_index()
        s = s[~s.index.duplicated(keep='last')]
        s = pd.to_numeric(s, errors='coerce')
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        s.name = name
        return s
    except Exception:
        return None

@st.cache_data(ttl=21600)
def parse_ofr_series_json(json_blob: dict) -> Dict[str, pd.Series]:
    out = {}
    if not json_blob:
        return out
    timeseries = json_blob.get('timeseries') or json_blob.get('data') or {}
    for mnemonic, details in timeseries.items():
        try:
            # find aggregation list
            ts = details.get('timeseries') if isinstance(details, dict) else None
            if isinstance(ts, dict) and 'aggregation' in ts:
                agg = ts['aggregation']
            else:
                # deeper search
                agg = None
                if isinstance(details, dict):
                    for v in details.values():
                        if isinstance(v, dict) and 'aggregation' in v:
                            agg = v['aggregation']; break
            if not agg:
                continue
            dates = [pd.to_datetime(x[0]) for x in agg if x and x[0] is not None]
            vals = [x[1] for x in agg if x and len(x) > 1]
            s = pd.Series(vals, index=pd.to_datetime(dates), name=mnemonic)
            s = _clean_ts(s, mnemonic)
            if s is not None and not s.empty:
                out[mnemonic] = s
        except Exception:
            continue
    return out

def safe_union_index(series_map: Dict[str, pd.Series]) -> pd.DatetimeIndex:
    valid = [s for s in series_map.values() if s is not None and not s.empty]
    if not valid:
        return pd.DatetimeIndex([])
    all_dates = set()
    for s in valid:
        all_dates.update(pd.to_datetime(s.index).tolist())
    if not all_dates:
        return pd.DatetimeIndex([])
    all_dates = pd.DatetimeIndex(sorted(all_dates))
    return pd.bdate_range(start=all_dates.min(), end=all_dates.max())

# --------------------------- Validation helpers ---------------------------

def validate_series(s: pd.Series) -> Dict[str, object]:
    """Return basic validation info: rows, first/last date, days since last, frequency estimate."""
    if s is None or s.empty:
        return {'ok': False, 'n': 0}
    first, last = s.index.min(), s.index.max()
    days_since = (pd.Timestamp.now(tz=None) - last).days
    # frequency: average delta in days
    deltas = s.index.to_series().diff().dropna().dt.days
    freq = deltas.median() if not deltas.empty else None
    return {'ok': True, 'n': len(s), 'first': first.date(), 'last': last.date(), 'days_since': int(days_since), 'freq_days': int(freq) if freq is not None else None}

# --------------------------- Scoring & modeling ---------------------------

def rolling_z(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=max(5, int(window/3))).apply(lambda x: (x[-1] - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) != 0 else np.nan))

def ewma_z(series: pd.Series, span: int) -> pd.Series:
    # EWMA mean & std -> compute z of latest value
    mu = series.ewm(span=span, adjust=False).mean()
    sigma = series.ewm(span=span, adjust=False).std(bias=False)
    z = (series - mu) / sigma
    return z

def mahala_score(Z: pd.DataFrame) -> float:
    # Mahalanobis distance of last observation across columns Z (z-scores) -> scaled 0-100
    try:
        X = Z.dropna()
        if X.shape[0] < 5 or X.shape[1] < 2:
            return 0.0
        last = X.iloc[-1].values
        mu = X.mean().values
        cov = np.cov(X.T)
        cov += np.eye(cov.shape[0]) * 1e-6
        inv = np.linalg.pinv(cov)
        md = float(np.sqrt((last - mu).T @ inv @ (last - mu)))
        return float(np.clip((md / 3.0) * 100, 0, 100))
    except Exception:
        return 0.0

# --------------------------- UI: Sidebar config ---------------------------
st.sidebar.title('Dashboard settings (v8)')
# FRED key resolution
fred_key = None
if hasattr(st, 'secrets') and isinstance(st.secrets, dict) and st.secrets.get('FRED_API_KEY'):
    fred_key = st.secrets.get('FRED_API_KEY')
fred_key = fred_key or os.environ.get('FRED_API_KEY')
fred_key = st.sidebar.text_input('FRED API key (optional)', value=fred_key or '')

horizon = st.sidebar.selectbox('Horizon', ['6M','1Y','2Y','5Y'], index=1)
h_map = {'6M':182,'1Y':365,'2Y':730,'5Y':1825}
h_days = h_map[horizon]
roll_window = st.sidebar.slider('Rolling z-window (days)', min_value=20, max_value=180, value=60)
ewma_span = st.sidebar.slider('EWMA span (days)', min_value=10, max_value=120, value=30)
weights = st.sidebar.slider('Combined weights: Funding / Credit / Liquidity (sum to 100) - Funding', min_value=0, max_value=100, value=40)
# compute remaining weights
credit_w = st.sidebar.slider('Credit weight', min_value=0, max_value=100, value=40)
liq_w = max(0, 100 - weights - credit_w)
st.sidebar.markdown(f'Liquidity weight auto = **{liq_w}** (keeps sum=100)')
compute_pca = st.sidebar.checkbox('Compute PCA composite (requires scikit-learn)', value=True)
show_raw = st.sidebar.checkbox('Show raw levels (separate charts)', value=False)

# --------------------------- Data sources & mapping ---------------------------
FRED_MAP = {
    'sofr':'SOFR', 'fedfunds':'FEDFUNDS','dtb3':'DTB3','dtb1':'DTB1','dtb6':'DTB6','dtb12':'DTB12',
    'ig_oas':'BAMLC0A0CM','hy_oas':'BAMLH0A0HYM2','totalcp':'TOTALCP','cpn3m':'CPN3M','cpf3m':'CPF3M','rrp':'RRPONTSYD',
    'tedrate':'TEDRATE'
}

# allow manual CSV uploads as fallback for repo/mmf/cp
st.sidebar.markdown('CSV fallbacks (date,value)')
up_repo = st.sidebar.file_uploader('Upload repo rates CSV', type=['csv','xls','xlsx'])
up_mmf = st.sidebar.file_uploader('Upload MMF flows CSV', type=['csv','xls','xlsx'])
up_cp = st.sidebar.file_uploader('Upload CP CSV', type=['csv','xls','xlsx'])

# --------------------------- Fetch & validate series ---------------------------
st.header('Data loading & validation (v8)')
status = {}
series: Dict[str,pd.Series] = {}

# 1) FRED
if fred_key and FREDAPI:
    st.sidebar.success('FRED: enabled')
    for k,sid in FRED_MAP.items():
        s = fred_get_series(sid, fred_key)
        info = validate_series(s)
        series[k] = s
        status[k] = {'source':'FRED','sid':sid,'ok':info.get('ok',False),'n':info.get('n',0),'last':info.get('last',None),'days_since':info.get('days_since',None)}
else:
    for k in FRED_MAP:
        status[k] = {'source':'FRED','sid':FRED_MAP[k],'ok':False,'n':0}
    if not FREDAPI:
        st.sidebar.warning('fredapi not installed — FRED pulls disabled')

# 2) OFR datasets
ofr_mmf = ofr_fetch_dataset('mmf')
ofr_repo = ofr_fetch_dataset('repo')
if ofr_mmf:
    ofr_map = parse_ofr_series_json(ofr_mmf)
    # keep names prefixed
    for k,v in ofr_map.items():
        series[f'ofr_mmf_{k}'] = v
    status['ofr_mmf'] = {'source':'OFR','ok':True,'n':len(ofr_map)}
else:
    status['ofr_mmf'] = {'source':'OFR','ok':False,'n':0}

if ofr_repo:
    repo_map = parse_ofr_series_json(ofr_repo)
    for k,v in repo_map.items():
        series[f'ofr_repo_{k}'] = v
    status['ofr_repo'] = {'source':'OFR','ok':True,'n':len(repo_map)}
else:
    status['ofr_repo'] = {'source':'OFR','ok':False,'n':0}

# 3) CSV fallbacks
def read_upload(u, name):
    try:
        if u is None:
            return None
        if str(u.name).lower().endswith(('.xls','.xlsx')):
            df = pd.read_excel(u)
        else:
            df = pd.read_csv(u)
        df.columns = [c.strip().lower() for c in df.columns]
        if 'date' in df.columns and 'value' in df.columns:
            s = pd.Series(df['value'].values, index=pd.to_datetime(df['date'], errors='coerce'), name=name)
            return _clean_ts(s, name)
    except Exception:
        return None

repo_up = read_upload(up_repo,'repo_rate_upload')
if repo_up is not None:
    series['repo_rate_upload'] = repo_up
    status['repo_rate_upload'] = {'source':'upload','ok':True,'n':len(repo_up)}

mmf_up = read_upload(up_mmf,'mmf_flows_upload')
if mmf_up is not None:
    series['mmf_flows_upload'] = mmf_up
    status['mmf_flows_upload'] = {'source':'upload','ok':True,'n':len(mmf_up)}

cp_up = read_upload(up_cp,'cp_upload')
if cp_up is not None:
    series['cp_upload'] = cp_up
    status['cp_upload'] = {'source':'upload','ok':True,'n':len(cp_up)}

# --------------------------- Compose DataFrame ---------------------------
# drop None series
series = {k:v for k,v in series.items() if v is not None and not v.empty}
if not series:
    st.error('No data available. Provide FRED key or upload fallback CSVs.')
    st.stop()

# union index and reindex
bidx = safe_union_index(series)
for k in list(series.keys()):
    series[k] = series[k].reindex(bidx)

df = pd.DataFrame(series)

# --------------------------- Construct engineered anomalies ---------------------------
anoms = pd.DataFrame(index=df.index)

# SOFR - FedFunds (bps)
if 'sofr' in df.columns and 'fedfunds' in df.columns:
    anoms['sofr_minus_fedfunds_bps'] = (df['sofr'] - df['fedfunds']) * 10000

# Repo spread: prefer OFR repo series or uploaded repo
repo_candidates = [c for c in df.columns if c.startswith('ofr_repo_') or c=='repo_rate_upload']
if repo_candidates:
    anoms['repo_rate'] = df[repo_candidates[0]]
    if 'fedfunds' in df.columns:
        anoms['repo_minus_fedfunds_bps'] = (anoms['repo_rate'] - df['fedfunds']) * 10000

# TED, FRA-OIS (if present)
if 'tedrate' in df.columns:
    anoms['ted_bps'] = df['tedrate'] * 100
if 'fra_ois' in df.columns:
    anoms['fra_ois_bps'] = df['fra_ois'] * 10000

# Credit spreads
if 'ig_oas' in df.columns:
    anoms['ig_oas_bps'] = df['ig_oas']
if 'hy_oas' in df.columns:
    anoms['hy_oas_bps'] = df['hy_oas']
if 'hy_oas_bps' in anoms.columns and 'ig_oas_bps' in anoms.columns:
    anoms['hy_minus_ig_bps'] = anoms['hy_oas_bps'] - anoms['ig_oas_bps']

# CP & MMF
if 'totalcp' in df.columns:
    anoms['totalcp'] = df['totalcp']
if 'cpn3m' in df.columns:
    anoms['cpn3m_bps'] = df['cpn3m'] * 100
# OFR MMF keys
mmf_cols = [c for c in df.columns if c.startswith('ofr_mmf_')]
if mmf_cols:
    # pick the most relevant named one if exists, else keep all
    for c in mmf_cols:
        anoms[c] = df[c]
elif 'mmf_flows_upload' in df.columns:
    anoms['mmf_flows_upload'] = df['mmf_flows_upload']

# RRP outstanding
if 'rrp' in df.columns:
    anoms['rrp_outstanding'] = df['rrp']

# drop empty
anoms = anoms.dropna(axis=1, how='all')
if anoms.empty:
    st.error('After engineering, no anomaly series available. Check data sources.')
    st.stop()

# --------------------------- Verification & sample preview ---------------------------
st.subheader('Source & sample preview')
st.write('Loaded series status (counts and last date)')
status_df = pd.DataFrame(status).T.fillna('').astype(str)
st.table(status_df)

with st.expander('Show sample head/tail of loaded series'):
    for k in df.columns:
        st.write(f'### {k}')
        st.write('First 3 rows:')
        st.write(df[k].dropna().head(3).to_frame())
        st.write('Last 3 rows:')
        st.write(df[k].dropna().tail(3).to_frame())

# --------------------------- Compute z-scores, EWMA, PCA, Mahalanobis ---------------------------
z = pd.DataFrame(index=anoms.index)
ewma = pd.DataFrame(index=anoms.index)
for col in anoms.columns:
    z[col] = rolling_z(anoms[col], window=roll_window)
    ew = ewma_z(anoms[col].fillna(method='ffill').fillna(method='bfill'), span=ewma_span)
    ewma[col] = ew

# PCA composite if requested
pc1 = pd.Series(index=anoms.index, data=np.nan)
if compute_pca and SKLEARN:
    valid_cols = [c for c in z.columns if z[c].notna().sum() > roll_window]
    if len(valid_cols) >= 2:
        try:
            xin = z[valid_cols].dropna()
            pca = PCA(n_components=1)
            comp = pca.fit_transform(xin.fillna(0))
            pc1.loc[xin.index] = comp[:,0]
        except Exception:
            pc1[:] = np.nan

# Mahalanobis on z matrix
maha = mahala_score(z)

# --------------------------- Scoring (category level + combined) ---------------------------
FUNDING_ANOMS = [c for c in anoms.columns if 'sofr' in c or 'repo' in c or 'fra' in c or 'ted' in c]
CREDIT_ANOMS = [c for c in anoms.columns if 'ig_oas' in c or 'hy_oas' in c or 'hy_minus_ig' in c]
LIQUIDITY_ANOMS = [c for c in anoms.columns if 'mmf' in c or 'totalcp' in c or 'cpn3m' in c or 'rrp' in c]

def mean_abs_latest(df_z: pd.DataFrame, cols: List[str]) -> float:
    cols = [c for c in cols if c in df_z.columns]
    if not cols:
        return np.nan
    vals = df_z[cols].iloc[-1].abs().dropna()
    return float(vals.mean()) if not vals.empty else np.nan

fund_z = mean_abs_latest(z, FUNDING_ANOMS)
cred_z = mean_abs_latest(z, CREDIT_ANOMS)
liq_z = mean_abs_latest(z, LIQUIDITY_ANOMS)

def z_to_0_100(zv: float) -> int:
    if pd.isna(zv):
        return 0
    return int(np.clip((zv / 3.0) * 100, 0, 100))

fund_score = z_to_0_100(fund_z)
cred_score = z_to_0_100(cred_z)
liq_score = z_to_0_100(liq_z)
combined_score = int(np.clip((weights*z_to_0_100(fund_z) + credit_w*z_to_0_100(cred_z) + liq_w*z_to_0_100(liq_z))/100.0, 0, 100))

# incorporate PCA & Mahalanobis into diagnostics
pc1_val = float(pc1.dropna().iloc[-1]) if not pc1.dropna().empty else np.nan

# --------------------------- UI: top meters & deltas ---------------------------
st.subheader('Stress meters (at a glance)')
# compute 1m change (approx 21 business days)
def delta_1m(df_z: pd.DataFrame, cols: List[str]) -> Optional[float]:
    cols = [c for c in cols if c in df_z.columns]
    if not cols:
        return None
    idx = df_z.index
    if len(idx) < 22:
        return None
    now = df_z[cols].iloc[-1].abs().dropna().mean()
    past_idx = max(0, len(idx)-22)
    past = df_z[cols].iloc[past_idx].abs().dropna().mean()
    if pd.isna(now) or pd.isna(past):
        return None
    return now - past

fund_delta = delta_1m(z, FUNDING_ANOMS)
cred_delta = delta_1m(z, CREDIT_ANOMS)
liq_delta = delta_1m(z, LIQUIDITY_ANOMS)

# responsive layout: try 4 columns, stack on narrow screens
c1,c2,c3,c4 = st.columns(4)

def color_for_score(s:int)->str:
    if s<=40: return '#0f9d58'
    if s<=70: return '#f4b400'
    return '#db4437'

def show_card(col, title, value, delta=None):
    if PLOTLY:
        fig = go.Figure(go.Indicator(mode='gauge+number+delta', value=value,
                                     title={'text':title}, delta={'reference': max(0, value-(delta or 0)) if delta is not None else None},
                                     gauge={'axis':{'range':[0,100]}, 'bar':{'color':color_for_score(value)}}))
        fig.update_layout(height=220, margin=dict(l=2,r=2,t=30,b=2))
        col.plotly_chart(fig, use_container_width=True)
    else:
        col.markdown(f'**{title}**')
        col.metric('', f'{value}', delta=f'{delta:+.2f}' if delta is not None else '')
        col.markdown(f"<div style='height:8px;background:{color_for_score(value)};border-radius:6px'></div>", unsafe_allow_html=True)

show_card(c1, 'Funding Stress (0-100)', fund_score, fund_delta)
show_card(c2, 'Credit Stress (0-100)', cred_score, cred_delta)
show_card(c3, 'Liquidity Stress (0-100)', liq_score, liq_delta)
show_card(c4, 'Combined Stress (0-100)', combined_score, None)

st.markdown('---')

# --------------------------- Charts: anomalies (single-scale per chart) ---------------------------
st.subheader('Anomaly charts — recent view')
end = anoms.index.max()
start = end - pd.Timedelta(days=h_days)
anoms_recent = anoms.loc[start:end]
z_recent = z.loc[start:end]

# helper plotting
def plot_series(df_plot: pd.DataFrame, title: str, height:int=300, ytitle:str=''):
    if df_plot.empty:
        st.info('No data for chart')
        return
    if PLOTLY:
        fig = go.Figure()
        for col in df_plot.columns:
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[col], name=col, mode='lines'))
        fig.update_layout(title=title, height=height, margin=dict(l=10,r=10,t=30,b=10), yaxis_title=ytitle)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(df_plot)

# Funding anomalies (z-scores)
fund_z_cols = [c for c in FUNDING_ANOMS if c in z_recent.columns]
if fund_z_cols:
    st.subheader('Funding anomalies (z-scores)')
    plot_series(z_recent[fund_z_cols].rename(columns=lambda c: c.replace('_',' ')), 'Funding anomalies (z)')
    if show_raw:
        plot_series(anoms_recent[[c for c in FUNDING_ANOMS if c in anoms_recent.columns]].rename(columns=lambda c: c.replace('_bps',' (bps)').replace('_',' ')), 'Funding anomalies (raw)')

# Credit anomalies
cred_z_cols = [c for c in CREDIT_ANOMS if c in z_recent.columns]
if cred_z_cols:
    st.subheader('Credit anomalies (z-scores)')
    plot_series(z_recent[cred_z_cols].rename(columns=lambda c: c.replace('_',' ')), 'Credit anomalies (z)')
    if show_raw:
        plot_series(anoms_recent[[c for c in CREDIT_ANOMS if c in anoms_recent.columns]].rename(columns=lambda c: c.replace('_bps',' (bps)').replace('_',' ')), 'Credit anomalies (raw)')

# Liquidity anomalies
liq_z_cols = [c for c in LIQUIDITY_ANOMS if c in z_recent.columns]
if liq_z_cols:
    st.subheader('Liquidity anomalies (z-scores)')
    plot_series(z_recent[liq_z_cols].rename(columns=lambda c: c.replace('_',' ')), 'Liquidity anomalies (z)')
    if show_raw:
        plot_series(anoms_recent[[c for c in LIQUIDITY_ANOMS if c in anoms_recent.columns]].rename(columns=lambda c: c.replace('_bps',' (bps)').replace('_',' ')), 'Liquidity anomalies (raw)')

# Heatmap of z
st.markdown('---')
st.subheader('Heatmap — recent z-scores (last ~120 bdays)')
heat_n = min(len(z), 120)
if heat_n >= 10:
    z_h = z.tail(heat_n)
    rows = [c for c in z_h.columns]
    heat_df = z_h[rows].T
    heat_df.index = [c.replace('_',' ') for c in heat_df.index]
    if PLOTLY:
        fig = px.imshow(heat_df, aspect='auto', labels=dict(x='Date', y='Indicator', color='z'))
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(heat_df)
else:
    st.info('Not enough history for heatmap')

# --------------------------- Alerts & Diagnostics ---------------------------
st.markdown('---')
st.subheader('Alerts — latest breaches')
alerts = []
if not z.empty:
    latest = z.iloc[-1]
    for c in latest.index:
        v = latest[c]
        if pd.notna(v) and abs(v) >= 2.0:
            cat = 'Funding' if c in FUNDING_ANOMS else 'Credit' if c in CREDIT_ANOMS else 'Liquidity'
            alerts.append((c,v,cat))
if alerts:
    for c,v,cat in alerts:
        st.error(f'⚠️ {c.replace("_"," ")} — {cat} — z = {v:.2f}')
else:
    st.success('No recent |z| >= 2 breaches')

# show diagnostics
with st.expander('Model diagnostics & advanced scores'):
    st.write({'pc1_latest': float(pc1.dropna().iloc[-1]) if not pc1.dropna().empty else None, 'mahala_score': maha})
    st.write('Category z (mean abs):', {'fund_z':fund_z,'cred_z':cred_z,'liq_z':liq_z})

# --------------------------- Export ---------------------------
st.markdown('---')
st.subheader('Export')
if st.button('Download anomalies CSV'):
    tmp = anoms.reset_index().rename(columns={'index':'date'})
    st.download_button('Download anomalies', tmp.to_csv(index=False).encode('utf-8'), file_name='anomalies_v8.csv', mime='text/csv')
if st.button('Download raw series CSV'):
    tmp2 = df.reset_index().rename(columns={'index':'date'})
    st.download_button('Download raw', tmp2.to_csv(index=False).encode('utf-8'), file_name='raw_series_v8.csv', mime='text/csv')

st.caption('v8 — professional anomaly-first money-market stress dashboard. Provide feedback and I will iterate (NY Fed Markets API, extra diagnostics, alert rules).')
