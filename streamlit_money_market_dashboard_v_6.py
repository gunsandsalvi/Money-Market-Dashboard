"""
streamlit_money_market_dashboard_v6.py

Funding / Credit / Liquidity Stress Dashboard — v6
- Mobile-first, anomaly-focused, public-data sources (FRED + scrapers for NY Fed/OFR/Fed DDP)
- Shows anomalies (deviations / z-scores) rather than raw levels
- Logs scraper status, handles missing data gracefully
- Top meters: Funding, Credit, Liquidity, Combined
- Charts focus on recent horizons (6M/1Y/2Y/5Y)

Run:
    pip install streamlit pandas numpy requests beautifulsoup4 fredapi plotly scikit-learn openpyxl xlrd
    streamlit run streamlit_money_market_dashboard_v6.py

Notes:
- Provide FRED API key in sidebar, or via st.secrets['FRED_API_KEY'] or FRED_API_KEY env var.
- Scrapers run and cache results (6 hours TTL). If they fail, upload CSVs as fallback.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from datetime import datetime, timedelta
import os
from typing import Optional, Dict, List

# Optional libs
PLOTLY = False
SKLEARN = False
FREDAPI = False
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY = True
except Exception:
    PLOTLY = False

try:
    from sklearn.decomposition import PCA
    SKLEARN = True
except Exception:
    SKLEARN = False

try:
    from fredapi import Fred
    FREDAPI = True
except Exception:
    FREDAPI = False

st.set_page_config(page_title="Funding Stress Dashboard v6", layout="wide")

# ---------------------- Helpers: cleaning, IO, scraping ----------------------
@st.cache_data(ttl=21600)
def fetch_url_content(url: str) -> Optional[str]:
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None

@st.cache_data(ttl=21600)
def fetch_bytes(url: str) -> Optional[bytes]:
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None


def clean_series(s: pd.Series, name: str) -> Optional[pd.Series]:
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


def read_csv_series(uploaded, name: str) -> Optional[pd.Series]:
    try:
        if hasattr(uploaded, 'read'):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_csv(str(uploaded))
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

@st.cache_data(ttl=21600)
def find_latest_file_link(page_url: str, extensions=('.xlsx', '.xls', '.csv')) -> Optional[str]:
    try:
        html = fetch_url_content(page_url)
        if not html:
            return None
        soup = BeautifulSoup(html, 'html.parser')
        anchors = soup.find_all('a', href=True)
        candidates = []
        for a in anchors:
            href = a['href']
            lower = href.lower()
            if any(lower.endswith(ext) for ext in extensions):
                text = (a.get_text(' ', strip=True) or '').lower()
                score = 0
                for kw in ['omo','transaction','repo','reverse','rrp','money market','mmf','commercial paper','cp']:
                    if kw in text or kw in lower:
                        score += 1
                candidates.append((score, href))
        if not candidates:
            return None
        candidates.sort(key=lambda x: -x[0])
        chosen = candidates[0][1]
        if chosen.startswith('http'):
            return chosen
        from urllib.parse import urljoin
        return urljoin(page_url, chosen)
    except Exception:
        return None

@st.cache_data(ttl=21600)
def scrape_nyfed_repo(omo_page: str = 'https://www.newyorkfed.org/markets/OMO_transaction_data.html') -> Dict[str, Optional[pd.Series]]:
    status = {'ok': False, 'msg': ''}
    try:
        link = find_latest_file_link(omo_page)
        if not link:
            status['msg'] = 'no link found'
            return {'repo_rate': None, 'status': status}
        content = fetch_bytes(link)
        if not content:
            status['msg'] = 'download failed'
            return {'repo_rate': None, 'status': status}
        # try excel first
        try:
            df = pd.read_excel(BytesIO(content))
        except Exception:
            try:
                df = pd.read_csv(BytesIO(content))
            except Exception:
                status['msg'] = 'could not parse file'
                return {'repo_rate': None, 'status': status}
        df.columns = [c.strip().lower() for c in df.columns]
        # heuristic find date & rate
        date_col = next((c for c in df.columns if 'date' in c), None)
        rate_col = next((c for c in df.columns if 'repo' in c and 'rate' in c or 'rate' in c), None)
        if rate_col is None:
            # fallback any 'rate' column
            rate_col = next((c for c in df.columns if 'rate' in c), None)
        if not date_col or not rate_col:
            # attempt to inspect first rows for numeric columns
            status['msg'] = 'could not identify date/rate columns'
            return {'repo_rate': None, 'status': status}
        s = pd.Series(df[rate_col].values, index=pd.to_datetime(df[date_col], errors='coerce'), name='repo_rate')
        s = clean_series(s, 'repo_rate')
        # convert percent strings
        if s is not None and s.dtype == object:
            try:
                s = pd.to_numeric(s.astype(str).str.replace('%','', regex=False), errors='coerce')/100.0
                s = clean_series(s, 'repo_rate')
            except Exception:
                pass
        status['ok'] = True
        status['msg'] = f'loaded {len(s) if s is not None else 0} rows'
        return {'repo_rate': s, 'status': status}
    except Exception as e:
        status['msg'] = f'exception {e}'
        return {'repo_rate': None, 'status': status}

@st.cache_data(ttl=21600)
def scrape_ofr_mmf(ofr_page: str = 'https://www.financialresearch.gov/') -> Dict[str, Optional[pd.Series]]:
    status = {'ok': False, 'msg': ''}
    try:
        link = find_latest_file_link(ofr_page)
        if not link:
            status['msg'] = 'no link found'
            return {'mmf_flows': None, 'status': status}
        # attempt load
        content = fetch_bytes(link)
        if not content:
            status['msg'] = 'download failed'
            return {'mmf_flows': None, 'status': status}
        try:
            df = pd.read_excel(BytesIO(content))
        except Exception:
            try:
                df = pd.read_csv(BytesIO(content))
            except Exception:
                status['msg'] = 'could not parse'
                return {'mmf_flows': None, 'status': status}
        df.columns = [c.strip().lower() for c in df.columns]
        date_col = next((c for c in df.columns if 'date' in c), None)
        flow_col = next((c for c in df.columns if 'flow' in c or 'net' in c), None)
        if not date_col or not flow_col:
            status['msg'] = 'could not identify cols'
            return {'mmf_flows': None, 'status': status}
        s = pd.Series(df[flow_col].values, index=pd.to_datetime(df[date_col], errors='coerce'), name='mmf_flows')
        s = clean_series(s, 'mmf_flows')
        status['ok'] = True
        status['msg'] = f'loaded {len(s) if s is not None else 0} rows'
        return {'mmf_flows': s, 'status': status}
    except Exception as e:
        status['msg'] = f'exception {e}'
        return {'mmf_flows': None, 'status': status}

@st.cache_data(ttl=21600)
def fetch_fed_ddp_cp(ddp_page: str = 'https://www.federalreserve.gov/feeds/datadownload.htm') -> Dict[str, Optional[pd.Series]]:
    status = {'ok': False, 'msg': ''}
    try:
        link = find_latest_file_link(ddp_page, extensions=('.csv',))
        if not link:
            status['msg'] = 'no link found'
            return {'totalcp': None, 'cpn3m': None, 'status': status}
        df = fetch_csv_from_link(link)
        if df is None:
            status['msg'] = 'download failed'
            return {'totalcp': None, 'cpn3m': None, 'status': status}
        df.columns = [c.strip().lower() for c in df.columns]
        date_col = next((c for c in df.columns if 'date' in c), df.columns[0])
        total_col = next((c for c in df.columns if 'total' in c and 'cp' in c or 'outstanding' in c), None)
        rate_col = next((c for c in df.columns if '3m' in c and 'cp' in c), None)
        out = {}
        if total_col:
            s = pd.Series(df[total_col].values, index=pd.to_datetime(df[date_col], errors='coerce'), name='totalcp')
            out['totalcp'] = clean_series(s, 'totalcp')
        if rate_col:
            s2 = pd.Series(df[rate_col].values, index=pd.to_datetime(df[date_col], errors='coerce'), name='cpn3m')
            out['cpn3m'] = clean_series(s2, 'cpn3m')
        status['ok'] = True
        status['msg'] = f'loaded {len(df)} rows'
        out['status'] = status
        return out
    except Exception as e:
        status['msg'] = f'exception {e}'
        return {'totalcp': None, 'cpn3m': None, 'status': status}

# small helper used above
@st.cache_data(ttl=21600)
def fetch_csv_from_link(link: str) -> Optional[pd.DataFrame]:
    try:
        b = fetch_bytes(link)
        if not b:
            return None
        return pd.read_csv(BytesIO(b))
    except Exception:
        return None

# ---------------------- FRED loader ----------------------
@st.cache_data(ttl=21600)
def load_fred(series_id: str, api_key: str) -> Optional[pd.Series]:
    if not (FREDAPI and api_key):
        return None
    try:
        fred = Fred(api_key=api_key)
        raw = fred.get_series(series_id)
        s = pd.Series(raw.values, index=pd.to_datetime(raw.index), name=series_id)
        return clean_series(s, series_id)
    except Exception:
        return None

# ---------------------- UI: Sidebar inputs ----------------------
st.sidebar.title('Inputs & data sources (v6)')
st.sidebar.markdown('Provide FRED API key or upload CSVs as fallback. App will scrape NY Fed/OFR/Fed DDP dynamically and log status.')

fred_key = None
if hasattr(st, 'secrets') and isinstance(st.secrets, dict) and st.secrets.get('FRED_API_KEY'):
    fred_key = st.secrets.get('FRED_API_KEY')
fred_key = fred_key or os.environ.get('FRED_API_KEY')
fk = st.sidebar.text_input('FRED API key (optional)', value=fred_key or '')
fred_key = fk or fred_key

nyfed_page = st.sidebar.text_input('NY Fed OMO page URL', value='https://www.newyorkfed.org/markets/OMO_transaction_data.html')
ofr_page = st.sidebar.text_input('OFR MMF page URL', value='https://www.financialresearch.gov/')
fed_ddp_page = st.sidebar.text_input('Fed DDP page URL', value='https://www.federalreserve.gov/feeds/datadownload.htm')

st.sidebar.markdown('Uploads (CSV date,value) — fallback')
up_repo = st.sidebar.file_uploader('Repo CSV', type=['csv','xls','xlsx'])
up_mmf = st.sidebar.file_uploader('MMF CSV', type=['csv','xls','xlsx'])
up_cp = st.sidebar.file_uploader('CP CSV', type=['csv','xls','xlsx'])

horizon = st.sidebar.radio('Horizon', options=['6M','1Y','2Y','5Y'], index=1)
h_map = {'6M':182,'1Y':365,'2Y':365*2,'5Y':365*5}
h_days = h_map[horizon]
roll_window = st.sidebar.slider('Rolling z-window (days)', min_value=20, max_value=120, value=60)
compute_pca = st.sidebar.checkbox('Compute PCA (optional)', value=True)

# ---------------------- Data ingestion: FRED + scrapers + uploads ----------------------
FRIENDLY = {
    'sofr': 'SOFR (overnight)', 'fedfunds':'Fed Funds (EFFR)', 'dtb1':'1M T-bill', 'dtb3':'3M T-bill',
    'repo_rate':'Repo Operation Rate (NY Fed)', 'repo_spread':'Repo - Fed Funds (anomaly)',
    'fra_ois':'FRA - OIS', 'mmf_flows':'MMF Net Flows', 'totalcp':'Total CP', 'cpn3m':'CP 3M',
    'ig_oas':'IG OAS', 'hy_oas':'HY OAS', 'rrp':'Reverse Repo Outstanding'
}

FRED_MAP = {
    'sofr':'SOFR', 'fedfunds':'FEDFUNDS', 'dtb1':'DTB1', 'dtb3':'DTB3', 'dtb6':'DTB6', 'dtb12':'DTB12',
    'ig_oas':'BAMLC0A0CM', 'hy_oas':'BAMLH0A0HYM2', 'totalcp':'TOTALCP', 'cpn3m':'CPN3M', 'cpf3m':'CPF3M', 'rrp':'RRPONTSYD'
}

series: Dict[str,pd.Series] = {}
source_status = {}

# FRED pulls
if fred_key:
    for k,v in FRED_MAP.items():
        s = load_fred(v, fred_key)
        source_status[k] = {'source':'FRED','ok': bool(s is not None), 'rows': len(s) if s is not None else 0}
        if s is not None:
            s.name = k
            series[k] = s
else:
    for k in FRED_MAP.keys():
        source_status[k] = {'source':'FRED','ok': False, 'rows': 0}

# NY Fed repo scraper or upload
if up_repo is not None:
    repo_s = None
    try:
        if str(up_repo.name).lower().endswith(('.xls','.xlsx')):
            dfx = pd.read_excel(up_repo)
            dfx.columns = [c.strip().lower() for c in dfx.columns]
            date_col = next((c for c in dfx.columns if 'date' in c), None)
            rate_col = next((c for c in dfx.columns if 'repo' in c or 'rate' in c), None)
            if date_col and rate_col:
                s = pd.Series(dfx[rate_col].values, index=pd.to_datetime(dfx[date_col], errors='coerce'))
                repo_s = clean_series(s,'repo_rate')
        else:
            repo_s = read_csv_series(up_repo,'repo_rate')
    except Exception:
        repo_s = None
    series['repo_rate'] = repo_s
    source_status['repo_rate'] = {'source':'upload','ok':bool(repo_s is not None),'rows':len(repo_s) if repo_s is not None else 0}
else:
    ny = scrape_nyfed_repo(nyfed_page)
    series['repo_rate'] = ny.get('repo_rate')
    source_status['repo_rate'] = {'source':'NYFed_scrape','ok':ny.get('status',{}).get('ok',False),'msg': ny.get('status',{}).get('msg','')}

# OFR MMF
if up_mmf is not None:
    mmfs = read_csv_series(up_mmf,'mmf_flows')
    series['mmf_flows'] = mmfs
    source_status['mmf_flows'] = {'source':'upload','ok':bool(mmfs is not None),'rows':len(mmfs) if mmfs is not None else 0}
else:
    ofr = scrape_ofr_mmf(ofr_page)
    series['mmf_flows'] = ofr.get('mmf_flows')
    source_status['mmf_flows'] = {'source':'OFR_scrape','ok':ofr.get('status',{}).get('ok',False),'msg': ofr.get('status',{}).get('msg','')}

# Fed DDP CP
if up_cp is not None:
    cp_map = {}
    try:
        if str(up_cp.name).lower().endswith(('.xls','.xlsx')):
            dfx = pd.read_excel(up_cp)
            dfx.columns = [c.strip().lower() for c in dfx.columns]
            date_col = next((c for c in dfx.columns if 'date' in c), None)
            cand = [c for c in dfx.columns if 'cp' in c or 'commercial' in c or 'outstanding' in c or '3m' in c]
            if date_col and cand:
                s = clean_series(pd.Series(dfx[cand[0]].values, index=pd.to_datetime(dfx[date_col], errors='coerce')), 'totalcp')
                cp_map['totalcp'] = s
    except Exception:
        cp_map = {}
    if cp_map.get('totalcp') is not None:
        series['totalcp'] = cp_map['totalcp']
        source_status['totalcp'] = {'source':'upload','ok':True,'rows':len(cp_map['totalcp'])}
else:
    cpmap = fetch_fed_ddp_cp(fed_ddp_page)
    if cpmap and isinstance(cpmap, dict):
        for k,v in cpmap.items():
            if k!='status':
                series[k] = v
                source_status[k] = {'source':'FedDDP_scrape','ok':bool(v is not None),'rows':len(v) if v is not None else 0}
    else:
        source_status['totalcp'] = {'source':'FedDDP_scrape','ok':False,'rows':0}

# If no series at all, stop
if not series:
    st.title('Funding Stress Dashboard v6')
    st.info('No data available. Provide FRED key or upload CSVs / check scrapers in sidebar.')
    st.stop()

# Align to business days across union of dates
# Only keep successful series (drop None)
clean_series = {k: s for k, s in series.items() if s is not None and not s.empty}

if not clean_series:
    st.error("No data available from scrapers or FRED — please check sources.")
    st.stop()

# Align to business days across union of dates
all_dates = sorted({d for s in clean_series.values() for d in s.index})
start, end = all_dates[0], all_dates[-1]
bidx = pd.bdate_range(start=start, end=end)

# Reindex all series
for k, s in clean_series.items():
    clean_series[k] = s.reindex(bidx)

series = clean_series
start, end = all_dates[0], all_dates[-1]
bidx = pd.bdate_range(start=start, end=end)
for k in list(series.keys()):
    series[k] = series[k].reindex(bidx)

# Compose DataFrame
df = pd.DataFrame(series)

# Derived: repo_spread = repo_rate - fedfunds (if both present)
if 'repo_rate' in df.columns and 'fedfunds' in df.columns:
    df['repo_spread'] = df['repo_rate'] - df['fedfunds']

# ---------------------- Anomaly calculations ----------------------
# Define anomaly series to display (differences or normalized versions)
# We'll compute anomalies in basis points for clarity where appropriate

anomalies = pd.DataFrame(index=df.index)

# SOFR anomaly: SOFR - Fed Funds
if 'sofr' in df.columns and 'fedfunds' in df.columns:
    anomalies['sofr_minus_fedfunds_bps'] = (df['sofr'] - df['fedfunds']) * 10000

# Repo spread (bps)
if 'repo_spread' in df.columns:
    anomalies['repo_spread_bps'] = df['repo_spread'] * 10000

# FRA-OIS if present (assume already a spread)
if 'fra_ois' in df.columns:
    anomalies['fra_ois_bps'] = df['fra_ois'] * 10000

# TED spread (already difference) in bps
if 'tedrate' in df.columns:
    anomalies['ted_bps'] = df['tedrate'] * 100

# Credit spreads in bps
if 'ig_oas' in df.columns:
    anomalies['ig_oas_bps'] = df['ig_oas']
if 'hy_oas' in df.columns:
    anomalies['hy_oas_bps'] = df['hy_oas']
if 'hy_oas' in df.columns and 'ig_oas' in df.columns:
    anomalies['hy_minus_ig_bps'] = df['hy_oas'] - df['ig_oas']

# CP outstanding (absolute) and CP rate (bps)
if 'totalcp' in df.columns:
    anomalies['totalcp'] = df['totalcp']
if 'cpn3m' in df.columns:
    anomalies['cpn3m_bps'] = df['cpn3m']

# MMF flows (absolute amounts)
if 'mmf_flows' in df.columns:
    anomalies['mmf_flows'] = df['mmf_flows']

# RRP outstanding as liquidity proxy
if 'rrp' in df.columns:
    anomalies['rrp'] = df['rrp']

# Clean anomalies
for c in anomalies.columns:
    anomalies[c] = clean_series(anomalies[c], c)

# ---------------------- Z-scores (rolling) ----------------------
z = pd.DataFrame(index=anomalies.index)
for col in anomalies.columns:
    z[col] = anomalies[col].rolling(window=roll_window, min_periods=max(5, roll_window//3)).apply(
        lambda x: (x[-1] - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) != 0 else np.nan)
    )

# For flows/outstanding where direction differs, keep sign as-is (we'll treat abs for stress)

# ---------------------- Scoring ----------------------
FUNDING_ANOMS = ['sofr_minus_fedfunds_bps','repo_spread_bps','fra_ois_bps','ted_bps']
CREDIT_ANOMS = ['ig_oas_bps','hy_oas_bps','hy_minus_ig_bps']
LIQUIDITY_ANOMS = ['mmf_flows','totalcp','cpn3m_bps','rrp']

def mean_abs_latest(cols: List[str]) -> Optional[float]:
    present = [c for c in cols if c in z.columns]
    if not present:
        return np.nan
    vals = z[present].iloc[-1].abs().dropna()
    return float(vals.mean()) if not vals.empty else np.nan

fund_z = mean_abs_latest(FUNDING_ANOMS)
cred_z = mean_abs_latest(CREDIT_ANOMS)
liq_z = mean_abs_latest(LIQUIDITY_ANOMS)

def z_to_score(zv: Optional[float]) -> int:
    if pd.isna(zv):
        return 0
    return int(np.clip((zv / 3.0) * 100, 0, 100))

fund_score = z_to_score(fund_z)
cred_score = z_to_score(cred_z)
liq_score = z_to_score(liq_z)
combined_score = int(np.clip(0.4*fund_score + 0.4*cred_score + 0.2*liq_score, 0, 100))

# ---------------------- UI: display & mobile formatting ----------------------
st.title('Funding / Credit / Liquidity Stress Dashboard (v6)')
st.markdown(f"Horizon: **{horizon}** — charts show recent anomalies (not raw levels). Rolling z-window = **{roll_window}** days.")

# Source status table
st.sidebar.header('Source status (last run)')
status_rows = []
for k,v in source_status.items():
    if isinstance(v, dict):
        ok = v.get('ok', False)
        rows = v.get('rows', v.get('rows', 0))
        msg = v.get('msg','')
        status_rows.append({'series':k, 'source': v.get('source',''), 'ok': ok, 'rows': rows, 'msg': msg})
st.sidebar.table(pd.DataFrame(status_rows))

# Top cards stacked for mobile — use single column on narrow
col1, col2, col3, col4 = st.columns([1,1,1,1])

def color_for(score:int)->str:
    if score<=40: return '#0f9d58'
    if score<=70: return '#f4b400'
    return '#db4437'

if PLOTLY:
    def gauge_plot(col, label, val):
        fig = go.Figure(go.Indicator(
            mode='gauge+number', value=val, title={'text':label},
            gauge={'axis':{'range':[0,100]}, 'bar':{'color':color_for(val)},
                   'steps':[{'range':[0,40],'color':'#eafaf1'},{'range':[40,70],'color':'#fff8e5'},{'range':[70,100],'color':'#ffecec'}]}
        ))
        fig.update_layout(height=200, margin=dict(l=5,r=5,t=25,b=5))
        col.plotly_chart(fig, use_container_width=True)
else:
    def gauge_plot(col, label, val):
        col.markdown(f'**{label}**')
        col.metric('', f'{val}')
        col.markdown(f"<div style='height:8px;background:{color_for(val)};border-radius:6px'></div>", unsafe_allow_html=True)

gauge_plot(col1, 'Funding Stress (0-100)', fund_score)
gauge_plot(col2, 'Credit Stress (0-100)', cred_score)
gauge_plot(col3, 'Liquidity Stress (0-100)', liq_score)
gauge_plot(col4, 'Combined Stress (0-100)', combined_score)

st.markdown('---')

# Charts: show anomalies (each chart single-scale) clipped to horizon
end_dt = df.index.max()
start_dt = end_dt - pd.Timedelta(days=h_days)

st.header('Anomaly charts (focused)')

# helper plot function

def plot_df(df_plot: pd.DataFrame, title: str, height: int=260):
    if df_plot.empty:
        st.info('No series available for chart')
        return
    if PLOTLY:
        fig = go.Figure()
        for c in df_plot.columns:
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[c], name=c, mode='lines'))
        fig.update_layout(title=title, height=height, margin=dict(l=5,r=5,t=30,b=5))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(df_plot)

# Funding anomalies
fund_anoms = [c for c in FUNDING_ANOMS if c in anomalies.columns]
if fund_anoms:
    df_fund = anomalies[fund_anoms].loc[start_dt: end_dt]
    df_fund = df_fund.rename(columns=lambda x: x.replace('_bps',' (bps)').replace('_',' '))
    st.subheader('Funding anomalies')
    plot_df(df_fund, f'Funding anomalies — last {horizon}')

# Credit anomalies
cred_anoms = [c for c in CREDIT_ANOMS if c in anomalies.columns]
if cred_anoms:
    df_cred = anomalies[cred_anoms].loc[start_dt:end_dt]
    df_cred = df_cred.rename(columns=lambda x: x.replace('_bps',' (bps)').replace('_',' '))
    st.subheader('Credit anomalies')
    plot_df(df_cred, f'Credit anomalies — last {horizon}')

# Liquidity anomalies
liq_anoms = [c for c in LIQUIDITY_ANOMS if c in anomalies.columns]
if liq_anoms:
    df_liq = anomalies[liq_anoms].loc[start_dt:end_dt]
    df_liq = df_liq.rename(columns=lambda x: x.replace('_bps',' (bps)').replace('_',' '))
    st.subheader('Liquidity anomalies')
    plot_df(df_liq, f'Liquidity anomalies — last {horizon}', height=320)

# Heatmap of recent z-scores (last ~120 bdays)
st.markdown('---')
st.header('Recent z-score heatmap')
heat_window = min(len(z), 120)
if heat_window >= 10:
    z_recent = z.tail(heat_window)
    rows = []
    for grp in (FUNDING_ANOMS, CREDIT_ANOMS, LIQUIDITY_ANOMS):
        rows.extend([c for c in grp if c in z_recent.columns])
    if rows:
        heat_df = z_recent[rows].T
        heat_df.index = [FRIENDLY.get(r,r) for r in heat_df.index]
        if PLOTLY:
            fig = px.imshow(heat_df, aspect='auto', labels=dict(x='Date', y='Indicator', color='z-score'))
            fig.update_layout(height=360, margin=dict(l=5,r=5,t=20,b=5))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(heat_df)
else:
    st.info('Not enough z-score history to build heatmap')

# Alerts panel
st.markdown('---')
st.header('Alerts — recent z breaches (|z| >= 2)')
alerts = []
if not z.empty:
    last = z.iloc[-1]
    for c in last.index:
        v = last[c]
        if pd.notna(v) and abs(v) >= 2.0:
            cat = 'Funding' if c in FUNDING_ANOMS else 'Credit' if c in CREDIT_ANOMS else 'Liquidity' if c in LIQUIDITY_ANOMS else 'Other'
            alerts.append((c,v,cat))
if alerts:
    for c,v,cat in alerts:
        st.error(f"⚠️ {FRIENDLY.get(c,c)} | {cat} | z = {v:.2f}")
else:
    st.success('No recent z-score breaches (|z| >= 2)')

# Export
st.markdown('---')
st.header('Export')
if st.button('Download raw indicators (CSV)'):
    tmp = df.reset_index().rename(columns={'index':'date'})
    st.download_button('Download CSV', tmp.to_csv(index=False).encode('utf-8'), file_name='indicators_raw.csv', mime='text/csv')
if st.button('Download anomalies (CSV)'):
    tmp2 = anomalies.reset_index().rename(columns={'index':'date'})
    st.download_button('Download CSV', tmp2.to_csv(index=False).encode('utf-8'), file_name='anomalies.csv', mime='text/csv')

st.caption('v6: anomaly-focused dashboard. All series cleaned before use. Scrapers cached for 6h. If scrapers fail, upload CSVs in sidebar.')
