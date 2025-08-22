# streamlit_money_market_dashboard_v5.py
"""
Funding/Credit/Liquidity Stress Dashboard (v5)
- Embedded dynamic scrapers for public sources:
  - NY Fed OMO (repo) page: finds latest Excel/XLSX and parses repo rates
  - Optional OFR MMF Monitor & Fed DDP scrapers (configurable URLs)
- Uses FRED for many public series (requires API key)
- No synthetic data. Upload CSV fallback available.
- Clean ingestion, recent-focused charts, stress meters and alerts.
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

# Defensive optional imports
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

st.set_page_config(page_title="Funding/Credit/Liquidity Stress Dashboard (v5)", layout="wide")

# ----------------------------
# Utilities: cleaning & IO
# ----------------------------
def clean_series(s: pd.Series, name: str) -> Optional[pd.Series]:
    """Normalize series: datetime index, numeric values, sort, dedupe, drop nans"""
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
        return s
    except Exception:
        return None

def read_csv_series(uploaded, name: str) -> Optional[pd.Series]:
    try:
        df = pd.read_csv(uploaded)
        df.columns = [c.strip().lower() for c in df.columns]
        if 'date' not in df.columns or 'value' not in df.columns:
            # try first two columns as date,value
            if len(df.columns) >= 2:
                df = df.rename(columns={df.columns[0]:'date', df.columns[1]:'value'})
            else:
                return None
        s = pd.Series(df['value'].values, index=pd.to_datetime(df['date'], errors='coerce'), name=name)
        return clean_series(s, name)
    except Exception:
        return None

def fetch_csv_from_url(url: str) -> Optional[pd.DataFrame]:
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        content = resp.content
        df = pd.read_csv(BytesIO(content))
        return df
    except Exception:
        return None

def fetch_excel_from_url(url: str, sheet_name: Optional[str] = 0) -> Optional[pd.DataFrame]:
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        x = BytesIO(resp.content)
        # let pandas infer engine
        df = pd.read_excel(x, sheet_name=sheet_name)
        return df
    except Exception:
        return None

# ----------------------------
# Scraper helpers (dynamic)
# ----------------------------
def find_latest_file_link(page_url: str, extensions=('.xlsx', '.xls', '.csv')) -> Optional[str]:
    """Scrape page_url, return absolute link to newest file with one of the extensions.
    Defensive: returns first matching link encountered that contains keywords like 'transaction', 'omo', 'repo', 'data'."""
    try:
        resp = requests.get(page_url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        anchors = soup.find_all('a', href=True)
        candidates = []
        for a in anchors:
            href = a['href']
            lower = href.lower()
            if any(lower.endswith(ext) for ext in extensions):
                text = (a.get_text(" ", strip=True) or "").lower()
                # prefer links that contain relevant keywords
                score = 0
                for kw in ['omo', 'transaction', 'repo', 'reverse', 'rrp', 'money market', 'mmf', 'cp']:
                    if kw in text or kw in lower:
                        score += 1
                candidates.append((score, href))
        if not candidates:
            return None
        # pick highest score first, else first found
        candidates.sort(key=lambda x: -x[0])
        chosen = candidates[0][1]
        if chosen.startswith('http'):
            return chosen
        # make absolute
        from urllib.parse import urljoin
        return urljoin(page_url, chosen)
    except Exception:
        return None

# ----------------------------
# Source-specific scrapers (cached)
# ----------------------------
@st.cache_data(ttl=21600)
def scrape_nyfed_repo_series(omo_page_url: str = "https://www.newyorkfed.org/markets/OMO_transaction_data.html") -> Optional[pd.Series]:
    """Scrape NY Fed OMO page to find latest Excel and parse repo rates by date.
    Returns pd.Series of repo-rate (decimal, e.g., 0.02 for 2%)."""
    try:
        link = find_latest_file_link(omo_page_url, extensions=('.xlsx', '.xls', '.csv'))
        if not link:
            return None
        # try excel first, fall back to csv
        if link.lower().endswith('.csv'):
            df = fetch_csv_from_url(link)
        else:
            df = fetch_excel_from_url(link)
        if df is None or df.empty:
            return None
        # normalize columns
        cols = [c.strip().lower() for c in df.columns]
        df.columns = cols
        # heuristic: find date column and repo rate column
        date_col = None
        rate_col = None
        for c in cols:
            if 'date' in c:
                date_col = c
            if 'rate' in c and ('repo' in c or 'operation' in c or 'rate' in c):
                # prefer repo rate column names
                if 'repo' in c or 'rate' in c:
                    rate_col = c
                    break
        # fallback: any 'rate' column
        if rate_col is None:
            for c in cols:
                if 'rate' in c:
                    rate_col = c
                    break
        if date_col is None or rate_col is None:
            # try looking for keywords in first 5 columns if names are odd
            for c in cols[:5]:
                if any(k in c for k in ['repo','rate','operation']):
                    rate_col = rate_col or c
                if 'date' in c:
                    date_col = c
        if date_col is None or rate_col is None:
            return None
        series = pd.Series(df[rate_col].values, index=pd.to_datetime(df[date_col], errors='coerce'), name='repo_rate')
        series = clean_series(series, 'repo_rate')
        # Many repo rates might be percent strings (e.g. '2.00%'); ensure numeric decimal
        if series is not None and series.dtype == object:
            series = pd.to_numeric(series.str.replace('%','', regex=False), errors='coerce')/100.0
            series = clean_series(series, 'repo_rate')
        return series
    except Exception:
        return None

@st.cache_data(ttl=21600)
def scrape_ofr_mmf_series(ofr_page_url: str = "https://www.financialresearch.gov/") -> Optional[pd.Series]:
    """Attempt to find OFR Money Market Fund monitor CSV and extract net flows.
    Default OFR home page is scanned; if you have a specific OFR monitor page URL provide it in sidebar.
    Return series of mmf net flows (numbers)."""
    try:
        link = find_latest_file_link(ofr_page_url, extensions=('.csv', '.xlsx', '.xls'))
        if not link:
            return None
        if link.lower().endswith('.csv'):
            df = fetch_csv_from_url(link)
        else:
            df = fetch_excel_from_url(link)
        if df is None or df.empty:
            return None
        cols = [c.strip().lower() for c in df.columns]
        df.columns = cols
        # heuristic: find date and net flows columns
        date_col = None
        flows_col = None
        for c in cols:
            if 'date' in c:
                date_col = c
            if 'flow' in c or 'net' in c:
                flows_col = c
        if date_col is None or flows_col is None:
            # try first two columns
            if len(cols) >= 2:
                date_col = date_col or cols[0]
                flows_col = flows_col or cols[1]
        s = pd.Series(df[flows_col].values, index=pd.to_datetime(df[date_col], errors='coerce'), name='mmf_flows')
        s = clean_series(s, 'mmf_flows')
        return s
    except Exception:
        return None

@st.cache_data(ttl=21600)
def fetch_fed_ddp_cp(ddp_base: str = "https://www.federalreserve.gov/feeds/datadownload.htm") -> Optional[Dict[str, pd.Series]]:
    """Attempt to fetch commercial paper CSVs from Fed DDP. This is a best-effort scraper because DDP pages vary.
    Returns dict of series, e.g., {'totalcp': Series, 'cpn3m': Series, ...}"""
    try:
        # Try to find CSV links on the DDP base page
        link = find_latest_file_link(ddp_base, extensions=('.csv',))
        if not link:
            return None
        df = fetch_csv_from_url(link)
        if df is None or df.empty:
            return None
        cols = [c.strip().lower() for c in df.columns]
        df.columns = cols
        # heuristics: look for total commercial paper outstanding, CP rates etc.
        series_map = {}
        # try to find total outstanding
        tot_candidates = [c for c in cols if 'total' in c and 'commercial' in c or 'outstanding' in c or 'cp' in c]
        if tot_candidates:
            s = pd.Series(df[tot_candidates[0]].values, index=pd.to_datetime(df[cols[0]], errors='coerce'), name='totalcp')
            s = clean_series(s, 'totalcp')
            series_map['totalcp'] = s
        # If CP rates present (3m), try to find them
        rate_candidates = [c for c in cols if '3m' in c and ('cp' in c or 'commercial' in c)]
        if rate_candidates:
            s = pd.Series(df[rate_candidates[0]].values, index=pd.to_datetime(df[cols[0]], errors='coerce'), name='cpn3m')
            s = clean_series(s, 'cpn3m')
            series_map['cpn3m'] = s
        return series_map if series_map else None
    except Exception:
        return None

# ----------------------------
# FRED fetcher
# ----------------------------
@st.cache_data(ttl=21600)
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

# ----------------------------
# UI / Sidebar inputs
# ----------------------------
st.sidebar.title("Inputs & data sources (v5)")
st.sidebar.markdown("Provide FRED API key (optional) or upload CSVs. The app scrapes NY Fed (repo), OFR (MMF) and Fed DDP (CP) dynamically; you can override URLs below.")

fred_key = None
if hasattr(st, 'secrets') and isinstance(st.secrets, dict) and st.secrets.get("FRED_API_KEY"):
    fred_key = st.secrets.get("FRED_API_KEY")
fred_key = fred_key or os.environ.get("FRED_API_KEY")
fk = st.sidebar.text_input("FRED API key (optional)", value=fred_key or "")
fred_key = fk or fred_key

# override pages if needed
nyfed_url = st.sidebar.text_input("NY Fed OMO page URL (auto-default)", value="https://www.newyorkfed.org/markets/OMO_transaction_data.html")
ofr_url = st.sidebar.text_input("OFR MMF monitor page URL (optional)", value="https://www.financialresearch.gov/")
fed_ddp_url = st.sidebar.text_input("Fed DDP page URL (optional)", value="https://www.federalreserve.gov/feeds/datadownload.htm")

# Upload fallbacks
st.sidebar.markdown("Uploads (CSV with columns: date,value)")
up_repo = st.sidebar.file_uploader("Upload Repo spread/rates CSV", type=['csv','xlsx','xls'])
up_mmf = st.sidebar.file_uploader("Upload MMF flows CSV", type=['csv','xlsx','xls'])
up_cp = st.sidebar.file_uploader("Upload CP CSV", type=['csv','xlsx','xls'])

# model / display settings
horizon = st.sidebar.selectbox("Chart horizon", options=["6M","1Y","2Y","5Y"], index=1)
h_map = {"6M":182,"1Y":365,"2Y":365*2,"5Y":365*5}
h_days = h_map[horizon]
roll_window = st.sidebar.slider("Rolling window (business days) for z-scores", min_value=20, max_value=120, value=60)
compute_pca = st.sidebar.checkbox("Compute PCA (optional, requires scikit-learn)", value=True)

# ----------------------------
# Ingest data: FRED + scrapers + uploads
# ----------------------------
# friendly name mapping
FRIENDLY = {
    'sofr': 'SOFR (Overnight Secured Financing Rate)',
    'fedfunds': 'Effective Fed Funds Rate',
    'dtb1': '1M T-Bill Yield',
    'dtb3': '3M T-Bill Yield',
    'dtb6': '6M T-Bill Yield',
    'dtb12': '12M T-Bill Yield',
    'tedrate': 'TED Spread',
    'totalcp': 'Total Commercial Paper Outstanding',
    'cpn3m': 'Commercial Paper 3M Rate',
    'cpf3m': 'Commercial Paper 3M (Financial)',
    'repo_rate': 'Repo Operation Rate (NY Fed OMO)',
    'repo_spread': 'Repo Spread (Repo - Fed Funds)',
    'mmf_flows': 'MMF Net Flows',
    'ig_oas': 'IG OAS',
    'hy_oas': 'HY OAS',
    'rrp': 'Overnight Reverse Repo Outstanding'
}

# FRED series mapping (public)
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
    'ig_oas': 'BAMLC0A0CM',
    'hy_oas': 'BAMLH0A0HYM2',
    'rrp': 'RRPONTSYD'  # Reverse repo outstanding
}

series: Dict[str, pd.Series] = {}

# 1) FRED pulls
if fred_key:
    for k,v in FRED_SERIES.items():
        s = load_fred_series(v, fred_key)
        if s is not None:
            # rename to short key
            s.name = k
            series[k] = s

# 2) NY Fed repo scraper (preferred)
repo_series = None
if up_repo is not None:
    # prefer uploaded CSV/XLSX
    if str(up_repo.name).lower().endswith(('.xls','.xlsx')):
        df_up = pd.read_excel(up_repo)
        # attempt to parse to series
        try:
            df_up.columns = [c.strip().lower() for c in df_up.columns]
            date_col = [c for c in df_up.columns if 'date' in c]
            rate_col = [c for c in df_up.columns if 'repo' in c or 'rate' in c]
            if date_col and rate_col:
                s = pd.Series(df_up[rate_col[0]].values, index=pd.to_datetime(df_up[date_col[0]], errors='coerce'))
                repo_series = clean_series(s,'repo_rate')
        except Exception:
            repo_series = None
    else:
        s = read_csv_series(up_repo, 'repo_rate')
        repo_series = s
else:
    # dynamic scrape
    repo_series = scrape_nyfed_repo_series(nyfed_url)

if repo_series is not None:
    series['repo_rate'] = repo_series

# 3) OFR MMF scraper or upload
mmf_series = None
if up_mmf is not None:
    mmf_series = read_csv_series(up_mmf, 'mmf_flows')
else:
    mmf_series = scrape_ofr_mmf_series(ofr_page_url=ofr_url)  # may be None
if mmf_series is not None:
    series['mmf_flows'] = mmf_series

# 4) Fed DDP CP scraper or upload
cp_map = {}
if up_cp is not None:
    # attempt to read file for cp series
    if str(up_cp.name).lower().endswith(('.xls','.xlsx')):
        df_up = pd.read_excel(up_cp)
        df_up.columns = [c.strip().lower() for c in df_up.columns]
        # heuristics
        date_col = [c for c in df_up.columns if 'date' in c]
        col_candidates = [c for c in df_up.columns if 'cp' in c or 'commercial' in c or 'outstanding' in c or '3m' in c]
        if date_col and col_candidates:
            # take first as total
            s = pd.Series(df_up[col_candidates[0]].values, index=pd.to_datetime(df_up[date_col[0]], errors='coerce'))
            cp_map['totalcp'] = clean_series(s,'totalcp')
else:
    cp_map = fetch_fed_ddp_cp(fed_ddp_url) or {}

if cp_map:
    for k,v in cp_map.items():
        if v is not None:
            series[k] = v

# 5) Ensure we have at least some series; otherwise show help
if not series:
    st.title("Funding/Credit/Liquidity Stress Dashboard (v5)")
    st.info("No public series loaded. Provide a FRED API key (sidebar) or upload CSVs for repo/MMF/CP. The app attempts to scrape NY Fed (repo), OFR (MMF) and Fed DDP automatically.")
    st.stop()

# 6) Build DataFrame aligned to business days across union of dates
all_dates = sorted({d for s in series.values() for d in s.index})
start, end = all_dates[0], all_dates[-1]
bidx = pd.bdate_range(start=start, end=end)
for k in list(series.keys()):
    series[k] = series[k].reindex(bidx)

df = pd.DataFrame(series)

# 7) Derive computed indicators: repo_spread = repo_rate - fedfunds (if both present)
if 'repo_rate' in df.columns and 'fedfunds' in df.columns:
    # fedfunds is usually monthly/daily in percent; convert to decimal if needed
    df['repo_spread'] = df['repo_rate'] - df['fedfunds']

# ----------------------------
# Modeling: z-scores, PCA, scores
# ----------------------------
# compute rolling z for each series
z = pd.DataFrame(index=df.index)
for col in df.columns:
    z[col] = df[col].rolling(window=roll_window, min_periods=max(5, roll_window//3)).apply(
        lambda x: (x[-1] - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) != 0 else np.nan)
    )

# invert mmf flows (outflows -> positive stress)
if 'mmf_flows' in z.columns:
    z['mmf_flows'] = -z['mmf_flows']

# category groupings
FUNDING_COLS = ['sofr', 'repo_spread', 'fra_ois', 'fedfunds', 'dtb1', 'dtb3']
CREDIT_COLS = ['ig_oas', 'hy_oas']
LIQUIDITY_COLS = ['mmf_flows', 'totalcp', 'cpn3m', 'cpf3m', 'rrp']

def mean_abs_z(cols: List[str]) -> Optional[float]:
    present = [c for c in cols if c in z.columns]
    if not present:
        return np.nan
    vals = z[present].iloc[-1].abs().dropna()
    return float(vals.mean()) if not vals.empty else np.nan

fund_z = mean_abs_z(FUNDING_COLS)
cred_z = mean_abs_z(CREDIT_COLS)
liq_z = mean_abs_z(LIQUIDITY_COLS)

def z_to_score(zv: Optional[float]) -> int:
    if pd.isna(zv):
        return 0
    return int(np.clip((zv / 3.0) * 100, 0, 100))

fund_score = z_to_score(fund_z)
cred_score = z_to_score(cred_z)
liq_score = z_to_score(liq_z)
combined_score = int(np.clip(0.4*fund_score + 0.4*cred_score + 0.2*liq_score, 0, 100))

# PCA (optional)
pc1 = pd.Series(index=df.index, data=np.nan)
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

# Mahalanobis anomaly (simple)
maha_score = 0
try:
    Xw = z.dropna(axis=1, how='all').tail(roll_window).dropna()
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

# ----------------------------
# Display: Top meters (Funding, Credit, Liquidity, Combined)
# ----------------------------
st.title("Funding / Credit / Liquidity Stress Dashboard (v5)")
st.markdown(f"Showing recent horizon: **{horizon}**. Rolling z-window = **{roll_window}** business days.")

def color_for_score(s:int)->str:
    if s <= 40: return "#0f9d58"  # green
    if s <= 70: return "#f4b400"  # orange
    return "#db4437"  # red

c1,c2,c3,c4 = st.columns([1,1,1,1])

def show_gauge_plotly(col, label, value):
    color = color_for_score(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
      
