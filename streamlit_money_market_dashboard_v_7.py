"""
Money Market Stress Dashboard v7
- Uses documented APIs: FRED (fredapi) + OFR Short-term Funding Monitor API (JSON)
- Focused on anomalies (SOFR-FedFunds, Repo spread, FRA-OIS, IG/HY anomalies, MMF flows, RRP)
- Robust: reports source status, handles missing series, no synthetic numbers

Run:
    pip install -r requirements.txt
    streamlit run streamlit_money_market_dashboard_v7.py

Notes:
- Provide FRED API key in the sidebar, or via st.secrets['FRED_API_KEY'] or FRED_API_KEY env var.
- The app caches API calls for 6 hours (TTL = 21600 seconds).
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from datetime import timedelta
import os
from typing import Dict, Optional, List

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
    from fredapi import Fred
    FREDAPI = True
except Exception:
    FREDAPI = False

try:
    from sklearn.decomposition import PCA
    SKLEARN = True
except Exception:
    SKLEARN = False

st.set_page_config(page_title="Money Market Stress Dashboard v7", layout="wide")

# -------------------------
# Utilities
# -------------------------
@st.cache_data(ttl=21600)
def ofr_fetch_dataset(dataset_key: str) -> Optional[dict]:
    """
    OFR short-term funding monitor API
    Example: https://data.financialresearch.gov/v1/series/dataset?dataset=mmf
    """
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
        return clean_ts(s, series_id)
    except Exception:
        return None


def clean_ts(s: pd.Series, name: str) -> Optional[pd.Series]:
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

@st.cache_data(ttl=21600)
def parse_ofr_series_json(json_blob: dict, series_id_hint: Optional[str] = None) -> Dict[str, pd.Series]:
    """
    Parse OFR dataset JSON response into dictionary of pandas.Series.
    """
    out = {}
    if not json_blob:
        return out
    timeseries = json_blob.get("timeseries") or json_blob.get("data") or {}
    for mnemonic, details in timeseries.items():
        try:
            ts = details.get("timeseries") or details.get("timeseries", {})
            if isinstance(ts, dict):
                inner = None
                if "aggregation" in ts:
                    inner = ts["aggregation"]
                else:
                    for k2, v2 in ts.items():
                        if isinstance(v2, dict) and "aggregation" in v2:
                            inner = v2["aggregation"]
                            break
                if inner:
                    dates = [pd.to_datetime(d[0]) for d in inner if d and d[0] is not None]
                    vals = [d[1] for d in inner if d and len(d) > 1]
                    s = pd.Series(vals, index=pd.to_datetime(dates), name=mnemonic)
                    s = clean_ts(s, mnemonic)
                    if s is not None and not s.empty:
                        out[mnemonic] = s
        except Exception:
            continue
    return out


def safe_union_index(series_map: Dict[str, pd.Series]) -> pd.DatetimeIndex:
    valid = [s for s in series_map.values() if s is not None and not s.empty]
    if not valid:
        return pd.DatetimeIndex([])
    all_dates = pd.Index([]).union_many([s.index for s in valid])
    bidx = pd.bdate_range(start=all_dates.min(), end=all_dates.max())
    return bidx

# -------------------------
# Inputs / sidebar
# -------------------------
st.sidebar.title("Settings & keys (v7)")
fred_key = None
if hasattr(st, "secrets") and isinstance(st.secrets, dict) and st.secrets.get("FRED_API_KEY"):
    fred_key = st.secrets.get("FRED_API_KEY")
fred_key = fred_key or os.environ.get("FRED_API_KEY")
fred_key_input = st.sidebar.text_input("FRED API key (optional)", value=fred_key or "")
fred_key = fred_key_input or fred_key

st.sidebar.markdown("APIs used: FRED (fredapi) and OFR Short-Term Funding Monitor (JSON API).")

horizon = st.sidebar.selectbox("Chart horizon", ["6M","1Y","2Y","5Y"], index=1)
H_MAP = {"6M":182, "1Y":365, "2Y":365*2, "5Y":365*5}
h_days = H_MAP[horizon]
roll_window = st.sidebar.slider("Rolling z window (business days)", 20, 120, 60)
compute_pca = st.sidebar.checkbox("Compute PCA (optional)", value=False)

# -------------------------
# Which series to fetch (FRED IDs)
# -------------------------
FRED_REQ = {
    "sofr": "SOFR",
    "fedfunds": "FEDFUNDS",
    "dtb3": "DTB3",
    "totalcp": "TOTALCP",
    "cpn3m": "CPN3M",
    "cpf3m": "CPF3M",
    "ig_oas": "BAMLC0A0CM",
    "hy_oas": "BAMLH0A0HYM2",
    "rrp": "RRPONTSYD",
}

# -------------------------
# Fetch FRED series
# -------------------------
st.header("Loading data (v7)")
status = {}
series: Dict[str, pd.Series] = {}

if fred_key and FREDAPI:
    st.sidebar.success("FRED: fredapi available")
    for key, sid in FRED_REQ.items():
        s = fred_get_series(sid, fred_key)
        status[key] = {"source":"FRED", "series_id":sid, "ok": bool(s is not None), "n": len(s) if s is not None else 0}
        if s is not None:
            s.name = key
            series[key] = s
else:
    for key in FRED_REQ.keys():
        status[key] = {"source":"FRED", "series_id": FRED_REQ[key], "ok": False, "n": 0}
    if not FREDAPI:
        st.sidebar.warning("fredapi not installed — FRED fetching disabled. Install fredapi if you want FRED pulls.")

# -------------------------
# Fetch OFR datasets (repo and mmf)
# -------------------------
st.sidebar.markdown("OFR API: short-term funding monitor")

ofr_mmf = ofr_fetch_dataset("mmf")
mmf_map = {}
if ofr_mmf:
    mmf_map = parse_ofr_series_json(ofr_mmf)
    for k,v in mmf_map.items():
        series[f"ofr_{k}"] = v.rename(k)
    status["ofr_mmf"] = {"source":"OFR", "dataset":"mmf", "ok": True, "n": len(mmf_map)}
else:
    status["ofr_mmf"] = {"source":"OFR", "dataset":"mmf", "ok": False, "n": 0}

ofr_repo = ofr_fetch_dataset("repo")
repo_map = {}
if ofr_repo:
    repo_map = parse_ofr_series_json(ofr_repo)
    for k,v in repo_map.items():
        series[f"ofr_{k}"] = v.rename(k)
    status["ofr_repo"] = {"source":"OFR", "dataset":"repo", "ok": True, "n": len(repo_map)}
else:
    status["ofr_repo"] = {"source":"OFR", "dataset":"repo", "ok": False, "n": 0}

# Show status table
st.sidebar.markdown("**Load status**")
st.sidebar.table(pd.DataFrame(status).T.fillna("").astype(str))

# -------------------------
# Validate at least one series
# -------------------------
valid_series = {k:v for k,v in series.items() if v is not None and not v.empty}
if not valid_series:
    st.error("No series loaded. Provide a FRED API key in the sidebar or ensure OFR API reachable. As fallback, upload CSVs (I can add upload fields if you prefer).")
    st.stop()

# -------------------------
# Align time index (business days union)
# -------------------------
bidx = safe_union_index(valid_series)
for k in list(valid_series.keys()):
    valid_series[k] = valid_series[k].reindex(bidx)

df = pd.DataFrame(valid_series)

# -------------------------
# Derived anomalies (bps or units)
# -------------------------
anoms = pd.DataFrame(index=df.index)

if "sofr" in df.columns and "fedfunds" in df.columns:
    anoms["sofr_minus_fedfunds_bps"] = (df["sofr"] - df["fedfunds"]) * 10000

ofr_repo_keys = [c for c in df.columns if c.startswith("ofr_REPO") or c.startswith("ofr_repo") or c.startswith("ofr_REPO-") or c.startswith("ofr_REPO")]
if ofr_repo_keys:
    repo_key = ofr_repo_keys[0]
    anoms["repo_rate_bps"] = df[repo_key] * 100
    if "fedfunds" in df.columns:
        anoms["repo_minus_fedfunds_bps"] = (df[repo_key] - df["fedfunds"]) * 10000

if "fra_ois" in df.columns:
    anoms["fra_ois_bps"] = df["fra_ois"] * 10000
if "tedrate" in df.columns:
    anoms["ted_bps"] = df["tedrate"] * 100

if "ig_oas" in df.columns:
    anoms["ig_oas_bps"] = df["ig_oas"]
if "hy_oas" in df.columns:
    anoms["hy_oas_bps"] = df["hy_oas"]
if "hy_oas_bps" in anoms.columns and "ig_oas_bps" in anoms.columns:
    anoms["hy_minus_ig_bps"] = anoms["hy_oas_bps"] - anoms["ig_oas_bps"]

if "totalcp" in df.columns:
    anoms["totalcp"] = df["totalcp"]
if "cpn3m" in df.columns:
    anoms["cpn3m"] = df["cpn3m"] * 100

mmf_keys = [c for c in df.columns if c.startswith("ofr_MMF") or c.startswith("ofr_mmf") or "mmf" in c.lower()]
for c in mmf_keys:
    anoms[f"{c}"] = df[c]

if "rrp" in df.columns:
    anoms["rrp_outstanding"] = df["rrp"]

anoms = anoms.dropna(axis=1, how="all")
if anoms.empty:
    st.error("No anomaly series constructed from the available data. Try adding FRED key or enabling more datasets.")
    st.stop()

# -------------------------
# Rolling z-scores
# -------------------------
z = pd.DataFrame(index=anoms.index)
for col in anoms.columns:
    z[col] = anoms[col].rolling(window=roll_window, min_periods=max(5, int(roll_window/3))).apply(
        lambda x: (x[-1] - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) != 0 else np.nan)
    )

FUNDING_ANOMS = [c for c in ["sofr_minus_fedfunds_bps","repo_minus_fedfunds_bps","fra_ois_bps","ted_bps"] if c in z.columns]
CREDIT_ANOMS = [c for c in ["ig_oas_bps","hy_oas_bps","hy_minus_ig_bps"] if c in z.columns]
LIQUIDITY_ANOMS = [c for c in ["mmf_flows","totalcp","cpn3m","rrp_outstanding"] if c in z.columns or c in anoms.columns]


def latest_mean_abs_z(cols: List[str]) -> Optional[float]:
    present = [c for c in cols if c in z.columns]
    if not present:
        return np.nan
    vals = z[present].iloc[-1].abs().dropna()
    return float(vals.mean()) if not vals.empty else np.nan

fund_z = latest_mean_abs_z(FUNDING_ANOMS)
cred_z = latest_mean_abs_z(CREDIT_ANOMS)
liq_z = latest_mean_abs_z(LIQUIDITY_ANOMS)

def z_to_score(zv: Optional[float]) -> int:
    if pd.isna(zv):
        return 0
    return int(np.clip((zv / 3.0) * 100, 0, 100))

fund_score = z_to_score(fund_z)
cred_score = z_to_score(cred_z)
liq_score = z_to_score(liq_z)
combined_score = int(np.clip(0.4*fund_score + 0.4*cred_score + 0.2*liq_score, 0, 100))

# -------------------------
# UI
# -------------------------
st.title("Money Market Stress Dashboard (v7) — anomalies & stress")
st.markdown(f"**Horizon:** {horizon} — rolling z-window = {roll_window} business days. Sources: FRED & OFR API.")

st.subheader("Source status")
st.table(pd.DataFrame(status).T.fillna("").astype(str))

cols = st.columns(4)

def color_for_score(s:int)->str:
    if s<=40: return "green"
    if s<=70: return "orange"
    return "red"

def show_metric(col, label, val):
    col.markdown(f"**{label}**")
    col.metric("", f"{val}")
    col.markdown(f"<div style='height:8px;background:{color_for_score(val)};border-radius:6px'></div>", unsafe_allow_html=True)

show_metric(cols[0], "Funding Stress (0-100)", fund_score)
show_metric(cols[1], "Credit Stress (0-100)", cred_score)
show_metric(cols[2], "Liquidity Stress (0-100)", liq_score)
show_metric(cols[3], "Combined Stress (0-100)", combined_score)

st.markdown("---")

end = anoms.index.max()
start = end - pd.Timedelta(days=h_days)
anoms_recent = anoms.loc[start:end]


def plot_timeseries(df_plot: pd.DataFrame, title: str, height:int=300):
    if df_plot.empty:
        st.info("No data for chart.")
        return
    if PLOTLY:
        fig = go.Figure()
        for col in df_plot.columns:
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[col], name=col, mode='lines'))
        fig.update_layout(title=title, height=height, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(df_plot)

if FUNDING_ANOMS:
    st.subheader("Funding anomalies (bps)")
    df_f = anoms_recent[[c for c in FUNDING_ANOMS if c in anoms_recent.columns]]
    df_f.columns = [c.replace("_bps"," (bps)").replace("_"," ") for c in df_f.columns]
    plot_timeseries(df_f, f"Funding anomalies — last {horizon}", height=300)

if CREDIT_ANOMS:
    st.subheader("Credit anomalies (bps)")
    df_c = anoms_recent[[c for c in CREDIT_ANOMS if c in anoms_recent.columns]]
    df_c.columns = [c.replace("_bps"," (bps)").replace("_"," ") for c in df_c.columns]
    plot_timeseries(df_c, f"Credit anomalies — last {horizon}", height=300)

liq_present = [c for c in LIQUIDITY_ANOMS if c in anoms_recent.columns or c in anoms.columns]
if liq_present:
    st.subheader("Liquidity anomalies")
    df_l = anoms_recent[[c for c in liq_present if c in anoms_recent.columns]]
    df_l.columns = [c.replace("_bps"," (bps)").replace("_"," ") for c in df_l.columns]
    plot_timeseries(df_l, f"Liquidity anomalies — last {horizon}", height=350)

st.markdown("---")

st.subheader("Heatmap — recent z-scores")
heat_len = min(len(z), 120)
if heat_len >= 10:
    z_recent = z.tail(heat_len)
    rows = []
    for grp in (FUNDING_ANOMS, CREDIT_ANOMS, LIQUIDITY_ANOMS):
        rows.extend([c for c in grp if c in z_recent.columns])
    if rows:
        heat_df = z_recent[rows].T
        heat_df.index = [c.replace("_bps"," (bps)").replace("_"," ") for c in heat_df.index]
        if PLOTLY:
            fig = px.imshow(heat_df, aspect='auto', labels=dict(x='Date', y='Indicator', color='z-score'))
            fig.update_layout(height=360, margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(heat_df)
    else:
        st.info("Not enough z-score series for heatmap.")
else:
    st.info("Not enough history to build heatmap (need ~10 obs).")

st.markdown("---")
st.subheader("Alerts — recent |z| >= 2")
alerts = []
if not z.empty:
    latest = z.iloc[-1]
    for c in latest.index:
        v = latest[c]
        if pd.notna(v) and abs(v) >= 2.0:
            cat = "Funding" if c in FUNDING_ANOMS else "Credit" if c in CREDIT_ANOMS else "Liquidity"
            alerts.append((c, v, cat))
if alerts:
    for c,v,cat in alerts:
        st.error(f"⚠️ {c} — {cat} — z = {v:.2f}")
else:
    st.success("No recent anomalous z-scores (|z| >= 2).")

st.markdown("---")
st.subheader("Export data")
if st.button("Download anomalies CSV"):
    tmp = anoms.reset_index().rename(columns={"index":"date"})
    st.download_button("Download anomalies", tmp.to_csv(index=False).encode("utf-8"), file_name="anomalies_v7.csv", mime="text/csv")
if st.button("Download raw fetched series (CSV)"):
    tmp2 = df.reset_index().rename(columns={"index":"date"})
    st.download_button("Download raw", tmp2.to_csv(index=False).encode("utf-8"), file_name="raw_series_v7.csv", mime="text/csv")

st.caption("Notes: v7 uses FRED + OFR API (documented endpoints). If you want NY Fed Markets API transaction-level repo data added, I can wire that next.")
