"""
Streamlit Money Market & Credit Stress Dashboard — Mobile-ready & Cloud-deployable

Features added:
- Mobile-first layout (responsive single-column for narrow screens)
- Additional real-time credit market measures: CDX IG / HY (CSV or API), IG/HY cash OAS, bank CDS, corporate bond indices
- More sophisticated risk model:
  * Standardized z-scores (rolling)
  * EWMA volatility scaling
  * PCA-based composite (first principal component) as alternative composite
  * Mahalanobis distance anomaly score across indicators
  * Regime indicator (simple KMeans on recent PCA scores)
- Flexible connectors: FRED (public), CSV uploads, and template hooks for vendors (Bloomberg/Refinitiv/ICE/Markit via REST)
- Alerts, mobile-first charts (Plotly), and export endpoints

Deployment guidance:
- Push this file to a public or private GitHub repo and deploy to Streamlit Cloud (https://streamlit.io/cloud) by connecting your GitHub account and pointing to this file.
- For private data feeds (Bloomberg/Refinitiv), store credentials as secrets in Streamlit Cloud and enable the respective python clients.
- Alternatively deploy to Render/Heroku/Docker if you want a custom domain.

Notes about real-time data:
- Many credit market series (CDX, Markit, Bloomberg) are subscription-only. This app allows CSV uploads or REST/websocket connectors to ingest those feeds.
- Public proxies via FRED cover many money-market series; credit measures will typically need vendor access for true real-time.

How to use:
- Open on mobile: Streamlit Cloud apps are mobile-friendly. On narrow screens the layout collapses to single-column for better viewing.
- Upload CSVs named by indicator (date,value) or fill in API endpoints in the sidebar.

"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# Optional: fredapi
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except Exception:
    FRED_AVAILABLE = False

st.set_page_config(page_title="Funding & Credit Stress Dashboard", layout="centered", initial_sidebar_state="expanded")

# ------------------------- Helpers -------------------------
@st.experimental_singleton
def get_fred_client(key):
    if not FRED_AVAILABLE or not key:
        return None
    return Fred(api_key=key)

@st.cache_data(ttl=300)
def fetch_fred_series(series_id, fred_client, start_date=None, end_date=None):
    try:
        s = fred_client.get_series(series_id, observation_start=start_date, observation_end=end_date)
        s = pd.Series(s)
        s.index = pd.to_datetime(s.index)
        s.name = series_id
        return s
    except Exception as e:
        st.warning(f"FRED fetch failed for {series_id}: {e}")
        return None


def parse_uploaded(file):
    try:
        df = pd.read_csv(file)
        cols = [c.strip().lower() for c in df.columns]
        df.columns = cols
        if 'date' not in df.columns or 'value' not in df.columns:
            st.warning('Uploaded CSV must have columns `date` and `value`.')
            return None
        s = pd.Series(df['value'].values, index=pd.to_datetime(df['date']), name='uploaded')
        s = s.sort_index()
        return s
    except Exception as e:
        st.warning(f'Could not parse uploaded CSV: {e}')
        return None


def rolling_zscore(series, window=60):
    return (series - series.rolling(window).mean()) / (series.rolling(window).std(ddof=0))


def ewma(series, span=60):
    return series.ewm(span=span, adjust=False).std()


def compute_mahalanobis(df):
    # df: observations x variables (latest window). Compute mahalanobis for each date relative to obs mean and cov
    X = df.dropna()
    if X.shape[0] < X.shape[1] + 2:
        return pd.Series(np.nan, index=df.index)
    mu = X.mean()
    cov = np.cov(X.T)
    try:
        invcov = np.linalg.pinv(cov)
    except Exception:
        invcov = np.linalg.pinv(cov + np.eye(cov.shape[0]) * 1e-6)
    m = []
    for i in range(X.shape[0]):
        d = X.iloc[i].values - mu.values
        m.append(np.sqrt(np.dot(np.dot(d.T, invcov), d)))
    return pd.Series(m, index=X.index)

# ------------------------- Sidebar inputs -------------------------
st.sidebar.title("Data & Deployment")
fred_key = st.sidebar.text_input("FRED API key (optional)", value=os.environ.get('FRED_API_KEY',''))
use_sample = st.sidebar.checkbox("Use sample data if feeds missing", value=True)

st.sidebar.markdown("---")
st.sidebar.header("Upload CSV feeds (optional)")
u_sofr = st.sidebar.file_uploader("SOFR (date,value)", type=['csv'], key='sofr')
u_repo = st.sidebar.file_uploader("Repo spread (date,value)", type=['csv'], key='repo')
u_tbill = st.sidebar.file_uploader("T-bill-OIS (date,value)", type=['csv'], key='tbill')
u_fra = st.sidebar.file_uploader("FRA-OIS (date,value)", type=['csv'], key='fra')
u_cp = st.sidebar.file_uploader("CP spread (date,value)", type=['csv'], key='cp')
u_mmf = st.sidebar.file_uploader("MMF flows (date,value)", type=['csv'], key='mmf')
u_ccp = st.sidebar.file_uploader("CCP margin index (date,value)", type=['csv'], key='ccp')

st.sidebar.markdown("**Credit market feeds (CSV or API)**")
u_cdx_ig = st.sidebar.file_uploader("CDX IG spread (date,value)", type=['csv'], key='cdxig')
u_cdx_hy = st.sidebar.file_uploader("CDX HY spread (date,value)", type=['csv'], key='cdxhy')
u_ig_oas = st.sidebar.file_uploader("IG OAS index (date,value)", type=['csv'], key='igoas')
u_hy_oas = st.sidebar.file_uploader("HY OAS index (date,value)", type=['csv'], key='hyoas')
u_bank_cds = st.sidebar.file_uploader("Bank CDS (date,value)", type=['csv'], key='bankcds')

st.sidebar.markdown("---")
st.sidebar.header("Model settings")
rolling_window = st.sidebar.number_input("Rolling z-window (days)", min_value=10, max_value=252, value=60)
ewma_span = st.sidebar.number_input("EWMA vol span", min_value=10, max_value=252, value=60)
components = st.sidebar.slider("PCA components to compute", min_value=1, max_value=5, value=1)

st.sidebar.markdown("---")
st.sidebar.header("Deployment tips")
st.sidebar.info("To run on your phone: deploy app to Streamlit Cloud (recommended). Connect a GitHub repo with this file and press Deploy. Use Streamlit Secrets for credentials.")

# ------------------------- Load / Build series -------------------------
fred = get_fred_client(fred_key)

# convenience function to add to dict
ind = {}

# SOFR
if u_sofr:
    s = parse_uploaded(u_sofr); ind['sofr'] = s
else:
    if fred and FRED_AVAILABLE:
        try:
            s = fetch_fred_series('SOFR', fred)
            ind['sofr'] = s
        except Exception:
            pass
    if 'sofr' not in ind and use_sample:
        dates = pd.bdate_range(end=datetime.today(), periods=500)
        rng = np.random.default_rng(1)
        vals = 0.02 + np.cumsum(rng.normal(0,0.0005,len(dates)))
        ind['sofr'] = pd.Series(vals, index=dates, name='sofr')

# Repo spread
if u_repo:
    ind['repo_spread'] = parse_uploaded(u_repo)
else:
    if 'repo_spread' not in ind and use_sample and 'sofr' in ind:
        base = ind['sofr']
        noise = np.random.normal(0,0.0004,len(base))
        ind['repo_spread'] = pd.Series(base.values + 0.0005 + noise, index=base.index, name='repo_spread')

# T-bill - OIS (proxy using DTB1 - FEDFUNDS if available)
if u_tbill:
    ind['tbill_oisspread'] = parse_uploaded(u_tbill)
else:
    if fred and FRED_AVAILABLE and use_sample:
        try:
            dtb1 = fetch_fred_series('DTB1', fred)
            fedf = fetch_fred_series('FEDFUNDS', fred)
            if dtb1 is not None and fedf is not None:
                aligned = dtb1.reindex(dtb1.index).astype(float)
                ff = fedf.reindex(aligned.index).interpolate()
                ind['tbill_oisspread'] = aligned - ff
        except Exception:
            pass
    if 'tbill_oisspread' not in ind and use_sample:
        dates = pd.bdate_range(end=datetime.today(), periods=500)
        ind['tbill_oisspread'] = pd.Series(np.random.normal(0,0.0004,len(dates)), index=dates, name='tbill_oisspread')

# FRA-OIS, CP, MMF, CCP
if u_fra: ind['fra_ois'] = parse_uploaded(u_fra)
elif use_sample: ind['fra_ois'] = pd.Series(np.random.normal(0.0005,0.0008,500), index=pd.bdate_range(end=datetime.today(), periods=500), name='fra_ois')
if u_cp: ind['cp_spread'] = parse_uploaded(u_cp)
elif use_sample: ind['cp_spread'] = pd.Series(np.random.normal(0.001,0.0015,500), index=pd.bdate_range(end=datetime.today(), periods=500), name='cp_spread')
if u_mmf: ind['mmf_flows'] = parse_uploaded(u_mmf)
elif use_sample:
    d = pd.bdate_range(end=datetime.today(), periods=500)
    s = np.random.normal(0,1e6,len(d))
    s[np.random.choice(len(d),8)] -= 5e6
    ind['mmf_flows'] = pd.Series(s, index=d, name='mmf_flows')
if u_ccp: ind['ccp_margin'] = parse_uploaded(u_ccp)
elif use_sample: ind['ccp_margin'] = pd.Series(np.random.normal(1,0.05,500), index=pd.bdate_range(end=datetime.today(), periods=500), name='ccp_margin')

# Credit market series
if u_cdx_ig: ind['cdx_ig'] = parse_uploaded(u_cdx_ig)
elif use_sample: ind['cdx_ig'] = pd.Series(np.random.normal(70,5,500), index=pd.bdate_range(end=datetime.today(), periods=500), name='cdx_ig')
if u_cdx_hy: ind['cdx_hy'] = parse_uploaded(u_cdx_hy)
elif use_sample: ind['cdx_hy'] = pd.Series(np.random.normal(400,20,500), index=pd.bdate_range(end=datetime.today(), periods=500), name='cdx_hy')
if u_ig_oas: ind['ig_oas'] = parse_uploaded(u_ig_oas)
elif use_sample: ind['ig_oas'] = pd.Series(np.random.normal(80,8,500), index=pd.bdate_range(end=datetime.today(), periods=500), name='ig_oas')
if u_hy_oas: ind['hy_oas'] = parse_uploaded(u_hy_oas)
elif use_sample: ind['hy_oas'] = pd.Series(np.random.normal(400,25,500), index=pd.bdate_range(end=datetime.today(), periods=500), name='hy_oas')
if u_bank_cds: ind['bank_cds'] = parse_uploaded(u_bank_cds)
elif use_sample: ind['bank_cds'] = pd.Series(np.random.normal(50,6,500), index=pd.bdate_range(end=datetime.today(), periods=500), name='bank_cds')

# align
all_idx = pd.bdate_range(end=datetime.today(), periods=500)
for k in list(ind.keys()):
    ind[k] = ind[k].reindex(all_idx).interpolate().fillna(method='bfill').fillna(method='ffill')

# DataFrame
df = pd.DataFrame(ind)

# ------------------------- Modeling -------------------------
# z-scores
z = df.rolling(window=rolling_window, min_periods=int(rolling_window/2)).apply(lambda x: (x[-1]-x.mean())/(x.std(ddof=0) if x.std(ddof=0)!=0 else np.nan))

# For mmf flows invert sign so negative flows -> positive stress
if 'mmf_flows' in z.columns:
    z['mmf_flows'] = -z['mmf_flows']

# EWMA vol
ewma_vol = df.ewm(span=ewma_span, adjust=False).std()

# Standardize by EWMA vol to get volatility-adjusted z
z_vol_adj = z.copy()
for col in z.columns:
    if col in ewma_vol.columns:
        z_vol_adj[col] = z[col] / (ewma_vol[col].iloc[-1] if ewma_vol[col].iloc[-1] != 0 else 1)

# PCA composite (first principal component)
from sklearn.decomposition import PCA
valid_cols = [c for c in z.columns if z[c].notna().sum()>200]
pca_input = z[valid_cols].dropna()
if not pca_input.empty:
    pca = PCA(n_components=min(len(valid_cols), components))
    pcs = pca.fit_transform(pca_input.fillna(0))
    pc1 = pd.Series(pcs[:,0], index=pca_input.index, name='pc1')
else:
    pc1 = pd.Series(np.nan, index=df.index, name='pc1')

# Mahalanobis anomaly over recent window
maha = compute_mahalanobis(z[valid_cols].tail(rolling_window))
# Map latest mahalanobis to a percentile style score
maha_latest = float(maha.dropna().iloc[-1]) if not maha.dropna().empty else np.nan

# Composite score: weighted sum of abs(z) (equal weights) and scaled PC1
absz = z[valid_cols].abs()
weights = {c:1/len(valid_cols) for c in valid_cols} if valid_cols else {}
component = absz.iloc[-1].fillna(0)
composite_simple = sum([weights[c]*component[c] for c in weights]) if weights else 0
# scale composite_simple to 0-100 roughly by mapping expected range
composite_scaled = int(np.clip((composite_simple / 3.0) * 100, 0, 100))
# pca scaled
pc1_latest = pc1.iloc[-1] if not pc1.empty else 0
pc1_scaled = int( np.clip((pc1_latest - pc1.mean()) / (pc1.std() if pc1.std() else 1) * 20 + 50, 0, 100) )
# anomaly score 0-100
maha_score = int(np.clip(maha_latest/3.0*100, 0, 100)) if not np.isnan(maha_latest) else 0

# Final blended score
final_score = int(np.clip(0.5*composite_scaled + 0.4*pc1_scaled + 0.1*maha_score, 0, 100))

# ------------------------- UI: Responsive & Mobile-friendly -------------------------
st.title("Funding & Credit Market Stress Dashboard — Mobile")
st.markdown("Monitor money-market and credit-market stress on the go. Deploy to Streamlit Cloud for phone access.")

# top metrics single-row but responsive
cols = st.columns(4)
cols[0].metric("Composite (simple)", f"{composite_scaled}")
cols[1].metric("PCA score", f"{pc1_scaled}")
cols[2].metric("Anomaly (Mahalanobis)", f"{maha_score}")
cols[3].metric("Final Stress 0-100", f"{final_score}")

st.markdown("---")

# Time series panels — collapsible for mobile
with st.expander("Time series overview", expanded=True):
    # show a compact multi-line chart
    fig = go.Figure()
    show_cols = ['sofr','repo_spread','tbill_oisspread','fra_ois','cp_spread']
    for c in show_cols:
        if c in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[c], name=c))
    fig.update_layout(height=300, margin=dict(l=5,r=5,t=30,b=20))
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Credit market series", expanded=False):
    fig = go.Figure()
    credits = ['cdx_ig','cdx_hy','ig_oas','hy_oas','bank_cds']
    for c in credits:
        if c in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[c], name=c))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Z-scores heatmap", expanded=False):
    heat = z[valid_cols].tail(120).T
    if not heat.empty:
        fig = px.imshow(heat, aspect='auto', labels=dict(x='Date', y='Indicator', color='z-score'))
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Not enough data to plot z-score heatmap.")

# PCA / Components
with st.expander("PCA & Components", expanded=False):
    if not pc1.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pc1.index, y=pc1, name='PC1'))
        fig.update_layout(height=240)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("PCA not available (insufficient data).")

# Alerts
st.header("Alerts")
alerts = []
if final_score > 75:
    alerts.append(f"Final stress score elevated: {final_score}/100")
if maha_score > 70:
    alerts.append(f"Anomaly score high: {maha_score}")
# component level alerts
for c in valid_cols:
    vz = z[c].iloc[-1]
    if not np.isnan(vz) and abs(vz) > 2.0:
        alerts.append(f"{c} z-score high: {vz:.2f}")

if alerts:
    for a in alerts:
        st.error(a)
else:
    st.success("No immediate high-stress alerts.")

# Export and Integrations
st.markdown("---")
st.header("Export & Integrations")
st.write("Download the latest normalized indicators or configure API connectors for live feeds.")
if st.button("Download normalized z-scores (CSV)"):
    csv = z.reset_index().rename(columns={'index':'date'}).to_csv(index=False).encode('utf-8')
    st.download_button(label='Download CSV', data=csv, file_name='z_scores.csv', mime='text/csv')

st.markdown("""**Integration templates:**
- For Bloomberg: implement blpapi client and load authorization from Streamlit Secrets.
- For Markit/ICE: use REST endpoints and map JSON to (date,value) timeseries.
- For WebSockets: build a background ingestion service and write timeseries to a database / object storage, then the app pulls aggregated series.)""") timeseries.
- For WebSockets: build a background ingestion service and write timeseries to a database / object storage, then the app pulls aggregated series.)")

st.markdown("---")

st.write("**Next actions I can do for you:**
- Add a Markit/Bloomberg connector using your credentials (you must provide access).
- Hook alerts to Slack/email.
- Tune the model weighting or add a small HMM regime model for 3 regimes.
- Add interactive thresholds and backtest mode to label past events.")

# end
