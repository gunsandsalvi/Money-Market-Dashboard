"""
Streamlit dashboard: Real-time / near-real-time indicators of funding stress
Includes: SOFR, repo spreads, T-bill - OIS, FRA-OIS, CP/CD spreads, MMF flows, dealer leverage proxies,
CCP margin spikes, FX USD-basis and CDS (placeholders), and a composite risk score.

Design notes:
- The app will attempt to fetch data from FRED if the user provides a FRED API key.
- The app accepts CSV uploads for any of the indicators (recommended if you have a preferred feed).
- If data sources are unavailable the app falls back to synthetic sample data so the UI still works.
- Risk levels are computed using rolling z-scores (default window 60 business days) and mapped to Low/Medium/High.

How to run:
1. pip install streamlit pandas numpy plotly requests fredapi
2. streamlit run streamlit_money_market_dashboard.py

Optional: set your FRED API key in the sidebar or set environment variable FRED_API_KEY.
If you have Bloomberg/Refinitiv/Proprietary feeds, upload CSVs with 'date' and 'value' columns for each indicator.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import io

# Try to import fredapi if available
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except Exception:
    FRED_AVAILABLE = False

st.set_page_config(page_title="Money & Funding Market Stress Dashboard", layout="wide")

# ----------------------------- Helper functions -----------------------------

def fetch_fred_series(series_id, fred_client, start_date=None, end_date=None):
    """Fetch series via fredapi (user must provide key). Returns pandas Series with datetime index."""
    try:
        s = fred_client.get_series(series_id, observation_start=start_date, observation_end=end_date)
        s = pd.Series(s)
        s.index = pd.to_datetime(s.index)
        s.name = series_id
        return s
    except Exception as e:
        st.debug(f"FRED fetch failed for {series_id}: {e}")
        return None


def generate_sample_series(name, days=365, seed=0, level=0.01, vol=0.002):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=datetime.today(), periods=days)
    shocks = rng.normal(loc=0, scale=vol, size=len(dates)).cumsum()
    vals = level + shocks
    s = pd.Series(vals, index=dates, name=name)
    return s


def rolling_zscore(series, window=60):
    return (series - series.rolling(window).mean()) / (series.rolling(window).std(ddof=0))


def risk_level_from_z(z):
    # z may be positive or negative; for some indicators (like flows) negative is bad.
    # We'll take absolute z for symmetric indicators, but some indicators should be signed.
    if pd.isna(z):
        return "N/A"
    if np.abs(z) < 1:
        return "Low"
    elif np.abs(z) < 2:
        return "Medium"
    else:
        return "High"


def color_for_level(level):
    return {"Low": "green", "Medium": "orange", "High": "red", "N/A": "gray"}.get(level, "gray")


# ----------------------------- Sidebar: Data inputs -----------------------------
st.sidebar.title("Data & Settings")
with st.sidebar.expander("Data sources / inputs", expanded=True):
    st.markdown("Provide a FRED API key (optional) to fetch many default series automatically.\nAlternatively upload CSVs for indicators (must include 'date' and 'value' columns).")
    fred_key = st.text_input("FRED API key (optional)", value=os.environ.get('FRED_API_KEY',''))
    use_sample = st.checkbox("Use sample data if real data unavailable", value=True)

    st.markdown("---")
    st.markdown("**Upload CSVs (optional)**")
    uploaded_sofr = st.file_uploader("SOFR CSV (date,value)", type=["csv"], key='sofr')
    uploaded_repo = st.file_uploader("Repo spread CSV (date,value)", type=["csv"], key='repo')
    uploaded_tbill_oisspread = st.file_uploader("T-bill - OIS CSV (date,value)", type=["csv"], key='tbill')
    uploaded_fraois = st.file_uploader("FRA-OIS CSV (date,value)", type=["csv"], key='fra')
    uploaded_cp = st.file_uploader("CP spread CSV (date,value)", type=["csv"], key='cp')
    uploaded_mmf = st.file_uploader("MMF flows CSV (date,value)", type=["csv"], key='mmf')
    uploaded_ccp_margin = st.file_uploader("CCP margin index CSV (date,value)", type=["csv"], key='ccp')

with st.sidebar.expander("Risk scoring settings", expanded=False):
    rolling_window = st.number_input("Rolling window for z-score (days)", min_value=10, max_value=252, value=60)
    weights = {}
    st.markdown("Weights for composite risk score (must sum to 1). Default equal-weight.")
    weights['sofr'] = st.slider("SOFR weight", 0.0, 1.0, 0.15)
    weights['repo'] = st.slider("Repo spread weight", 0.0, 1.0, 0.15)
    weights['tbill'] = st.slider("T-bill-OIS weight", 0.0, 1.0, 0.15)
    weights['fra'] = st.slider("FRA-OIS weight", 0.0, 1.0, 0.15)
    weights['cp'] = st.slider("CP/CD weight", 0.0, 1.0, 0.15)
    weights['mmf'] = st.slider("MMF flows weight", 0.0, 1.0, 0.15)
    weights['ccp'] = st.slider("CCP margin weight", 0.0, 1.0, 0.1)

# Normalize weights
wsum = sum(weights.values())
if wsum <= 0:
    # fallback to equal weights
    n = len(weights)
    for k in weights:
        weights[k] = 1.0/n
else:
    for k in weights:
        weights[k] = weights[k]/wsum

st.sidebar.markdown("---")
if not FRED_AVAILABLE and fred_key:
    st.sidebar.warning("fredapi not installed in the runtime. Install fredapi or rely on CSV uploads.")

# ----------------------------- Data loading -----------------------------
st.title("Money & Funding Market Stress Dashboard")
st.markdown("Practical, real-time indicators for funding stress. Provide FRED API key or upload CSVs to populate charts.")

# Prepare a dictionary to hold series
series = {}

# Helper to parse uploaded CSV

def parse_uploaded(file):
    try:
        df = pd.read_csv(file)
        df.columns = [c.strip().lower() for c in df.columns]
        if 'date' not in df.columns or 'value' not in df.columns:
            st.warning('Uploaded CSV must have columns `date` and `value`.')
            return None
        s = pd.Series(df['value'].values, index=pd.to_datetime(df['date']), name='uploaded')
        s = s.sort_index()
        return s
    except Exception as e:
        st.warning(f'Could not parse uploaded CSV: {e}')
        return None

# 1) SOFR
if uploaded_sofr is not None:
    s = parse_uploaded(uploaded_sofr)
    if s is not None:
        series['sofr'] = s
else:
    # Try FRED
    if fred_key and FRED_AVAILABLE:
        fred = Fred(api_key=fred_key)
        s = fetch_fred_series('SOFR', fred)
        if s is not None:
            series['sofr'] = s
    # fallback
    if 'sofr' not in series and use_sample:
        series['sofr'] = generate_sample_series('SOFR', days=500, level=0.02, vol=0.001)

# 2) Repo spread (GC repo - SOFR or repo spread series)
if uploaded_repo is not None:
    s = parse_uploaded(uploaded_repo)
    if s is not None:
        series['repo_spread'] = s
else:
    # attempt to fetch FRED GC repo proxy (notionally 'RRPONTSYD' or 'BGCR' may not be present)
    if fred_key and FRED_AVAILABLE:
        fred = Fred(api_key=fred_key)
        candidates = ['BGNREPO', 'RRPONTSYD']  # placeholders; actual series names may differ
        fetched = None
        for cid in candidates:
            fetched = fetch_fred_series(cid, fred)
            if fetched is not None:
                series['repo_spread'] = fetched
                break
    if 'repo_spread' not in series and use_sample:
        # construct repo_spread as small spread over SOFR plus noise
        base = series['sofr'].reindex(series['sofr'].index).fillna(method='ffill')
        noise = np.random.normal(0, 0.0003, size=len(base))
        repo = base + 0.0005 + noise
        series['repo_spread'] = pd.Series(repo.values, index=base.index, name='repo_spread')

# 3) T-bill - OIS
if uploaded_tbill_oisspread is not None:
    s = parse_uploaded(uploaded_tbill_oisspread)
    if s is not None:
        series['tbill_oisspread'] = s
else:
    if fred_key and FRED_AVAILABLE:
        fred = Fred(api_key=fred_key)
        # as a pragmatic approach, use 1M Treasury bill surrogate 'DTB1' minus FEDFUNDS (not exact OIS)
        try:
            dtb1 = fetch_fred_series('DTB1', fred)
            fedf = fetch_fred_series('FEDFUNDS', fred)
            if dtb1 is not None and fedf is not None:
                tb = dtb1.reindex(dtb1.index).astype(float)
                ff = fedf.reindex(tb.index).interpolate().astype(float)
                series['tbill_oisspread'] = tb - ff
        except Exception:
            pass
    if 'tbill_oisspread' not in series and use_sample:
        series['tbill_oisspread'] = generate_sample_series('tbill_oisspread', days=500, level=0.0, vol=0.0004)

# 4) FRA-OIS
if uploaded_fraois is not None:
    s = parse_uploaded(uploaded_fraois)
    if s is not None:
        series['fra_ois'] = s
else:
    if 'fra_ois' not in series and use_sample:
        series['fra_ois'] = generate_sample_series('fra_ois', days=500, level=0.0005, vol=0.0007)

# 5) CP spread
if uploaded_cp is not None:
    s = parse_uploaded(uploaded_cp)
    if s is not None:
        series['cp_spread'] = s
else:
    if 'cp_spread' not in series and use_sample:
        series['cp_spread'] = generate_sample_series('cp_spread', days=500, level=0.001, vol=0.0015)

# 6) MMF flows
if uploaded_mmf is not None:
    s = parse_uploaded(uploaded_mmf)
    if s is not None:
        series['mmf_flows'] = s
else:
    if 'mmf_flows' not in series and use_sample:
        # flows: positive = inflows to govt MMFs (which may be flight to safety); make spikes occasionally
        s = generate_sample_series('mmf_flows', days=500, level=0.0, vol=1e6)
        # add occasional negative spikes (redemptions)
        rng = np.random.default_rng(42)
        spikes = rng.choice(len(s), size=8, replace=False)
        s.iloc[spikes] += -5e6
        series['mmf_flows'] = s

# 7) CCP margin index
if uploaded_ccp_margin is not None:
    s = parse_uploaded(uploaded_ccp_margin)
    if s is not None:
        series['ccp_margin'] = s
else:
    if 'ccp_margin' not in series and use_sample:
        series['ccp_margin'] = generate_sample_series('ccp_margin', days=500, level=1.0, vol=0.05)

# Align series to a common date index (business days over last N days)
all_idx = pd.bdate_range(end=datetime.today(), periods=500)
for k, s in series.items():
    series[k] = s.reindex(all_idx).interpolate().fillna(method='bfill').fillna(method='ffill')

# ----------------------------- Compute indicator z-scores & levels -----------------------------
indicator_df = pd.DataFrame(index=all_idx)
for k, s in series.items():
    indicator_df[k] = s

z_scores = indicator_df.rolling(window=rolling_window, min_periods=rolling_window).apply(
    lambda x: (x[-1] - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) != 0 else np.nan))

# For mmf flows we treat negative large outflows as bad -> invert sign so positive z = more risk
if 'mmf_flows' in z_scores.columns:
    # invert: big negative flows -> positive risk
    z_scores['mmf_flows_signed'] = -z_scores['mmf_flows']

# Create a summary table for current values and risk
latest = indicator_df.iloc[-1]
latest_z = z_scores.iloc[-1]

summary = []
for k in indicator_df.columns:
    val = latest.get(k, np.nan)
    z = latest_z.get(k, np.nan)
    if k == 'mmf_flows':
        z = -z  # invert sign for flows
    lvl = risk_level_from_z(z)
    summary.append({'indicator': k, 'value': val, 'zscore': float(np.nan_to_num(z, nan=0.0)), 'level': lvl})

summary_df = pd.DataFrame(summary).set_index('indicator')

# Composite risk score: weighted average of absolute z-scores (or signed where appropriate)
composite_components = []
for key in ['sofr','repo_spread','tbill_oisspread','fra_ois','cp_spread','mmf_flows','ccp_margin']:
    if key in z_scores.columns or (key=='mmf_flows' and 'mmf_flows_signed' in z_scores.columns):
        if key == 'mmf_flows':
            z = -latest_z.get('mmf_flows', 0)
        else:
            z = latest_z.get(key, 0)
        composite_components.append(weights.get(key, 0) * min(3.0, abs(z)))
composite_score = sum(composite_components)
# map composite to 0-100 scale
composite_percent = int(100 * (composite_score / 3.0 / 1.0))
composite_percent = max(0, min(100, composite_percent))

# ----------------------------- Layout: Top metrics -----------------------------
col1, col2, col3, col4 = st.columns([1,1,1,1])
with col1:
    st.metric(label="Composite Funding Stress (0-100)", value=f"{composite_percent}",
              delta=None)
with col2:
    # show latest SOFR
    if 'sofr' in latest:
        st.metric(label="SOFR (latest)", value=f"{latest['sofr']*100:.2f}%", delta=f"{latest_z.get('sofr',0):.2f} z")
    else:
        st.metric(label="SOFR (latest)", value="n/a")
with col3:
    if 'repo_spread' in latest:
        st.metric(label="Repo spread (proxy)", value=f"{latest['repo_spread']*100:.2f} bps",
                  delta=f"{latest_z.get('repo_spread',0):.2f} z")
    else:
        st.metric(label="Repo spread (proxy)", value="n/a")
with col4:
    if 'tbill_oisspread' in latest:
        st.metric(label="T-bill - OIS (proxy)", value=f"{latest['tbill_oisspread']*100:.2f} bps",
                  delta=f"{latest_z.get('tbill_oisspread',0):.2f} z")
    else:
        st.metric(label="T-bill - OIS (proxy)", value="n/a")

st.markdown("---")

# ----------------------------- Time series charts -----------------------------
st.header("Indicator Time Series & Z-scores")
# For each indicator show chart and small risk chip
chart_cols = st.columns(2)
left_inds = ['sofr','repo_spread','tbill_oisspread','fra_ois']
right_inds = ['cp_spread','mmf_flows','ccp_margin']

with chart_cols[0]:
    for k in left_inds:
        if k in indicator_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=indicator_df.index, y=indicator_df[k], name=k))
            if k in z_scores.columns:
                fig.add_trace(go.Scatter(x=z_scores.index, y=z_scores[k], name=f"{k} z-score", yaxis="y2", opacity=0.6))
                fig.update_layout(yaxis2=dict(overlaying='y', side='right', title='z-score'))
            fig.update_layout(height=300, margin=dict(l=10,r=10,t=30,b=20), title=k)
            st.plotly_chart(fig, use_container_width=True)

with chart_cols[1]:
    for k in right_inds:
        if k in indicator_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=indicator_df.index, y=indicator_df[k], name=k))
            if k == 'mmf_flows' and 'mmf_flows_signed' in z_scores.columns:
                zs = z_scores['mmf_flows_signed']
                fig.add_trace(go.Scatter(x=zs.index, y=zs, name='mmf_flows z-score', yaxis='y2', opacity=0.6))
                fig.update_layout(yaxis2=dict(overlaying='y', side='right', title='z-score'))
            elif k in z_scores.columns:
                fig.add_trace(go.Scatter(x=z_scores.index, y=z_scores[k], name=f"{k} z-score", yaxis='y2', opacity=0.6))
                fig.update_layout(yaxis2=dict(overlaying='y', side='right', title='z-score'))
            fig.update_layout(height=300, margin=dict(l=10,r=10,t=30,b=20), title=k)
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ----------------------------- Summary & Heatmap -----------------------------
st.header("Summary & Risk Heatmap")

# Show summary table
summary_display = summary_df.copy()
summary_display['level_color'] = summary_display['level'].map(lambda x: color_for_level(x))

# Use markdown to show colored chips
cols = st.columns([2,1,1,1])
cols[0].write("**Indicator**")
cols[1].write("**Latest**")
cols[2].write("**z-score**")
cols[3].write("**Level**")
for idx, row in summary_display.iterrows():
    cols = st.columns([2,1,1,1])
    cols[0].write(idx)
    val = row['value']
    if pd.isna(val):
        cols[1].write("n/a")
    else:
        # format as percent when small (rates)
        if abs(val) < 0.1:
            cols[1].write(f"{val*100:.2f}%")
        else:
            cols[1].write(f"{val:,.0f}")
    cols[2].write(f"{row['zscore']:.2f}")
    cols[3].markdown(f"<span style='color: white; background-color: {row['level_color']}; padding:4px 8px; border-radius:6px'>{row['level']}</span>", unsafe_allow_html=True)

# Heatmap of recent z-scores
st.subheader("Recent z-score heatmap (last 120 business days)")
heat_df = z_scores.tail(120).copy()
# drop mmf flows and add signed version if present
if 'mmf_flows' in heat_df.columns:
    heat_df['mmf_flows'] = -heat_df['mmf_flows']
# limit to relevant columns and fillna
heat_plot = heat_df[list(series.keys())].T
fig = px.imshow(heat_plot, aspect='auto', labels=dict(x='Date', y='Indicator', color='z-score'))
fig.update_layout(height=300)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ----------------------------- Alerts panel -----------------------------
st.header("Alerts & Signals")
alerts = []
for idx, row in summary_df.iterrows():
    if row['level'] == 'High':
        alerts.append(f"{idx} shows HIGH stress (z={row['zscore']:.2f})")

if composite_percent > 60:
    alerts.insert(0, f"Composite funding stress elevated: {composite_percent}/100")

if alerts:
    for a in alerts:
        st.error(a)
else:
    st.success("No immediate HIGH stress flags from the monitored indicators.")

st.markdown("---")

# ----------------------------- Downloads & Export -----------------------------
st.header("Export & Next steps")
st.markdown("You can download the underlying normalized z-score table or the composite risk series for integration into other systems.")
if st.button("Download latest indicator CSV"):
    csv = indicator_df.reset_index().rename(columns={'index':'date'}).to_csv(index=False).encode('utf-8')
    st.download_button(label='Download CSV', data=csv, file_name='indicators.csv', mime='text/csv')

st.markdown("---")
st.write("**Notes & further improvements:**")
st.write("1) Connect live proprietary feeds (Bloomberg/Refinitiv) by uploading CSVs or implementing their python clients.\n2) Replace proxy FRED series with precise series IDs for GC repo, tri-party repo, Fed OIS, and FRA-OIS.\n3) Alerting: integrate with e-mail/Slack when composite exceeds threshold.\n4) Add dealer-balance-sheet indicators by ingesting weekly primary dealer data and call reports.")

# End of app
