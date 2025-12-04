import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import keras
except ImportError:
    from tensorflow import keras

# ---------- Configuration ----------
DATA_PATH = "Test.xlsx"
MODEL_PATH = r"C:\Users\W64295\OneDrive - Kimberly-Clark\2025\DS\categoryShareModel.mft.kcp.keras"

COLORS = {'primary': '#667eea', 'success': '#2ecc71', 'warning': '#f39c12', 
          'danger': '#e74c3c', 'baseline': '#3498db', 'updated': '#2ecc71'}

def fmt_pct(x): return f"{x:.2f}%"
def fmt_currency(x): return f"₹{x:,.2f}"

@st.cache_data
def load_data(path):
    df = pd.read_excel(path, engine="openpyxl")
    if df.shape[1] < 2: raise ValueError("Excel must have at least two columns")
    
    row_label_col = df.columns[0]
    feature_cols = [c for c in df.columns[1:] if pd.notna(c)]
    NON_EDITABLE_ALWAYS = {"RestWeightedPr", "AllWeightedPr"}
    labels = df[row_label_col].astype(str).str.strip().str.lower()
    
    price_mask = labels.eq("current price") | labels.str.contains("current")
    if not price_mask.any(): raise ValueError("'Current Price' row not found")
    baseline_prices = pd.to_numeric(df.loc[price_mask].iloc[0][feature_cols], errors="coerce").fillna(0.0)
    baseline_prices.index = feature_cols
    
    share_mask = labels.eq("untsshr") | labels.str.contains("untsshr")
    if share_mask.any():
        shares_all = pd.to_numeric(df.loc[share_mask].iloc[0][feature_cols], errors="coerce").fillna(0.0)
        baseline_shares = shares_all[[c for c in feature_cols if c not in NON_EDITABLE_ALWAYS]]
    else:
        baseline_shares = pd.Series(0.0, index=[c for c in feature_cols if c not in NON_EDITABLE_ALWAYS])
    
    total_baseline_share = float(baseline_shares.sum())
    if len(baseline_shares) and baseline_shares.max() <= 1: total_baseline_share *= 100.0
    
    kcc_mask = labels.eq("kcc")
    if not kcc_mask.any(): raise ValueError("KCC flag row not found")
    raw_flags = df.loc[kcc_mask].iloc[0][feature_cols].astype(str).str.strip().str.upper()
    editable_cols = [c for c in feature_cols if raw_flags.get(c, "") == "Y" and c not in NON_EDITABLE_ALWAYS]
    
    return df, feature_cols, NON_EDITABLE_ALWAYS, baseline_prices, baseline_shares, total_baseline_share, editable_cols

@st.cache_resource
def load_model(path, unsafe=False):
    if not os.path.exists(path): raise FileNotFoundError(f"Model not found: {path}")
    if unsafe:
        try: keras.config.enable_unsafe_deserialization()
        except: pass
    return keras.models.load_model(path, safe_mode=False)

# ---------- Page Setup ----------
st.set_page_config(page_title="KCP Share Simulator", layout="wide")

st.markdown("""
<style>
    * { font-family: 'Inter', sans-serif; }
    .main-header {
        text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 42px; font-weight: 800; margin-bottom: 10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 25px 20px; border-radius: 12px;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.25);
        text-align: center; transition: all 0.3s ease;
    }
    .metric-card:hover { transform: translateY(-5px); box-shadow: 0 10px 30px rgba(102, 126, 234, 0.35); }
    .metric-card.success { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .metric-card.warning { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .metric-value { font-size: 32px; font-weight: 700; margin-top: 8px; }
    .metric-label { font-size: 11px; opacity: 0.95; text-transform: uppercase; letter-spacing: 1.2px; }
    .stTabs [role="tab"] {
        font-size: 16px; font-weight: 600; background-color: white;
        border: 2px solid #e0e0e0; border-radius: 8px; padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: 2px solid #667eea;
    }
    .info-box {
        background: linear-gradient(135deg, #e8eaf6 0%, #f3e5f5 100%);
        border-left: 4px solid #667eea; padding: 15px; border-radius: 8px; margin: 15px 0;
    }
    .badge { padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 600; margin-right: 8px; }
    .badge-editable { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .badge-locked { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎯 KCP Share Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown("<hr style='border: 2px solid #667eea;'>", unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.markdown("### ⚙️ Configuration")
unsafe_load = st.sidebar.checkbox("🔓 Enable unsafe deserialization", value=True)
units_base = st.sidebar.number_input("📦 Units Base", min_value=1000, value=59000, step=1000)

# ---------- Load ----------
with st.spinner("📁 Loading..."):
    df, FEATURE_COLS, NON_EDITABLE, BASELINE_PRICES, BASELINE_SHARES, TOTAL_BASELINE_SHARE, XL_EDITABLE = load_data(DATA_PATH)
    model = load_model(MODEL_PATH, unsafe=unsafe_load)

SKU_COLS = [c for c in FEATURE_COLS if c not in NON_EDITABLE]
EDITABLE_COLS = [c for c in XL_EDITABLE if c in SKU_COLS]
LOCKED_COLS = [c for c in FEATURE_COLS if c not in EDITABLE_COLS]



# ---------- Price Editor ----------
st.markdown("### 🔧 Price Configuration")
header_name = {c: (f"🟨 {c}" if c in EDITABLE_COLS else f"🔒 {c}") for c in FEATURE_COLS}
price_table = pd.DataFrame([BASELINE_PRICES.values, BASELINE_PRICES.values],
                           columns=FEATURE_COLS, index=["Baseline Price", "Updated Price"])
price_table_display = price_table.rename(columns=header_name)

column_config = {header_name[c]: st.column_config.NumberColumn(
    header_name[c], min_value=0.0, step=0.5, format="₹%.2f", disabled=(c not in EDITABLE_COLS))
    for c in FEATURE_COLS}

edited_display = st.data_editor(price_table_display, use_container_width=True, num_rows="fixed",
                                column_config=column_config, disabled=["Baseline Price"])
edited_original = edited_display.rename(columns={v: k for k, v in header_name.items()})

BASELINE_INPUTS = {c: float(BASELINE_PRICES[c]) for c in FEATURE_COLS}
UPDATED_INPUTS = {c: (float(edited_original.loc["Updated Price", c]) if c in EDITABLE_COLS else BASELINE_INPUTS[c])
                  for c in FEATURE_COLS}

# ---------- Predictions ----------
baseline_vector = np.array([BASELINE_INPUTS[c] for c in FEATURE_COLS], dtype=np.float32).reshape(1, -1)
updated_vector = np.array([UPDATED_INPUTS[c] for c in FEATURE_COLS], dtype=np.float32).reshape(1, -1)

baseline_pred = np.asarray(model.predict(baseline_vector, verbose=0)).squeeze()
updated_pred = np.asarray(model.predict(updated_vector, verbose=0)).squeeze()

N_SKU = len(SKU_COLS)

def convert_outputs(arr):
    arr = np.array(arr, dtype=float).reshape(-1)
    if arr.size >= N_SKU + 1:
        kcp_raw, comp_raw = arr[:N_SKU], arr[N_SKU]
        if kcp_raw.sum() + comp_raw <= 1.5:
            kcp_pct, comp_pct = kcp_raw * 100.0, comp_raw * 100.0
        else:
            kcp_pct, comp_pct = kcp_raw, comp_raw
        return kcp_pct, float(kcp_pct.sum()), comp_pct
    else:
        kcp_pct = arr * 100.0 if arr.sum() <= 1.5 else arr
        return kcp_pct, float(kcp_pct.sum()), None

baseline_sku_pct, baseline_total_pred, baseline_comp = convert_outputs(baseline_pred)
updated_sku_pct, updated_total_pred, updated_comp = convert_outputs(updated_pred)

if baseline_comp is None:
    baseline_total_pct = TOTAL_BASELINE_SHARE
    price_changes_pct = [((UPDATED_INPUTS[c] - BASELINE_INPUTS[c]) / (BASELINE_INPUTS[c] or 1) * 100) for c in EDITABLE_COLS]
    avg_change = np.mean(price_changes_pct) if price_changes_pct else 0.0
    delta_share = -0.2 * avg_change
    updated_total_pct = max(0.0, min(100.0, baseline_total_pct + delta_share))
else:
    baseline_total_pct, updated_total_pct = baseline_total_pred, updated_total_pred

# Revenue
n_base, skus_base = len(baseline_sku_pct), SKU_COLS[:len(baseline_sku_pct)]
if n_base:
    base_units = (baseline_sku_pct / 100.0) * units_base
    base_prices = np.array([BASELINE_INPUTS[c] for c in skus_base])
    base_rev = float((base_units * base_prices).sum())
else:
    base_rev = 0.0

n_upd, skus_upd = len(updated_sku_pct), SKU_COLS[:len(updated_sku_pct)]
if n_upd:
    upd_units = (updated_sku_pct / 100.0) * units_base
    upd_prices = np.array([UPDATED_INPUTS[c] for c in skus_upd])
    upd_rev = float((upd_units * upd_prices).sum())
else:
    upd_rev = 0.0

rev_change = upd_rev - base_rev
rev_change_pct = (rev_change / base_rev * 100) if base_rev else 0.0

# ---------- Tabs ----------
tab_overview, tab_sku, tab_price, tab_analysis = st.tabs(["📊 Overview", "🎯 SKU Details", "💰 Price", "📈 Analysis"])

with tab_overview:
    st.markdown("### 📈 Key Performance Indicators")
    cols = st.columns(4)
    
    with cols[0]:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Baseline KCP Share</div>
            <div class="metric-value">{fmt_pct(baseline_total_pct)}</div></div>""", unsafe_allow_html=True)
    
    with cols[1]:
        change_class = "success" if updated_total_pct >= baseline_total_pct else "warning"
        st.markdown(f"""<div class="metric-card {change_class}">
            <div class="metric-label">Updated KCP Share</div>
            <div class="metric-value">{fmt_pct(updated_total_pct)}</div></div>""", unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown(f"""<div class="metric-card" style="background: linear-gradient(135deg, #485563 0%, #29323c 100%);">
            <div class="metric-label">Baseline Revenue</div>
            <div class="metric-value" style="font-size: 24px;">{fmt_currency(base_rev)}</div></div>""", unsafe_allow_html=True)
    
    with cols[3]:
        rev_class = "success" if rev_change >= 0 else "warning"
        arrow = "↑" if rev_change >= 0 else "↓"
        st.markdown(f"""<div class="metric-card {rev_class}">
            <div class="metric-label">Revenue Impact</div>
            <div class="metric-value" style="font-size: 22px;">{arrow} {fmt_currency(abs(rev_change))}</div>
            <div style="font-size: 14px;">({rev_change_pct:+.2f}%)</div></div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Market Share Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=updated_total_pct, domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "KCP Share (%)"}, delta={'reference': baseline_total_pct},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': COLORS['primary']},
                   'steps': [{'range': [0, 33], 'color': '#ecf0f1'}, 
                            {'range': [33, 67], 'color': '#e8eaf6'},
                            {'range': [67, 100], 'color': '#c5cae9'}],
                   'threshold': {'line': {'color': "red", 'width': 4}, 'value': baseline_total_pct}}
        ))
        fig_gauge.update_layout(height=400, margin=dict(l=20, r=20, t=70, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        st.markdown("#### 💰 Revenue Comparison")
        fig_rev = go.Figure([
            go.Bar(name='Baseline', x=['Revenue'], y=[base_rev], marker_color=COLORS['baseline'],
                   text=[fmt_currency(base_rev)], textposition='outside'),
            go.Bar(name='Updated', x=['Revenue'], y=[upd_rev], marker_color=COLORS['updated'],
                   text=[fmt_currency(upd_rev)], textposition='outside')
        ])
        fig_rev.update_layout(barmode='group', height=400, plot_bgcolor='rgba(0,0,0,0)',
                             yaxis_title="Revenue (₹)", legend=dict(x=0.7, y=1.1, orientation='h'))
        st.plotly_chart(fig_rev, use_container_width=True)

with tab_sku:
    st.markdown("### 🎯 SKU-Level Analysis")
    if n_base and n_upd:
        n = min(n_base, n_upd)
        skus = SKU_COLS[:n]
        base_units_n = (baseline_sku_pct[:n] / 100.0) * units_base
        upd_units_n = (updated_sku_pct[:n] / 100.0) * units_base
        base_prices_n = np.array([BASELINE_INPUTS[c] for c in skus])
        upd_prices_n = np.array([UPDATED_INPUTS[c] for c in skus])
        
        sku_df = pd.DataFrame({
            "SKU": skus,
            "Baseline Share (%)": np.round(baseline_sku_pct[:n], 2),
            "Updated Share (%)": np.round(updated_sku_pct[:n], 2),
            "Share Δ (%)": np.round(updated_sku_pct[:n] - baseline_sku_pct[:n], 2),
            "Baseline Units": np.round(base_units_n, 0).astype(int),
            "Updated Units": np.round(upd_units_n, 0).astype(int),
            "Baseline Revenue (₹)": np.round(base_units_n * base_prices_n, 2),
            "Updated Revenue (₹)": np.round(upd_units_n * upd_prices_n, 2),
            "Revenue Δ (₹)": np.round(upd_units_n * upd_prices_n - base_units_n * base_prices_n, 2)
        })
        st.dataframe(sku_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Share by SKU")
            fig_share = go.Figure([
                go.Bar(name='Baseline', x=skus, y=baseline_sku_pct[:n], marker_color=COLORS['baseline']),
                go.Bar(name='Updated', x=skus, y=updated_sku_pct[:n], marker_color=COLORS['updated'])
            ])
            fig_share.update_layout(barmode='group', height=400, plot_bgcolor='rgba(0,0,0,0)',
                                   yaxis_title="Share (%)", xaxis_tickangle=-45)
            st.plotly_chart(fig_share, use_container_width=True)
        
        with col2:
            st.markdown("#### 💵 Revenue by SKU")
            fig_rev_sku = go.Figure([
                go.Bar(name='Baseline', x=skus, y=base_units_n * base_prices_n, marker_color='#1abc9c'),
                go.Bar(name='Updated', x=skus, y=upd_units_n * upd_prices_n, marker_color='#f39c12')
            ])
            fig_rev_sku.update_layout(barmode='group', height=400, plot_bgcolor='rgba(0,0,0,0)',
                                     yaxis_title="Revenue (₹)", xaxis_tickangle=-45)
            st.plotly_chart(fig_rev_sku, use_container_width=True)

with tab_price:
    st.markdown("### 💰 Price Changes")
    price_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Baseline (₹)": [float(BASELINE_INPUTS[c]) for c in FEATURE_COLS],
        "Updated (₹)": [float(UPDATED_INPUTS[c]) for c in FEATURE_COLS]
    })
    price_df["Δ (₹)"] = price_df["Updated (₹)"] - price_df["Baseline (₹)"]
    price_df["Δ (%)"] = (price_df["Δ (₹)"] / price_df["Baseline (₹)"].replace(0, np.nan) * 100).fillna(0).round(2)
    st.dataframe(price_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Absolute Changes")
        fig_abs = px.bar(price_df, x='Feature', y='Δ (₹)', color='Δ (₹)',
                        color_continuous_scale=['#e74c3c', '#ecf0f1', '#2ecc71'])
        fig_abs.update_layout(height=400, showlegend=False, plot_bgcolor='rgba(0,0,0,0)', xaxis_tickangle=-45)
        st.plotly_chart(fig_abs, use_container_width=True)
    
    with col2:
        st.markdown("#### Percentage Changes")
        fig_pct = px.bar(price_df, x='Feature', y='Δ (%)', color='Δ (%)',
                        color_continuous_scale=['#e74c3c', '#ecf0f1', '#2ecc71'])
        fig_pct.update_layout(height=400, showlegend=False, plot_bgcolor='rgba(0,0,0,0)', xaxis_tickangle=-45)
        st.plotly_chart(fig_pct, use_container_width=True)

with tab_analysis:
    st.markdown("### 📈 Deep Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        if n_base:
            st.markdown("#### Baseline Distribution")
            fig_pie1 = px.pie(values=baseline_sku_pct[:n_base], names=SKU_COLS[:n_base],
                             color_discrete_sequence=px.colors.qualitative.Set3)
            fig_pie1.update_layout(height=400)
            st.plotly_chart(fig_pie1, use_container_width=True)
    
    with col2:
        if n_upd:
            st.markdown("#### Updated Distribution")
            fig_pie2 = px.pie(values=updated_sku_pct[:n_upd], names=SKU_COLS[:n_upd],
                             color_discrete_sequence=px.colors.qualitative.Set2)
            fig_pie2.update_layout(height=400)
            st.plotly_chart(fig_pie2, use_container_width=True)
    
    if n_base and n_upd:
        st.markdown("---")
        st.markdown("#### Price Elasticity")
        n = min(n_base, n_upd)
        scatter_df = pd.DataFrame({
            'SKU': SKU_COLS[:n],
            'Price Δ (%)': [(UPDATED_INPUTS[c] - BASELINE_INPUTS[c]) / BASELINE_INPUTS[c] * 100 for c in SKU_COLS[:n]],
            'Share Δ (%)': updated_sku_pct[:n] - baseline_sku_pct[:n]
        })
        fig_scatter = px.scatter(scatter_df, x='Price Δ (%)', y='Share Δ (%)', hover_name='SKU',
                                size=abs(scatter_df['Share Δ (%)']), color='Share Δ (%)',
                                color_continuous_scale='RdYlGn')
        fig_scatter.update_layout(height=450, plot_bgcolor='rgba(0,0,0,0)')
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")
st.caption("📌 Editable columns are from the KCC row. Model predictions use softmax outputs.")