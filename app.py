import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from rfm_engine import compute_rfm, segment_customer, get_recommendations
from data_loader import load_olist_data, load_sample_data

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation · Olist RFM",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main { background: #0f1117; }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1d2e 0%, #16213e 100%);
        border: 1px solid #2d3561;
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .metric-card .label {
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #6b7db3;
        margin-bottom: 8px;
    }
    .metric-card .value {
        font-size: 32px;
        font-weight: 700;
        color: #e2e8f0;
        line-height: 1;
    }
    .metric-card .delta {
        font-size: 12px;
        color: #48bb78;
        margin-top: 6px;
    }
    
    .segment-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .section-header {
        font-size: 20px;
        font-weight: 700;
        color: #e2e8f0;
        margin: 32px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #2d3561;
    }
    
    .stDataFrame { border-radius: 12px; overflow: hidden; }
    
    .rec-card {
        background: linear-gradient(135deg, #1a1d2e, #16213e);
        border-left: 4px solid;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
    }
    
    div[data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #21262d;
    }
    
    .stSelectbox > div, .stFileUploader > div {
        background: #161b22;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Segment Color Map ─────────────────────────────────────────────────────────
SEGMENT_COLORS = {
    "Champions":          "#f6c90e",
    "Loyal Customers":    "#4ade80",
    "Promising":          "#60a5fa",
    "Potential Loyalists":"#a78bfa",
    "At Risk":            "#fb923c",
    "Needs Attention":    "#f472b6",
    "Hibernating":        "#94a3b8",
    "Lost":               "#ef4444",
}

SEGMENT_ICONS = {
    "Champions":          "🏆",
    "Loyal Customers":    "💎",
    "Promising":          "🌟",
    "Potential Loyalists":"🌱",
    "At Risk":            "⚠️",
    "Needs Attention":    "👀",
    "Hibernating":        "😴",
    "Lost":               "❌",
}

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 RFM Segmentation")
    st.markdown("---")
    
    data_source = st.radio(
        "**Data Source**",
        ["📦 Sample Olist Data", "📁 Upload CSV Files"],
        index=0,
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ RFM Thresholds")
    
    recency_bins  = st.slider("Recency Quantiles",  2, 5, 4)
    frequency_bins = st.slider("Frequency Quantiles", 2, 5, 4)
    monetary_bins  = st.slider("Monetary Quantiles",  2, 5, 4)
    
    st.markdown("---")
    st.markdown("### 🔍 Filter Segments")
    
    segments_to_show = st.multiselect(
        "Show Segments",
        options=list(SEGMENT_COLORS.keys()),
        default=list(SEGMENT_COLORS.keys()),
    )
    
    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px;color:#4a5568;text-align:center'>"
        "Dataset: <a href='https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce' "
        "style='color:#60a5fa' target='_blank'>Olist Brazilian E-Commerce</a>"
        "</div>",
        unsafe_allow_html=True,
    )

# ─── Load Data ─────────────────────────────────────────────────────────────────
st.markdown("# 🛒 Customer Segmentation Dashboard")
st.markdown("<div style='color:#6b7db3;margin-top:-12px;margin-bottom:24px;'>RFM Analysis · Olist Brazilian E-Commerce</div>", unsafe_allow_html=True)

if data_source == "📦 Sample Olist Data":
    with st.spinner("Loading Olist sample data..."):
        orders_df, items_df = load_sample_data()
    st.success("✅ Sample data loaded (500 customers · synthetic Olist schema)")
else:
    st.info("Upload `olist_orders_dataset.csv` and `olist_order_items_dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).")
    col1, col2 = st.columns(2)
    with col1:
        orders_file = st.file_uploader("orders CSV", type="csv", key="orders")
    with col2:
        items_file  = st.file_uploader("order items CSV", type="csv", key="items")
    
    if orders_file and items_file:
        orders_df = pd.read_csv(orders_file)
        items_df  = pd.read_csv(items_file)
        st.success("✅ Files loaded successfully!")
    else:
        st.warning("👆 Upload both files to proceed, or switch to sample data.")
        st.stop()

# ─── Compute RFM ───────────────────────────────────────────────────────────────
with st.spinner("Computing RFM scores..."):
    rfm = compute_rfm(orders_df, items_df, recency_bins, frequency_bins, monetary_bins)

rfm_filtered = rfm[rfm["customer_segment"].isin(segments_to_show)]

# ─── KPI Row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

kpis = [
    ("Total Customers",   f"{len(rfm):,}",                    ""),
    ("Total Revenue",     f"R$ {rfm['monetary_value'].sum():,.0f}", ""),
    ("Avg Order Value",   f"R$ {rfm['monetary_value'].mean():,.0f}", ""),
    ("Champions",         f"{(rfm['customer_segment']=='Champions').sum():,}", "🏆"),
    ("At-Risk Customers", f"{(rfm['customer_segment']=='At Risk').sum():,}", "⚠️"),
]

for col, (label, value, icon) in zip([c1, c2, c3, c4, c5], kpis):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{icon} {value}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Charts Row 1 ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Segment Overview</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1])

with col_left:
    seg_counts = rfm_filtered["customer_segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]
    seg_counts["Color"] = seg_counts["Segment"].map(SEGMENT_COLORS)

    fig_donut = go.Figure(go.Pie(
        labels=seg_counts["Segment"],
        values=seg_counts["Count"],
        hole=0.55,
        marker=dict(colors=seg_counts["Color"].tolist(),
                    line=dict(color="#0f1117", width=3)),
        textinfo="label+percent",
        textfont=dict(size=12, color="white"),
        hovertemplate="<b>%{label}</b><br>Customers: %{value}<br>Share: %{percent}<extra></extra>",
    ))
    fig_donut.update_layout(
        title=dict(text="Customer Distribution", font=dict(color="#e2e8f0", size=15)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#a0aec0"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=40, b=0, l=0, r=0),
        height=350,
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with col_right:
    seg_rev = rfm_filtered.groupby("customer_segment")["monetary_value"].sum().reset_index()
    seg_rev.columns = ["Segment", "Revenue"]
    seg_rev = seg_rev.sort_values("Revenue", ascending=True)
    seg_rev["Color"] = seg_rev["Segment"].map(SEGMENT_COLORS)

    fig_bar = go.Figure(go.Bar(
        x=seg_rev["Revenue"],
        y=seg_rev["Segment"],
        orientation="h",
        marker=dict(color=seg_rev["Color"].tolist(), line=dict(width=0)),
        text=[f"R$ {v:,.0f}" for v in seg_rev["Revenue"]],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=11),
        hovertemplate="<b>%{y}</b><br>Revenue: R$ %{x:,.0f}<extra></extra>",
    ))
    fig_bar.update_layout(
        title=dict(text="Revenue by Segment", font=dict(color="#e2e8f0", size=15)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, color="#6b7db3"),
        yaxis=dict(color="#a0aec0", tickfont=dict(size=12)),
        margin=dict(t=40, b=20, l=10, r=80),
        height=350,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ─── Charts Row 2 ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔬 RFM Deep Dive</div>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)

for col, metric, label, color in zip(
    [col_a, col_b, col_c],
    ["recency_days", "frequency", "monetary_value"],
    ["Recency (days)", "Frequency (orders)", "Monetary Value (R$)"],
    ["#60a5fa", "#4ade80", "#f6c90e"],
):
    with col:
        fig = px.histogram(
            rfm_filtered, x=metric, nbins=30,
            color_discrete_sequence=[color],
            labels={metric: label},
        )
        fig.update_layout(
            title=dict(text=label, font=dict(color="#e2e8f0", size=13)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#6b7db3", gridcolor="#1a1d2e"),
            yaxis=dict(color="#6b7db3", gridcolor="#1a1d2e"),
            margin=dict(t=36, b=20, l=10, r=10),
            height=260,
            showlegend=False,
            bargap=0.05,
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

# ─── Scatter Plot ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📡 Recency vs Monetary · Bubble = Frequency</div>', unsafe_allow_html=True)

sample = rfm_filtered.sample(min(800, len(rfm_filtered)), random_state=42)
sample["color"] = sample["customer_segment"].map(SEGMENT_COLORS)

fig_scatter = px.scatter(
    sample,
    x="recency_days",
    y="monetary_value",
    size="frequency",
    color="customer_segment",
    color_discrete_map=SEGMENT_COLORS,
    hover_data={"recency_days": True, "frequency": True, "monetary_value": True},
    labels={
        "recency_days":    "Days Since Last Purchase",
        "monetary_value":  "Total Spend (R$)",
        "frequency":       "Order Frequency",
        "customer_segment": "Segment",
    },
    size_max=22,
)
fig_scatter.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(color="#6b7db3", gridcolor="#1a1d2e"),
    yaxis=dict(color="#6b7db3", gridcolor="#1a1d2e"),
    legend=dict(font=dict(color="#a0aec0"), bgcolor="rgba(0,0,0,0)", title_font_color="#e2e8f0"),
    margin=dict(t=10, b=20, l=10, r=10),
    height=420,
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ─── Segment Recommendations ───────────────────────────────────────────────────
st.markdown('<div class="section-header">💡 Actionable Recommendations</div>', unsafe_allow_html=True)

recs = get_recommendations()
cols = st.columns(2)

for i, (segment, data) in enumerate(recs.items()):
    if segment not in segments_to_show:
        continue
    color  = SEGMENT_COLORS.get(segment, "#6b7db3")
    icon   = SEGMENT_ICONS.get(segment, "📌")
    count  = (rfm["customer_segment"] == segment).sum()
    pct    = count / len(rfm) * 100

    with cols[i % 2]:
        st.markdown(f"""
        <div class="rec-card" style="border-color:{color}">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
                <span style="font-size:16px;font-weight:700;color:{color}">{icon} {segment}</span>
                <span style="font-size:12px;color:#6b7db3;background:#1a1d2e;padding:3px 10px;border-radius:12px">
                    {count:,} customers · {pct:.1f}%
                </span>
            </div>
            <div style="font-size:13px;color:#a0aec0;margin-bottom:8px">
                <b style="color:#e2e8f0">Strategy:</b> {data['recommendation']}
            </div>
            <div style="font-size:12px;color:#6b7db3">
                <b>Actions:</b> {data['suggested_action']}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── Data Table ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Customer Data Table</div>', unsafe_allow_html=True)

col_search, col_seg, col_dl = st.columns([2, 2, 1])
with col_search:
    search = st.text_input("🔍 Search customer ID", "")
with col_seg:
    seg_filter = st.selectbox("Filter by segment", ["All"] + list(SEGMENT_COLORS.keys()))
with col_dl:
    st.markdown("<br>", unsafe_allow_html=True)
    csv = rfm_filtered.to_csv(index=False).encode()
    st.download_button("⬇️ Export CSV", csv, "rfm_segments.csv", "text/csv")

display_df = rfm_filtered.copy()
if search:
    display_df = display_df[display_df.index.astype(str).str.contains(search, case=False)]
if seg_filter != "All":
    display_df = display_df[display_df["customer_segment"] == seg_filter]

display_df = display_df[["recency_days", "frequency", "monetary_value",
                          "r_score", "f_score", "m_score", "rfm_total_score", "customer_segment"]]
display_df.columns = ["Recency (days)", "Frequency", "Monetary (R$)",
                      "R Score", "F Score", "M Score", "RFM Total", "Segment"]

st.dataframe(
    display_df.sort_values("RFM Total", ascending=False).head(200),
    use_container_width=True,
    height=380,
)

st.markdown(f"<div style='color:#4a5568;font-size:12px;text-align:right'>Showing top 200 of {len(display_df):,} records</div>", unsafe_allow_html=True)
