"""
app.py — Fraud Detection Analytics Dashboard
Run: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, json, sys

# ── Path setup ───────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
SRC_DIR     = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700; color: #1a1a2e;
        border-bottom: 3px solid #e63946; padding-bottom: 0.5rem; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.2rem 1.5rem; border-radius: 12px; color: white;
        border-left: 4px solid #e63946;
    }
    .metric-value { font-size: 2rem; font-weight: 800; color: #e63946; }
    .metric-label { font-size: 0.85rem; color: #adb5bd; text-transform: uppercase; letter-spacing: 0.05em; }
    .risk-critical { background-color: #e63946; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }
    .risk-high     { background-color: #f77f00; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }
    .risk-medium   { background-color: #fcbf49; color: black;  padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }
    .risk-low      { background-color: #06d6a0; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }
    .stDataFrame { border: 1px solid #dee2e6 !important; border-radius: 8px; }
    div[data-testid="stSidebarContent"] { background: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

RISK_COLORS = {
    "CRITICAL": "#e63946",
    "HIGH":     "#f77f00",
    "MEDIUM":   "#fcbf49",
    "LOW":      "#06d6a0",
}

TIER_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]


# ── Data loaders (cached) ─────────────────────────────────────

@st.cache_data(ttl=300)
def load_transactions():
    path = os.path.join(DATA_DIR, "transactions.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["txn_datetime"])
    df["txn_hour"] = df["txn_datetime"].dt.hour
    df["txn_month"] = df["txn_datetime"].dt.to_period("M").astype(str)
    return df

@st.cache_data(ttl=300)
def load_scored():
    path = os.path.join(DATA_DIR, "scored_transactions.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["txn_datetime"])

@st.cache_data(ttl=300)
def load_users():
    path = os.path.join(DATA_DIR, "users.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_data(ttl=300)
def load_report(name: str):
    path = os.path.join(REPORTS_DIR, f"{name}.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_data(ttl=300)
def load_summary():
    path = os.path.join(REPORTS_DIR, "summary_stats.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ── Sidebar ───────────────────────────────────────────────────

def render_sidebar(txn: pd.DataFrame, scored: pd.DataFrame):
    with st.sidebar:
        st.markdown("## 🛡️ Fraud Detection")
        st.markdown("---")
        st.markdown("### Filters")

        risk_filter = st.multiselect(
            "Risk Tier",
            options=TIER_ORDER,
            default=TIER_ORDER,
            key="risk_filter"
        )

        categories = ["All"] + sorted(txn["merchant_category"].dropna().unique().tolist())
        cat_filter = st.selectbox("Merchant Category", categories, key="cat_filter")

        countries = ["All"] + sorted(txn["transaction_country"].dropna().unique().tolist())
        cntry_filter = st.selectbox("Country", countries, key="cntry_filter")

        min_score, max_score = st.slider(
            "Composite Score Range",
            min_value=0.0, max_value=1.0,
            value=(0.0, 1.0), step=0.01, key="score_range"
        )

        st.markdown("---")
        st.markdown("### Navigation")
        page = st.radio(
            "",
            ["📊 Executive Overview",
             "🚨 Transaction Monitor",
             "👤 User Risk Profiles",
             "📈 Trend Analysis",
             "🔬 Model Insights"],
            key="nav"
        )

    return page, risk_filter, cat_filter, cntry_filter, min_score, max_score


# ── KPI Cards ─────────────────────────────────────────────────

def render_kpis(stats: dict):
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, "Total Transactions", f"{stats.get('total_transactions', 0):,}", ""),
        (c2, "Confirmed Fraud",    f"{stats.get('total_fraud', 0):,}", ""),
        (c3, "Fraud Rate",         f"{stats.get('fraud_rate_pct', 0):.2f}%", ""),
        (c4, "Fraud Exposure",     f"${stats.get('total_fraud_exposure', 0):,.0f}", ""),
        (c5, "High-Risk Users",    f"{stats.get('high_risk_users', 0):,}", ""),
    ]
    for col, label, value, suffix in kpis:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)


# ── Page: Executive Overview ──────────────────────────────────

def page_overview(txn, scored, stats):
    st.markdown('<div class="main-header">📊 Executive Overview</div>', unsafe_allow_html=True)
    render_kpis(stats)
    st.markdown("---")

    col1, col2 = st.columns(2)

    # Risk tier donut
    with col1:
        tier_df = load_report("risk_tier_distribution")
        if not tier_df.empty:
            tier_df["risk_tier"] = pd.Categorical(tier_df["risk_tier"], categories=TIER_ORDER, ordered=True)
            tier_df = tier_df.sort_values("risk_tier")
            fig = px.pie(tier_df, names="risk_tier", values="txn_count",
                         color="risk_tier",
                         color_discrete_map=RISK_COLORS,
                         title="Transaction Risk Tier Distribution",
                         hole=0.5)
            fig.update_traces(textposition="outside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

    # Monthly fraud trend
    with col2:
        monthly = load_report("monthly_fraud_summary")
        if not monthly.empty:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=monthly["year_month"], y=monthly["total_txns"],
                                  name="Total Txns", marker_color="#457b9d", opacity=0.6))
            fig.add_trace(go.Scatter(x=monthly["year_month"], y=monthly["fraud_rate_pct"],
                                      mode="lines+markers", name="Fraud Rate %",
                                      marker_color="#e63946", line=dict(width=2.5)),
                          secondary_y=True)
            fig.update_layout(title="Monthly Transaction Volume & Fraud Rate", showlegend=True)
            fig.update_yaxes(title_text="Transaction Count", secondary_y=False)
            fig.update_yaxes(title_text="Fraud Rate (%)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    # Fraud by category
    with col3:
        cat_df = load_report("fraud_by_category")
        if not cat_df.empty:
            fig = px.bar(cat_df.sort_values("fraud_rate_pct"),
                         x="fraud_rate_pct", y="merchant_category",
                         orientation="h", color="fraud_rate_pct",
                         color_continuous_scale="Reds",
                         title="Fraud Rate by Merchant Category (%)")
            fig.update_layout(coloraxis_showscale=False, yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

    # Fraud by country
    with col4:
        cntry_df = load_report("fraud_by_country")
        if not cntry_df.empty:
            fig = px.choropleth(cntry_df,
                                locations="transaction_country",
                                locationmode="ISO-3 alpha",
                                color="fraud_rate_pct",
                                hover_name="transaction_country",
                                hover_data=["fraud_txns", "total_txns"],
                                color_continuous_scale="OrRd",
                                title="Fraud Rate by Country (%)")
            st.plotly_chart(fig, use_container_width=True)


# ── Page: Transaction Monitor ─────────────────────────────────

def page_transaction_monitor(scored, risk_filter, cat_filter, cntry_filter, min_s, max_s):
    st.markdown('<div class="main-header">🚨 Transaction Monitor</div>', unsafe_allow_html=True)

    if scored.empty:
        st.warning("No scored transactions found. Run src/risk_scoring.py first.")
        return

    filtered = scored[scored["risk_tier"].isin(risk_filter)]
    filtered = filtered[
        (filtered["composite_score"] >= min_s) &
        (filtered["composite_score"] <= max_s)
    ]
    if cat_filter != "All":
        filtered = filtered[filtered["merchant_category"] == cat_filter]
    if cntry_filter != "All":
        filtered = filtered[filtered["transaction_country"] == cntry_filter]

    st.markdown(f"**{len(filtered):,}** transactions match current filters.")

    # Scatter: amount vs composite score
    fig = px.scatter(
        filtered.sample(min(5000, len(filtered))),
        x="composite_score", y="amount",
        color="risk_tier",
        color_discrete_map=RISK_COLORS,
        hover_data=["transaction_id", "user_id", "merchant_category", "transaction_country"],
        title="Transaction Risk Scatter (Amount vs Composite Score)",
        opacity=0.7,
        size_max=8,
        log_y=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table of High/Critical
    st.subheader("High & Critical Risk Transactions")
    high_crit = filtered[filtered["risk_tier"].isin(["HIGH", "CRITICAL"])].copy()
    display_cols = ["transaction_id", "user_id", "txn_datetime", "amount",
                    "merchant_category", "transaction_country",
                    "composite_score", "risk_tier", "is_fraud"]
    available_cols = [c for c in display_cols if c in high_crit.columns]
    high_crit = high_crit[available_cols].sort_values("composite_score", ascending=False)
    high_crit["composite_score"] = high_crit["composite_score"].round(4)
    st.dataframe(high_crit.head(200), use_container_width=True)

    # CSV download
    csv = high_crit.to_csv(index=False)
    st.download_button("⬇️ Download High/Critical CSV", csv,
                       file_name="high_critical_transactions.csv", mime="text/csv")


# ── Page: User Risk Profiles ──────────────────────────────────

def page_user_profiles(scored, users):
    st.markdown('<div class="main-header">👤 User Risk Profiles</div>', unsafe_allow_html=True)

    if scored.empty or users.empty:
        st.warning("Data not available.")
        return

    # User aggregation
    user_agg = scored.groupby("user_id").agg(
        total_txns=("transaction_id", "count"),
        total_amount=("amount", "sum"),
        avg_score=("composite_score", "mean"),
        max_score=("composite_score", "max"),
        confirmed_frauds=("is_fraud", "sum"),
        critical_count=("risk_tier", lambda x: (x == "CRITICAL").sum()),
        high_count=("risk_tier", lambda x: (x == "HIGH").sum()),
    ).reset_index()

    user_agg = user_agg.merge(
        users[["user_id", "credit_score", "account_age_days", "is_flagged_account"]],
        on="user_id", how="left"
    )
    user_agg["risk_label"] = user_agg["avg_score"].apply(
        lambda s: "CRITICAL" if s >= 0.75 else "HIGH" if s >= 0.55 else "MEDIUM" if s >= 0.30 else "LOW"
    )

    col1, col2 = st.columns(2)
    with col1:
        # Credit score vs avg risk score
        fig = px.scatter(
            user_agg,
            x="credit_score", y="avg_score",
            color="risk_label",
            color_discrete_map=RISK_COLORS,
            size="total_txns",
            hover_data=["user_id", "confirmed_frauds", "total_amount"],
            title="Credit Score vs Average Risk Score (per User)",
            opacity=0.7
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Account age vs fraud count
        fig = px.scatter(
            user_agg[user_agg["confirmed_frauds"] > 0],
            x="account_age_days", y="confirmed_frauds",
            color="risk_label",
            color_discrete_map=RISK_COLORS,
            hover_data=["user_id", "avg_score"],
            title="Account Age vs Confirmed Fraud Transactions",
            opacity=0.8
        )
        st.plotly_chart(fig, use_container_width=True)

    # Top risky users
    st.subheader("Top 50 Highest-Risk Users")
    top_users = user_agg.sort_values("avg_score", ascending=False).head(50)
    top_users["avg_score"] = top_users["avg_score"].round(4)
    top_users["total_amount"] = top_users["total_amount"].round(2)
    st.dataframe(top_users, use_container_width=True)

    # User lookup
    st.subheader("🔍 User Lookup")
    uid = st.text_input("Enter User ID (e.g. U00001)")
    if uid:
        user_txns = scored[scored["user_id"] == uid]
        if user_txns.empty:
            st.warning(f"No transactions found for {uid}")
        else:
            st.success(f"Found {len(user_txns)} transactions for {uid}")
            st.dataframe(user_txns.sort_values("composite_score", ascending=False), use_container_width=True)
            fig = px.line(
                user_txns.sort_values("txn_datetime"),
                x="txn_datetime", y="composite_score",
                color="risk_tier", color_discrete_map=RISK_COLORS,
                title=f"Risk Score Timeline — {uid}"
            )
            st.plotly_chart(fig, use_container_width=True)


# ── Page: Trend Analysis ──────────────────────────────────────

def page_trends(txn):
    st.markdown('<div class="main-header">📈 Trend Analysis</div>', unsafe_allow_html=True)

    if txn.empty:
        st.warning("Transactions data not available.")
        return

    col1, col2 = st.columns(2)

    # Hourly fraud heatmap
    with col1:
        hourly = load_report("hourly_fraud_heatmap")
        if not hourly.empty:
            fig = go.Figure(go.Bar(
                x=hourly["hour_of_day"], y=hourly["fraud_rate_pct"],
                marker_color=hourly["fraud_rate_pct"],
                marker_colorscale="Reds",
                text=hourly["fraud_rate_pct"].round(2),
                textposition="outside"
            ))
            fig.update_layout(title="Fraud Rate by Hour of Day (%)", xaxis_title="Hour", yaxis_title="Fraud Rate %")
            st.plotly_chart(fig, use_container_width=True)

    # Fraud by device
    with col2:
        device_df = txn.groupby("device_type").agg(
            total=("transaction_id", "count"),
            fraud=("is_fraud", "sum")
        ).reset_index()
        device_df["fraud_rate"] = (device_df["fraud"] / device_df["total"] * 100).round(2)
        fig = px.bar(device_df, x="device_type", y="fraud_rate",
                     color="fraud_rate", color_continuous_scale="Reds",
                     title="Fraud Rate by Device Type (%)", text="fraud_rate")
        fig.update_traces(texttemplate="%{text}%")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Monthly fraud exposure
    monthly = load_report("monthly_fraud_summary")
    if not monthly.empty:
        fig = px.area(monthly, x="year_month", y="fraud_exposure",
                      title="Monthly Fraud Exposure ($)",
                      color_discrete_sequence=["#e63946"])
        fig.update_layout(xaxis_title="Month", yaxis_title="Fraud Exposure ($)")
        st.plotly_chart(fig, use_container_width=True)


# ── Page: Model Insights ──────────────────────────────────────

def page_model_insights(scored):
    st.markdown('<div class="main-header">🔬 Model Insights</div>', unsafe_allow_html=True)

    if scored.empty:
        st.warning("Scored transactions not found.")
        return

    col1, col2 = st.columns(2)

    # Score distribution
    with col1:
        fig = px.histogram(
            scored, x="composite_score", color="risk_tier",
            color_discrete_map=RISK_COLORS,
            nbins=50,
            title="Composite Score Distribution",
            barmode="overlay",
            opacity=0.7,
            category_orders={"risk_tier": TIER_ORDER}
        )
        fig.update_layout(xaxis_title="Composite Score", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    # Rule vs ML score comparison
    with col2:
        if "rule_score" in scored.columns and "ml_score" in scored.columns:
            sample = scored.sample(min(3000, len(scored)))
            fig = px.scatter(
                sample,
                x="rule_score", y="ml_score",
                color="risk_tier",
                color_discrete_map=RISK_COLORS,
                title="Rule Score vs ML Score",
                opacity=0.6,
                hover_data=["transaction_id", "composite_score"]
            )
            st.plotly_chart(fig, use_container_width=True)

    # Precision / Recall metrics display
    st.subheader("📋 Model Performance Summary")
    col3, col4, col5 = st.columns(3)
    col3.metric("Model Type", "Random Forest + Rules + Isolation Forest")
    col4.metric("Features Used", "14 engineered features")
    col5.metric("Anomaly Contamination Rate", "2.5%")

    st.info("""
    **Scoring Architecture:**
    - **Rule-Based Score (35%)**: Domain heuristics — amount z-score, velocity, odd-hour, country risk, category risk, account flags
    - **Random Forest Score (45%)**: Supervised classifier trained on labeled fraud data; accounts for feature interactions
    - **Isolation Forest Score (20%)**: Unsupervised anomaly detector; identifies transactions deviating from normal distribution
    
    **Risk Tiers:**
    - 🟢 LOW (0.00–0.30): Standard processing
    - 🟡 MEDIUM (0.30–0.55): Soft review recommended
    - 🟠 HIGH (0.55–0.75): Manual review required
    - 🔴 CRITICAL (0.75–1.00): Immediate block + investigation
    """)

    # Isolation flag overlap
    if "isolation_flag" in scored.columns:
        st.subheader("Isolation Forest vs Confirmed Fraud Overlap")
        cross = pd.crosstab(scored["isolation_flag"], scored["is_fraud"])
        cross.index = ["Not Flagged", "Flagged"]
        cross.columns = ["Not Fraud", "Fraud"]
        st.dataframe(cross, use_container_width=False)


# ── MAIN ──────────────────────────────────────────────────────

def main():
    txn    = load_transactions()
    scored = load_scored()
    users  = load_users()
    stats  = load_summary()

    page, risk_filter, cat_filter, cntry_filter, min_s, max_s = render_sidebar(txn, scored)

    if page == "📊 Executive Overview":
        page_overview(txn, scored, stats)
    elif page == "🚨 Transaction Monitor":
        page_transaction_monitor(scored, risk_filter, cat_filter, cntry_filter, min_s, max_s)
    elif page == "👤 User Risk Profiles":
        page_user_profiles(scored, users)
    elif page == "📈 Trend Analysis":
        page_trends(txn)
    elif page == "🔬 Model Insights":
        page_model_insights(scored)


if __name__ == "__main__":
    main()
