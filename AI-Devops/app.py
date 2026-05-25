"""
app.py - AI-DevOps Monitoring Dashboard  (Streamlit)
=====================================================
Single-page interactive dashboard providing:

  • Live KPI cards  — CPU / Memory / Disk / Network
  • Time-series charts (Plotly dark theme) with red ✕ anomaly markers
  • Isolation Forest decision-score sparkline
  • Predictive threshold-breach panel (linear / exp-smoothing forecast)
  • NLP error-cluster accordion (TF-IDF + KMeans RCA)
  • Incident history table with CSV export
  • Recent raw log incidents (last hour)
  • Sidebar: self-healing toggle + manual action buttons + auto-refresh

Start:
    streamlit run app.py
"""

import os
import sys
import time

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Ensure module imports work regardless of CWD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from analytics import (
    load_metrics,
    train_isolation_forest,
    detect_anomalies,
    get_anomaly_summary,
    forecast_threshold_breach,
    analyze_logs,
    get_recent_incidents,
)
from remediation import (
    init_incident_db,
    get_incidents,
    is_healing_enabled,
    set_healing_enabled,
    action_rotate_logs,
    action_disk_cleanup,
    action_flush_memory_cache,
    action_restart_service,
)

# ===========================================================================
# Page config  (must be the very first Streamlit call)
# ===========================================================================
st.set_page_config(
    page_title="AI-DevOps Monitor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# Global styling
# ===========================================================================
st.markdown("""
<style>
/* Tighten up the top padding */
.block-container { padding-top: 1rem; }

/* KPI card tint */
div[data-testid="metric-container"] {
    background: #1e1e2e;
    border: 1px solid #333355;
    border-radius: 8px;
    padding: 12px 16px;
}

/* Section header accent */
h2 { border-left: 4px solid #4dabf7; padding-left: 8px; }

/* Badge helpers */
.badge-critical { background:#e03131; color:#fff; padding:2px 8px;
                  border-radius:4px; font-size:0.78em; font-weight:600; }
.badge-warning  { background:#e67700; color:#fff; padding:2px 8px;
                  border-radius:4px; font-size:0.78em; font-weight:600; }
.badge-info     { background:#2f9e44; color:#fff; padding:2px 8px;
                  border-radius:4px; font-size:0.78em; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ===========================================================================
# Cached data loaders  (short TTL so the dashboard feels live)
# ===========================================================================

@st.cache_data(ttl=5)
def _cached_metrics(limit: int = 120) -> pd.DataFrame:
    return load_metrics(limit=limit)


@st.cache_data(ttl=8)
def _cached_analysis():
    """Returns (df_annotated, summary) or (empty_df, None)."""
    df = load_metrics(limit=200)
    if df.empty:
        return df, None
    model, scaler  = train_isolation_forest(df)
    df_annotated   = detect_anomalies(df, model, scaler)
    summary        = get_anomaly_summary(df_annotated)
    return df_annotated, summary


@st.cache_data(ttl=15)
def _cached_log_analysis():
    return analyze_logs()


@st.cache_data(ttl=10)
def _cached_incidents():
    return get_incidents(limit=50)


@st.cache_data(ttl=10)
def _cached_recent_incidents():
    return get_recent_incidents(hours=1)


# ===========================================================================
# Utility helpers
# ===========================================================================

def _status_icon(value: float, warn: float = 70.0, crit: float = 85.0) -> str:
    if value >= crit:  return "🔴"
    if value >= warn:  return "🟡"
    return "🟢"


def _plotly_base() -> dict:
    """Shared Plotly layout settings for a consistent dark look."""
    return dict(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#e0e0e0", size=12),
        margin=dict(l=40, r=20, t=40, b=40),
    )


# ===========================================================================
# Sidebar
# ===========================================================================

def _render_sidebar() -> tuple:
    """Render sidebar controls.  Returns (auto_refresh: bool, refresh_sec: int)."""
    st.sidebar.title("⚙️ Control Panel")

    # ── Self-healing toggle ───────────────────────────────────────────────
    st.sidebar.subheader("🔧 Self-Healing Automation")
    healing_on = is_healing_enabled()

    col_en, col_dis = st.sidebar.columns(2)
    with col_en:
        if st.button("✅ Enable",  disabled=healing_on,     key="btn_en",
                     use_container_width=True):
            set_healing_enabled(True)
            st.cache_data.clear()
            st.rerun()
    with col_dis:
        if st.button("❌ Disable", disabled=not healing_on, key="btn_dis",
                     use_container_width=True):
            set_healing_enabled(False)
            st.cache_data.clear()
            st.rerun()

    badge = "🟢 **ENABLED**" if healing_on else "🔴 **DISABLED**"
    st.sidebar.markdown(f"Status: {badge}")

    st.sidebar.divider()

    # ── Manual action triggers ────────────────────────────────────────────
    st.sidebar.subheader("🛠️ Manual Remediation")

    if st.sidebar.button("🔄 Rotate Logs",        use_container_width=True):
        msg = action_rotate_logs()
        st.sidebar.success(msg)

    if st.sidebar.button("🧹 Disk Cleanup",        use_container_width=True):
        msg = action_disk_cleanup()
        st.sidebar.success(msg)

    if st.sidebar.button("🧠 Flush Memory Cache",  use_container_width=True):
        msg = action_flush_memory_cache()
        st.sidebar.success(msg)

    if st.sidebar.button("🔁 Restart api-gateway", use_container_width=True):
        msg = action_restart_service("api-gateway")
        st.sidebar.success(msg)

    st.sidebar.divider()

    # ── Auto-refresh controls ─────────────────────────────────────────────
    st.sidebar.subheader("🔃 Auto-Refresh")
    auto_refresh  = st.sidebar.checkbox("Enable Auto-Refresh", value=True)
    refresh_rate  = st.sidebar.slider("Interval (seconds)", 5, 60, 10)

    if st.sidebar.button("🔄 Refresh Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    return auto_refresh, refresh_rate


# ===========================================================================
# Section: Header / System Status Banner
# ===========================================================================

def _render_header(summary: dict | None) -> None:
    col_title, col_status, col_healing = st.columns([3, 1, 1])

    with col_title:
        st.title("🔍 AI-DevOps Monitoring Dashboard")
        st.caption(f"UTC: {datetime.utcnow().strftime('%Y-%m-%d  %H:%M:%S')}")

    with col_status:
        if summary is None:
            st.info("⏳ Collecting…")
        elif summary.get("status") == "critical":
            st.error("🔴 ANOMALY DETECTED")
        else:
            st.success("🟢 SYSTEM HEALTHY")

    with col_healing:
        if is_healing_enabled():
            st.success("🔧 Self-Healing ON")
        else:
            st.error("🔧 Self-Healing OFF")


# ===========================================================================
# Section: KPI Cards
# ===========================================================================

def _render_kpi_cards(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("⏳ Waiting for metrics… start `collector.py` first.")
        return

    latest  = df.iloc[-1]
    tail10  = df.tail(10)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        cpu  = float(latest.get("cpu_percent", 0))
        avg  = float(tail10["cpu_percent"].mean())
        st.metric(
            label=f"{_status_icon(cpu)} CPU Usage",
            value=f"{cpu:.1f}%",
            delta=f"{cpu - avg:+.1f}% vs 10-pt avg",
        )
    with c2:
        mem  = float(latest.get("memory_percent", 0))
        avg  = float(tail10["memory_percent"].mean())
        st.metric(
            label=f"{_status_icon(mem)} Memory Usage",
            value=f"{mem:.1f}%",
            delta=f"{mem - avg:+.1f}% vs 10-pt avg",
        )
    with c3:
        disk = float(latest.get("disk_percent", 0))
        st.metric(
            label=f"{_status_icon(disk, 70, 85)} Disk Usage",
            value=f"{disk:.1f}%",
        )
    with c4:
        net_s = float(latest.get("net_sent_mb", 0))
        net_r = float(latest.get("net_recv_mb", 0))
        st.metric(
            label="🌐 Network I/O",
            value=f"↑ {net_s:.3f} MB/s",
            delta=f"↓ {net_r:.3f} MB/s recv",
        )


# ===========================================================================
# Section: Time-Series Charts with Anomaly Markers
# ===========================================================================

def _render_timeseries(df: pd.DataFrame) -> None:
    st.subheader("📈 Real-Time Metrics  (red ✕ = AI anomaly)")

    if df.empty or len(df) < 2:
        st.info("Not enough data points yet.")
        return

    ts    = df["timestamp"]
    has_a = "anomaly" in df.columns

    normal_mask  = (~df["anomaly"])          if has_a else pd.Series([True]  * len(df))
    anomaly_mask = df["anomaly"].astype(bool) if has_a else pd.Series([False] * len(df))

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("CPU (%)", "Memory (%)", "Disk (%)", "Network I/O (MB/s)"),
        shared_xaxes=False,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    def _add_series(row: int, col: int, key: str, color: str, threshold: float) -> None:
        """Add normal line + anomaly scatter + threshold dashed line."""
        # Normal line
        fig.add_trace(go.Scatter(
            x=ts[normal_mask], y=df[key][normal_mask],
            mode="lines+markers",
            name=f"{key} normal",
            line=dict(color=color, width=1.8),
            marker=dict(size=3),
            showlegend=False,
        ), row=row, col=col)

        # Anomaly markers
        if anomaly_mask.any():
            fig.add_trace(go.Scatter(
                x=ts[anomaly_mask], y=df[key][anomaly_mask],
                mode="markers",
                name="Anomaly",
                marker=dict(size=11, color="red", symbol="x",
                            line=dict(color="darkred", width=2)),
                showlegend=(row == 1 and col == 1),  # show legend once
            ), row=row, col=col)

        # Warning threshold dashed line
        fig.add_hline(y=threshold, line_dash="dash",
                      line_color="orange", opacity=0.45,
                      row=row, col=col)

    _add_series(1, 1, "cpu_percent",    "#4dabf7", 85.0)
    _add_series(1, 2, "memory_percent", "#51cf66", 85.0)
    _add_series(2, 1, "disk_percent",   "#fcc419", 85.0)

    # Network — two overlaid lines (no anomaly overlay needed)
    fig.add_trace(go.Scatter(
        x=ts, y=df["net_sent_mb"],
        mode="lines", name="Net Sent",
        line=dict(color="#74c0fc", width=1.5),
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=ts, y=df["net_recv_mb"],
        mode="lines", name="Net Recv",
        line=dict(color="#63e6be", width=1.5),
    ), row=2, col=2)

    fig.update_layout(
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **_plotly_base(),
    )
    st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# Section: Anomaly Score Sparkline
# ===========================================================================

def _render_anomaly_score(df: pd.DataFrame, summary: dict | None) -> None:
    st.subheader("🚨 AI Anomaly Detection  (Isolation Forest)")

    col_status, col_chart = st.columns([1, 3])

    with col_status:
        if summary is None:
            st.info("⏳ Training model…\n\nNeed ≥ 20 data points (~2 min).")
        else:
            status = summary.get("status", "unknown")
            count  = summary.get("anomalies_detected", 0)
            if status == "critical":
                st.error(f"🔴 **ANOMALY**\n\n{count} events in last 10 readings")
            else:
                st.success("🟢 **HEALTHY**\n\nNo active anomalies")

            cpu = summary.get("latest_cpu", 0)
            mem = summary.get("latest_memory", 0)
            dsk = summary.get("latest_disk", 0)
            st.markdown(f"CPU **{cpu}%** | Mem **{mem}%** | Disk **{dsk}%**")

    with col_chart:
        if df is not None and not df.empty and "anomaly_score" in df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df["anomaly_score"],
                mode="lines",
                name="IF Decision Score",
                line=dict(color="#74c0fc", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(116,192,252,0.08)",
            ))
            # Mark anomalous points
            if "anomaly" in df.columns:
                adf = df[df["anomaly"] == True]
                if not adf.empty:
                    fig.add_trace(go.Scatter(
                        x=adf["timestamp"], y=adf["anomaly_score"],
                        mode="markers", name="Anomaly",
                        marker=dict(size=9, color="red", symbol="x"),
                    ))
            fig.add_hline(y=0, line_dash="dash", line_color="red",
                          opacity=0.5, annotation_text="Decision boundary (0)")
            fig.update_layout(
                height=220,
                title="Isolation Forest Decision Score  (negative = anomalous)",
                legend=dict(orientation="h"),
                **_plotly_base(),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Score chart will appear once the model has been trained.")


# ===========================================================================
# Section: Forecast Panel
# ===========================================================================

def _render_forecasts(df: pd.DataFrame) -> None:
    st.subheader("🔮 Predictive Threshold Forecasting")

    if df.empty:
        st.info("No data available for forecasting yet.")
        return

    metrics = [
        ("Memory",  "memory_percent", "🧠", 90.0),
        ("Disk",    "disk_percent",   "💾", 90.0),
        ("CPU",     "cpu_percent",    "⚡", 85.0),
    ]

    cols = st.columns(3)
    for (label, key, icon, threshold), col in zip(metrics, cols):
        fc = forecast_threshold_breach(df, key, threshold=threshold)
        with col:
            current = fc.get("current_value", 0)
            slope   = fc.get("trend_slope", 0)
            trend   = "📈" if slope > 0.01 else ("📉" if slope < -0.01 else "➡️")

            if fc.get("will_breach"):
                mins = fc.get("minutes_to_breach", "?")
                st.warning(
                    f"{icon} **{label}**\n\n"
                    f"⚠️ Breach in ≈ **{mins} min**\n\n"
                    f"Current: {current}%  {trend}"
                )
            else:
                st.success(
                    f"{icon} **{label}**\n\n"
                    f"✅ No breach predicted\n\n"
                    f"Current: {current}%  {trend}"
                )

        # Mini forecast sparkline
        future_vals = fc.get("forecasted_values", [])
        if future_vals:
            with col:
                fig = go.Figure(go.Scatter(
                    y=future_vals,
                    mode="lines",
                    line=dict(
                        color="red" if fc.get("will_breach") else "#51cf66",
                        width=2,
                    ),
                    fill="tozeroy",
                    fillcolor=(
                        "rgba(224,49,49,0.15)"
                        if fc.get("will_breach")
                        else "rgba(81,207,102,0.1)"
                    ),
                ))
                fig.add_hline(y=threshold, line_dash="dot",
                              line_color="orange", opacity=0.6)
                fig.update_layout(
                    height=130,
                    showlegend=False,
                    xaxis=dict(visible=False),
                    yaxis=dict(title=""),
                    **_plotly_base(),
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# Section: NLP Log Cluster Analysis
# ===========================================================================

def _render_log_analysis(log_analysis: dict) -> None:
    st.subheader("📋 NLP Log Analysis  (TF-IDF + KMeans Clustering)")

    total    = log_analysis.get("total_count", 0)
    errors   = log_analysis.get("error_count", 0)
    err_rate = (errors / total * 100) if total > 0 else 0.0

    c_meta, c_clusters = st.columns([1, 3])

    with c_meta:
        st.metric("Log Entries Scanned", total)
        st.metric("Error / Warning Count", errors)
        if err_rate > 20:
            st.error(f"⚠️ Error Rate: {err_rate:.1f}%")
        elif err_rate > 10:
            st.warning(f"⚠️ Error Rate: {err_rate:.1f}%")
        else:
            st.success(f"✅ Error Rate: {err_rate:.1f}%")

    with c_clusters:
        clusters = log_analysis.get("clusters", [])
        if not clusters:
            st.info("No significant error clusters detected yet.")
        else:
            st.markdown("**Top Error Clusters  (ranked by frequency)**")
            for cl in clusters[:4]:
                icon   = "🔴" if cl["count"] > 5 else "🟡"
                svc    = ", ".join(cl["affected_services"][:3])
                label  = cl["root_cause"]
                label  = label[:70] + "…" if len(label) > 70 else label
                with st.expander(f"{icon}  [{cl['count']}×]  {label}"):
                    st.markdown(f"**RCA:**  {cl['rca']}")
                    st.markdown(f"**Affected services:**  {svc}")
                    st.markdown("**Sample messages:**")
                    for msg in cl.get("sample_messages", [])[:3]:
                        st.code(msg, language=None)


# ===========================================================================
# Section: Incident History Table
# ===========================================================================

def _render_incident_history() -> None:
    st.subheader("📜 Incident History & Root Cause Analysis")

    incidents = _cached_incidents()

    if not incidents:
        st.info("No incidents recorded yet.  "
                "Start `remediation.py` to populate this table.")
        return

    df = pd.DataFrame(incidents)
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"])
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    display = df[
        ["timestamp", "severity", "category", "description", "rca", "action_taken"]
    ].copy()
    display.columns = ["Time", "Severity", "Category", "Description", "Root Cause", "Auto-Action"]

    # Severity colour mapping via pandas Styler (.map is the non-deprecated API)
    def _colour_severity(val: str) -> str:
        return {
            "CRITICAL": "color: #ff6b6b; font-weight: bold",
            "WARNING":  "color: #ffa94d; font-weight: bold",
            "INFO":     "color: #69db7c",
        }.get(val, "")

    styled = display.style.map(_colour_severity, subset=["Severity"])

    st.dataframe(styled, use_container_width=True, height=380)

    csv = display.to_csv(index=False)
    st.download_button(
        "⬇️  Download Incident Report (CSV)",
        data=csv,
        file_name=f"incidents_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )


# ===========================================================================
# Section: Recent Raw Log Incidents
# ===========================================================================

def _render_recent_log_incidents() -> None:
    st.subheader("⚡ Recent Log Incidents  (last hour)")

    recent = _cached_recent_incidents()

    if not recent:
        st.success("✅ No ERROR / CRITICAL log entries in the past hour.")
        return

    df = pd.DataFrame(recent)
    # Strip trailing Z before parsing to keep pandas happy
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"].str.rstrip("Z"))
        .dt.strftime("%H:%M:%S")
    )
    display = df[["timestamp", "service", "level", "message", "rca"]].head(25)
    display.columns = ["Time", "Service", "Level", "Message", "RCA"]

    st.dataframe(display, use_container_width=True, height=300)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    # Ensure incident DB exists before any reads
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    init_incident_db()

    # Sidebar — must be called before body content for correct layout order
    auto_refresh, refresh_rate = _render_sidebar()

    # Load all data (cached)
    df, summary    = _cached_analysis()
    log_analysis   = _cached_log_analysis()

    # ── Header ───────────────────────────────────────────────────────────
    _render_header(summary)
    st.divider()

    # ── KPI cards ─────────────────────────────────────────────────────────
    _render_kpi_cards(df)
    st.divider()

    # ── Time-series + anomaly chart ───────────────────────────────────────
    _render_timeseries(df)
    st.divider()

    # ── Anomaly detection detail ──────────────────────────────────────────
    _render_anomaly_score(df, summary)
    st.divider()

    # ── Forecast panel ────────────────────────────────────────────────────
    _render_forecasts(df)
    st.divider()

    # ── NLP log analysis ──────────────────────────────────────────────────
    _render_log_analysis(log_analysis)
    st.divider()

    # ── Incident history ──────────────────────────────────────────────────
    _render_incident_history()
    st.divider()

    # ── Recent raw log incidents ──────────────────────────────────────────
    _render_recent_log_incidents()

    # ── Auto-refresh loop (blocking — Streamlit handles threading) ─────────
    if auto_refresh:
        time.sleep(refresh_rate)
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()
