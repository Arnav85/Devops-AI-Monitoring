"""
analytics.py - AI / ML Processing Engine
=========================================
Three analytical capabilities:

  1. Anomaly Detection
     Trains an Isolation Forest on recent CPU/Memory/Disk metrics read from
     SQLite.  Each new snapshot is scored; scores below the decision boundary
     are flagged as anomalies.

  2. Threshold-Breach Forecasting
     Uses a smoothed Linear-Regression trend (with optional ExponentialSmoothing
     from statsmodels when available) to predict when a metric will exceed a
     configurable threshold.  Returns minutes-to-breach so operators can act
     proactively.

  3. NLP Log Analysis
     Reads the JSON app.log file, vectorises error messages with TF-IDF, groups
     them into clusters via KMeans, and extracts a root-cause string from each
     cluster.

All public functions return plain Python dicts / DataFrames — no side effects.
"""

import os
import re
import json
import sqlite3
from collections import Counter
from datetime import datetime, timedelta

import numpy  as np
import pandas as pd
from sklearn.ensemble            import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster             import KMeans
from sklearn.linear_model        import LinearRegression
from sklearn.preprocessing       import StandardScaler

# Optional: statsmodels for Exponential Smoothing
try:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    _STATSMODELS_OK = True
except ImportError:
    _STATSMODELS_OK = False

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file so every module uses the same data)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH  = os.path.join(DATA_DIR, "metrics.db")
LOG_PATH = os.path.join(DATA_DIR, "app.log")

# Minimum rows before the anomaly model can be trained
MIN_TRAINING_SAMPLES = 20

# Features used for anomaly detection
ANOMALY_FEATURES = ["cpu_percent", "memory_percent", "disk_percent"]


# ===========================================================================
# 1. Anomaly Detection — Isolation Forest
# ===========================================================================

def load_metrics(limit: int = 500) -> pd.DataFrame:
    """
    Load the most recent *limit* rows from the metrics SQLite table.
    Returns an empty DataFrame if the DB does not exist yet.
    """
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            f"""
            SELECT timestamp, cpu_percent, memory_percent, memory_used_mb,
                   disk_percent, disk_read_mb, disk_write_mb,
                   net_sent_mb, net_recv_mb
            FROM   metrics
            ORDER  BY id DESC
            LIMIT  {int(limit)}
            """,
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Re-order chronologically so charts render correctly
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def train_isolation_forest(df: pd.DataFrame):
    """
    Fit an Isolation Forest on *df*.
    Returns (model, scaler) — or (None, None) when there is not enough data.
    """
    feature_df = df[ANOMALY_FEATURES].dropna()

    if len(feature_df) < MIN_TRAINING_SAMPLES:
        return None, None

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df)

    model = IsolationForest(
        n_estimators=150,
        contamination=0.05,   # expect ~5 % anomalous points
        max_samples="auto",
        random_state=42,
    )
    model.fit(X_scaled)
    return model, scaler


def detect_anomalies(
    df: pd.DataFrame, model, scaler
) -> pd.DataFrame:
    """
    Score every row with the fitted Isolation Forest.

    Adds two columns:
      • anomaly       — bool, True = Isolation Forest says outlier
      • anomaly_score — float, decision_function output
                        (negative values are more anomalous; 0 = boundary)

    When model is None (not trained yet) both columns default to safe values.
    """
    df = df.copy()
    df["anomaly"]       = False
    df["anomaly_score"] = 0.0

    if model is None:
        return df

    feature_df = df[ANOMALY_FEATURES].dropna()
    if feature_df.empty:
        return df

    X_scaled = scaler.transform(feature_df)

    predictions = model.predict(X_scaled)          # -1 = anomaly, 1 = normal
    scores      = model.decision_function(X_scaled) # lower → more anomalous

    df.loc[feature_df.index, "anomaly"]       = predictions == -1
    df.loc[feature_df.index, "anomaly_score"] = scores

    return df


def get_anomaly_summary(df: pd.DataFrame) -> dict:
    """Return a concise dict describing the current anomaly state."""
    if df.empty or "anomaly" not in df.columns:
        return {
            "status":             "insufficient_data",
            "anomalies_detected": 0,
        }

    recent        = df.tail(10)
    anomaly_count = int(recent["anomaly"].sum())
    latest        = df.iloc[-1]
    is_anomaly    = bool(latest.get("anomaly", False))

    return {
        "status":             "critical" if is_anomaly else "healthy",
        "anomalies_detected": anomaly_count,
        "latest_cpu":         round(float(latest["cpu_percent"]),    2),
        "latest_memory":      round(float(latest["memory_percent"]), 2),
        "latest_disk":        round(float(latest["disk_percent"]),   2),
        "timestamp":          str(latest["timestamp"]),
    }


# ===========================================================================
# 2. Threshold-Breach Forecasting
# ===========================================================================

def _smooth_series(series: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply simple moving-average smoothing to reduce noise."""
    s = pd.Series(series)
    return s.rolling(window=min(window, len(s)), min_periods=1).mean().values


def _forecast_linear(series: np.ndarray, steps: int) -> np.ndarray:
    """Linear-regression extrapolation over *steps* future points."""
    X = np.arange(len(series)).reshape(-1, 1)
    y = series
    reg = LinearRegression().fit(X, y)
    future_X = np.arange(len(series), len(series) + steps).reshape(-1, 1)
    return reg.predict(future_X), float(reg.coef_[0])


def _forecast_exp_smoothing(series: np.ndarray, steps: int):
    """
    Exponential Smoothing via statsmodels (preferred when available).
    Falls back to linear regression on any failure.
    """
    try:
        model    = SimpleExpSmoothing(series).fit(smoothing_level=0.3, optimized=False)
        forecast = model.forecast(steps)
        # Derive approximate slope from first and last forecast points
        slope    = (float(forecast[-1]) - float(series[-1])) / steps if steps > 1 else 0.0
        return np.array(forecast), slope
    except Exception:
        return _forecast_linear(series, steps)


def forecast_threshold_breach(
    df: pd.DataFrame,
    metric: str  = "memory_percent",
    threshold: float = 90.0,
    forecast_steps: int = 12,    # 12 × 5 s intervals = 1 minute ahead
) -> dict:
    """
    Predict when *metric* will breach *threshold*.

    Returns a dict with:
      will_breach         — bool
      minutes_to_breach   — float or None
      current_value       — latest observed value
      trend_slope         — rate of change per 5-second interval
      forecasted_values   — list of predicted values
    """
    base = {"metric": metric, "threshold": threshold, "will_breach": False}

    if df.empty or metric not in df.columns:
        return {**base, "reason": "no_data"}

    raw = df[metric].dropna().values
    if len(raw) < 10:
        return {**base, "reason": "insufficient_data"}

    # Work on last 60 points (~5 minutes) to avoid stale history
    raw = raw[-60:]
    smoothed = _smooth_series(raw, window=5)

    if _STATSMODELS_OK:
        future_vals, slope = _forecast_exp_smoothing(smoothed, forecast_steps)
    else:
        future_vals, slope = _forecast_linear(smoothed, forecast_steps)

    current_value = float(raw[-1])

    # Find the first future step that crosses the threshold
    will_breach       = False
    steps_to_breach   = None

    for i, val in enumerate(future_vals):
        if float(val) >= threshold:
            will_breach     = True
            steps_to_breach = i + 1
            break

    return {
        "metric":            metric,
        "current_value":     round(current_value, 2),
        "threshold":         threshold,
        "will_breach":       will_breach,
        "steps_to_breach":   steps_to_breach,
        "minutes_to_breach": round((steps_to_breach * 5) / 60, 1) if steps_to_breach else None,
        "trend_slope":       round(float(slope), 5),
        "forecasted_values": [round(float(v), 2) for v in future_vals],
    }


# ===========================================================================
# 3. NLP Log Analysis — TF-IDF + KMeans Clustering
# ===========================================================================

def _parse_log_file(max_lines: int = 2000) -> list:
    """
    Read the last *max_lines* lines from app.log and parse each as JSON.
    Lines that fail JSON parsing are silently skipped.
    """
    if not os.path.exists(LOG_PATH):
        return []

    try:
        with open(LOG_PATH, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return []

    entries = []
    for line in lines[-max_lines:]:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return entries


def _extract_rca(message: str) -> str:
    """
    Apply regex patterns to produce a concise root-cause string from
    a raw error message.  Returns a truncated fallback if no pattern matches.
    """
    patterns = [
        (r"Error\s+(\d+):\s*(.+)",          lambda m: f"HTTP {m.group(1)}: {m.group(2).strip()}"),
        (r"Out of Memory Exception[: ]*(.+)",lambda m: f"OOM – {(m.group(1).strip() or 'memory exhausted')}"),
        (r"CRITICAL[: ]*(.+)",               lambda m: f"Critical failure: {m.group(1).strip()}"),
        (r"[Tt]imeout[: ]*(.+)",             lambda m: f"Timeout: {m.group(1).strip()}"),
        (r"503 Service Unavailable",         lambda m: "Upstream returned 503"),
    ]
    for pattern, formatter in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            try:
                return formatter(match)
            except Exception:
                pass
    # Fallback: first 100 characters
    return message[:100]


def analyze_logs(max_lines: int = 2000) -> dict:
    """
    Parse app.log, extract error/warning messages, cluster them with
    TF-IDF + KMeans, and return a summary dict.

    Returns:
      clusters      — list of cluster dicts sorted by occurrence count
      error_count   — total errors+warnings seen
      total_count   — total log entries parsed
    """
    entries = _parse_log_file(max_lines)

    total_count = len(entries)
    error_entries = [
        e for e in entries if e.get("log_level") in ("ERROR", "CRITICAL", "WARNING")
    ]
    error_count = len(error_entries)

    if error_count < 2:
        return {
            "clusters":    [],
            "error_count": error_count,
            "total_count": total_count,
        }

    error_messages = [e.get("message", "") for e in error_entries]

    # TF-IDF vectorisation
    vectorizer = TfidfVectorizer(
        max_features=200,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    try:
        X = vectorizer.fit_transform(error_messages)
    except ValueError:
        return {"clusters": [], "error_count": error_count, "total_count": total_count}

    # Choose cluster count: at most 5, at most half the messages
    n_clusters = min(5, max(2, len(error_messages) // 2))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    try:
        labels = kmeans.fit_predict(X)
    except Exception:
        return {"clusters": [], "error_count": error_count, "total_count": total_count}

    clusters = []
    for cid in range(n_clusters):
        indices       = [i for i, lbl in enumerate(labels) if lbl == cid]
        cluster_msgs  = [error_messages[i] for i in indices]
        cluster_ents  = [error_entries[i]  for i in indices]

        if not cluster_msgs:
            continue

        most_common_msg, freq = Counter(cluster_msgs).most_common(1)[0]
        services = sorted(set(e.get("service_name", "unknown") for e in cluster_ents))

        clusters.append({
            "cluster_id":        cid,
            "count":             len(cluster_msgs),
            "root_cause":        most_common_msg,
            "rca":               _extract_rca(most_common_msg),
            "frequency":         freq,
            "affected_services": services,
            "sample_messages":   cluster_msgs[:3],
        })

    # Rank by occurrence
    clusters.sort(key=lambda c: c["count"], reverse=True)

    return {
        "clusters":    clusters,
        "error_count": error_count,
        "total_count": total_count,
    }


def get_recent_incidents(hours: float = 1.0) -> list:
    """
    Return up to 50 most recent ERROR/CRITICAL log entries within the last
    *hours* hours as a list of incident dicts (for the dashboard table).
    """
    entries = _parse_log_file(max_lines=5000)
    cutoff  = datetime.utcnow() - timedelta(hours=hours)

    incidents = []
    for entry in entries:
        if entry.get("log_level") not in ("ERROR", "CRITICAL"):
            continue
        try:
            raw_ts = entry["timestamp"].rstrip("Z")
            ts     = datetime.fromisoformat(raw_ts)
            if ts < cutoff:
                continue
        except (KeyError, ValueError):
            continue

        incidents.append({
            "timestamp": entry.get("timestamp", ""),
            "service":   entry.get("service_name", "unknown"),
            "level":     entry.get("log_level", "ERROR"),
            "message":   entry.get("message", ""),
            "rca":       _extract_rca(entry.get("message", "")),
        })

    # Newest first, cap at 50
    incidents.sort(key=lambda x: x["timestamp"], reverse=True)
    return incidents[:50]


# ===========================================================================
# Convenience wrapper — full pipeline in one call
# ===========================================================================

def run_full_analysis() -> dict:
    """
    Execute the complete analytics pipeline and return a structured report.

    Keys:
      generated_at      — ISO timestamp
      metrics_available — row count in DB
      anomaly_detection — output of get_anomaly_summary()
      forecasts         — {memory, disk, cpu} forecasts
      log_analysis      — output of analyze_logs()
      recent_incidents  — list from get_recent_incidents()
      df_annotated      — annotated DataFrame (for callers that need raw data)
    """
    df = load_metrics(limit=200)

    report = {
        "generated_at":      datetime.utcnow().isoformat(),
        "metrics_available": len(df),
        "anomaly_detection": {},
        "forecasts":         {},
        "log_analysis":      {},
        "recent_incidents":  [],
        "df_annotated":      None,
    }

    if not df.empty:
        model, scaler      = train_isolation_forest(df)
        df_annotated       = detect_anomalies(df, model, scaler)
        report["anomaly_detection"] = get_anomaly_summary(df_annotated)
        report["df_annotated"]      = df_annotated

        report["forecasts"]["memory"] = forecast_threshold_breach(
            df_annotated, "memory_percent", threshold=90.0
        )
        report["forecasts"]["disk"] = forecast_threshold_breach(
            df_annotated, "disk_percent", threshold=90.0
        )
        report["forecasts"]["cpu"] = forecast_threshold_breach(
            df_annotated, "cpu_percent", threshold=85.0
        )

    report["log_analysis"]     = analyze_logs()
    report["recent_incidents"] = get_recent_incidents(hours=1)

    return report


# ===========================================================================
# Standalone test
# ===========================================================================

if __name__ == "__main__":
    import time as _time
    print("[Analytics] Running standalone analysis — ensure collector.py is running first.")
    _time.sleep(2)
    result = run_full_analysis()
    # Remove the DataFrame before pretty-printing (not JSON-serialisable)
    result.pop("df_annotated", None)
    print(json.dumps(result, indent=2, default=str))
