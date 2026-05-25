"""
remediation.py - Self-Healing Rules Engine & Alert System
===========================================================
Components:

  • Incident database  — lightweight SQLite store for every triggered event
  • Self-healing flag  — a simple flag file toggles auto-remediation on/off;
                         readable from the dashboard without shared state
  • Remediation actions — 100 % mock / simulated; no real system services are
                          touched.  All actions are logged to remediation_log.txt
  • Alert system       — sends structured JSON payloads to Slack or MS Teams
                         incoming webhooks (env vars); falls back to local log
  • Deduplication      — MD5 fingerprint + time-window guard prevents storms
  • Rules engine       — evaluate_rules() maps analytics findings to actions
  • Daemon             — remediation_daemon() loops every 30 s (run standalone)

Run standalone:
    python remediation.py
"""

import os
import sys
import json
import time
import sqlite3
import hashlib
import threading
from datetime import datetime, timedelta

import requests  # standard pip install — no admin required

# Import our analytics pipeline
from analytics import run_full_analysis

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_DIR         = os.path.join(BASE_DIR, "data")
INCIDENT_DB_PATH = os.path.join(DATA_DIR, "incidents.db")
REMEDIATION_LOG  = os.path.join(BASE_DIR, "remediation_log.txt")
HEALING_FLAG     = os.path.join(DATA_DIR, "healing_enabled.flag")

# Safe dummy directories — the only filesystem locations we touch
DUMMY_TEMP_DIR   = os.path.join(DATA_DIR, "temp_logs")
DUMMY_ARCHIVE    = os.path.join(DATA_DIR, "log_archive")

# ---------------------------------------------------------------------------
# Webhook configuration (optional — set environment variables to activate)
# ---------------------------------------------------------------------------
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")
TEAMS_WEBHOOK_URL = os.environ.get("TEAMS_WEBHOOK_URL", "")

# ---------------------------------------------------------------------------
# Alert deduplication
# ---------------------------------------------------------------------------
_alert_history_lock = threading.Lock()
_alert_history: dict = {}   # fingerprint → datetime of last send
DEDUP_WINDOW_SECONDS = 300  # suppress duplicate alert for 5 minutes


# ===========================================================================
# Incident Database
# ===========================================================================

def init_incident_db() -> None:
    """Create the incidents table if it does not exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(INCIDENT_DB_PATH)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    TEXT NOT NULL,
                severity     TEXT NOT NULL,
                category     TEXT NOT NULL,
                description  TEXT NOT NULL,
                rca          TEXT DEFAULT '',
                action_taken TEXT DEFAULT '',
                resolved     INTEGER DEFAULT 0
            )
        """)
        conn.commit()
    finally:
        conn.close()


def log_incident(
    severity: str,
    category: str,
    description: str,
    rca: str = "",
    action_taken: str = "",
) -> None:
    """Persist one incident record to the database."""
    conn = sqlite3.connect(INCIDENT_DB_PATH)
    try:
        conn.execute(
            """
            INSERT INTO incidents
                (timestamp, severity, category, description, rca, action_taken)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (datetime.utcnow().isoformat(), severity, category,
             description, rca, action_taken),
        )
        conn.commit()
    finally:
        conn.close()


def get_incidents(limit: int = 50) -> list:
    """Return the most recent *limit* incidents as a list of dicts."""
    if not os.path.exists(INCIDENT_DB_PATH):
        return []

    conn = sqlite3.connect(INCIDENT_DB_PATH)
    try:
        cursor = conn.execute(
            """
            SELECT timestamp, severity, category, description, rca,
                   action_taken, resolved
            FROM   incidents
            ORDER  BY id DESC
            LIMIT  ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    return [
        {
            "timestamp":    row[0],
            "severity":     row[1],
            "category":     row[2],
            "description":  row[3],
            "rca":          row[4],
            "action_taken": row[5],
            "resolved":     bool(row[6]),
        }
        for row in rows
    ]


# ===========================================================================
# Self-Healing Toggle
# ===========================================================================

def is_healing_enabled() -> bool:
    """Return True when the self-healing flag file exists."""
    return os.path.exists(HEALING_FLAG)


def set_healing_enabled(enabled: bool) -> None:
    """Create or remove the healing flag file and write an audit entry."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if enabled:
        with open(HEALING_FLAG, "w") as fh:
            fh.write(f"enabled at {datetime.utcnow().isoformat()}\n")
        _log_action("Self-healing ENABLED by operator")
    else:
        if os.path.exists(HEALING_FLAG):
            os.remove(HEALING_FLAG)
        _log_action("Self-healing DISABLED by operator")


# ===========================================================================
# Internal logging helper
# ===========================================================================

def _log_action(message: str, level: str = "INFO") -> None:
    """Append a timestamped entry to remediation_log.txt and stdout."""
    ts    = datetime.utcnow().isoformat()
    entry = f"[{ts}] [{level:>12s}] {message}"
    print(f"[Remediation] {entry}")
    try:
        with open(REMEDIATION_LOG, "a", encoding="utf-8") as fh:
            fh.write(entry + "\n")
    except OSError:
        pass  # log write failure must never crash the engine


# ===========================================================================
# Mock Remediation Actions
# ===========================================================================

def action_rotate_logs() -> str:
    """
    Simulate log rotation:
      1. Creates a few dummy .tmp files in data/temp_logs/
      2. Removes them (the 'cleanup')
      3. Writes an archive marker to data/log_archive/
    Zero system impact — operates only inside the project data/ directory.
    """
    _log_action("ACTION: Initiating log rotation procedure", "REMEDIATION")

    os.makedirs(DUMMY_TEMP_DIR, exist_ok=True)
    os.makedirs(DUMMY_ARCHIVE,  exist_ok=True)

    # Seed some dummy files so the cleanup looks realistic
    for i in range(3):
        path = os.path.join(DUMMY_TEMP_DIR, f"temp_log_{i}.tmp")
        if not os.path.exists(path):
            with open(path, "w") as fh:
                fh.write("dummy temp content\n" * 50)

    removed = 0
    for fname in os.listdir(DUMMY_TEMP_DIR):
        fpath = os.path.join(DUMMY_TEMP_DIR, fname)
        try:
            os.remove(fpath)
            removed += 1
        except OSError as exc:
            _log_action(f"  Could not remove {fpath}: {exc}", "WARNING")

    archive_name = f"archive_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.marker"
    with open(os.path.join(DUMMY_ARCHIVE, archive_name), "w") as fh:
        fh.write(f"Rotation completed at {datetime.utcnow().isoformat()}\n")

    _log_action(f"  Cleared {removed} temp files; archive marker: {archive_name}")
    _log_action("ACTION: Log rotation completed", "REMEDIATION")
    return f"Log rotation: removed {removed} temp files, created archive marker"


def action_disk_cleanup() -> str:
    """
    Simulate disk space recovery:
      Prints a realistic multi-step cleanup sequence.
      No real files outside the project data/ directory are modified.
    """
    _log_action("ACTION: Disk pressure — initiating cleanup simulation", "REMEDIATION")
    steps = [
        "Scanning for log files older than 7 days (simulated) ...",
        "Compressing 4 rotated logs → .gz  (simulated) ...",
        "Removing build artifact cache (simulated) ...",
        "Clearing pip wheel cache (simulated) ...",
        "Estimated 12 GB freed (simulated)",
    ]
    for step in steps:
        _log_action(f"  {step}")
    _log_action("ACTION: Disk cleanup completed", "REMEDIATION")
    return "Disk cleanup: compressed old logs, removed build artifacts (simulated)"


def action_flush_memory_cache() -> str:
    """
    Simulate application-level memory pressure mitigation.
    """
    _log_action("ACTION: Memory pressure — flushing application cache", "REMEDIATION")
    steps = [
        "Identifying top-3 memory consumers (simulated) ...",
        "Flushing Redis/in-memory cache (simulated) ...",
        "Requesting JVM garbage-collection cycle (simulated) ...",
        "Reducing connection-pool size from 100 → 50 (simulated) ...",
    ]
    for step in steps:
        _log_action(f"  {step}")
    _log_action("ACTION: Memory optimisation completed", "REMEDIATION")
    return "Memory cache flushed and GC triggered (simulated)"


def action_restart_service(service_name: str = "unknown-service") -> str:
    """
    Simulate a graceful service restart sequence.
    Writes a realistic sequence to the remediation log — no actual
    process management calls are made.
    """
    _log_action(f"ACTION: Restarting service '{service_name}'", "REMEDIATION")
    steps = [
        f"[1/4] Sending SIGTERM to {service_name} (simulated) ...",
        f"[2/4] Waiting up to 15 s for graceful shutdown (simulated) ...",
        f"[3/4] Clearing PID file for {service_name} (simulated) ...",
        f"[4/4] Launching new instance of {service_name} (simulated) ...",
    ]
    for step in steps:
        _log_action(f"  {step}")
    _log_action(f"ACTION: Service '{service_name}' restart complete", "REMEDIATION")
    return f"Service '{service_name}' graceful restart completed (simulated)"


# ===========================================================================
# Alert / Webhook System
# ===========================================================================

def _fingerprint(category: str, description: str) -> str:
    """Create an MD5 hash from category + first 60 chars of description."""
    raw = f"{category}:{description[:60]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()   # noqa: S324 (non-cryptographic use)


def _is_duplicate(fp: str) -> bool:
    """Return True if this fingerprint was sent within the dedup window."""
    with _alert_history_lock:
        last_sent = _alert_history.get(fp)
    if last_sent is None:
        return False
    return (datetime.utcnow() - last_sent).total_seconds() < DEDUP_WINDOW_SECONDS


def _record_sent(fp: str) -> None:
    """Mark fingerprint as sent and evict stale entries."""
    cutoff = datetime.utcnow() - timedelta(seconds=DEDUP_WINDOW_SECONDS * 2)
    with _alert_history_lock:
        _alert_history[fp] = datetime.utcnow()
        # Evict old entries to prevent unbounded growth
        stale = [k for k, v in _alert_history.items() if v < cutoff]
        for k in stale:
            del _alert_history[k]


def send_alert(
    severity: str,
    category: str,
    description: str,
    rca: str = "",
    action_taken: str = "",
) -> bool:
    """
    Send a structured alert to Slack and/or MS Teams via incoming webhook.

    Returns True if at least one webhook delivery succeeded (or if the alert
    was logged locally as a simulated event).  Returns False when deduplicated.

    Webhook URLs are read from environment variables:
      SLACK_WEBHOOK_URL
      TEAMS_WEBHOOK_URL
    If neither is set, the alert is written to remediation_log.txt only.
    """
    fp = _fingerprint(category, description)

    if _is_duplicate(fp):
        _log_action(f"  Suppressed duplicate alert: [{category}]", "DEDUP")
        return False

    _record_sent(fp)

    ts            = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    sev_icon      = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🟢"}.get(severity, "⚪")

    # ── Slack Block Kit payload ────────────────────────────────────────────
    slack_payload = {
        "text": f"{sev_icon} *AI-DevOps Alert* | {severity} | {category}",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"AI-DevOps Alert: {severity} — {category}",
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Severity:* {sev_icon} {severity}"},
                    {"type": "mrkdwn", "text": f"*Time:* {ts}"},
                    {"type": "mrkdwn", "text": f"*Category:* {category}"},
                    {"type": "mrkdwn", "text": f"*Detail:* {description}"},
                ],
            },
        ],
    }
    if rca:
        slack_payload["blocks"].append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Root Cause:* {rca}"},
        })
    if action_taken:
        slack_payload["blocks"].append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Auto-Remediation:* {action_taken}"},
        })

    # ── MS Teams MessageCard payload ───────────────────────────────────────
    teams_payload = {
        "@type":       "MessageCard",
        "@context":    "https://schema.org/extensions",
        "summary":     f"AI-DevOps Alert: {severity} — {category}",
        "themeColor":  {"CRITICAL": "FF0000", "WARNING": "FFA500"}.get(severity, "00AA00"),
        "title":       f"AI-DevOps Alert: {severity} — {category}",
        "sections": [{
            "activityTitle": description,
            "facts": [
                {"name": "Severity",     "value": severity},
                {"name": "Category",     "value": category},
                {"name": "Time",         "value": ts},
                {"name": "Root Cause",   "value": rca          or "Analysing…"},
                {"name": "Auto-Action",  "value": action_taken or "None"},
            ],
        }],
    }

    delivered = False
    headers   = {"Content-Type": "application/json"}

    # Deliver to Slack
    if SLACK_WEBHOOK_URL and SLACK_WEBHOOK_URL.startswith("https://"):
        try:
            resp = requests.post(SLACK_WEBHOOK_URL, json=slack_payload,
                                 headers=headers, timeout=10)
            if resp.status_code == 200:
                _log_action(f"  Slack alert delivered: [{category}]", "ALERT")
                delivered = True
            else:
                _log_action(
                    f"  Slack delivery failed: HTTP {resp.status_code}", "WARNING"
                )
        except requests.RequestException as exc:
            _log_action(f"  Slack network error: {exc}", "WARNING")

    # Deliver to Teams
    if TEAMS_WEBHOOK_URL and TEAMS_WEBHOOK_URL.startswith("https://"):
        try:
            resp = requests.post(TEAMS_WEBHOOK_URL, json=teams_payload,
                                 headers=headers, timeout=10)
            if resp.status_code in (200, 202):
                _log_action(f"  Teams alert delivered: [{category}]", "ALERT")
                delivered = True
            else:
                _log_action(
                    f"  Teams delivery failed: HTTP {resp.status_code}", "WARNING"
                )
        except requests.RequestException as exc:
            _log_action(f"  Teams network error: {exc}", "WARNING")

    # Always write a local simulated alert entry
    _log_action(
        f"  [ALERT] {sev_icon} {severity} | {category} | {description}", "ALERT"
    )
    return True


# ===========================================================================
# Rules Engine
# ===========================================================================

def evaluate_rules(report: dict) -> list:
    """
    Inspect an analytics report dict and trigger appropriate actions.

    Rule table:
      R1  Isolation Forest flags latest snapshot as anomaly     → flush memory / escalate
      R2  Memory trend will breach 90 % within forecast window  → flush memory cache
      R3  Disk trend will breach 90 % within forecast window    → disk cleanup + rotate logs
      R4  Recurring log error cluster (≥3 occurrences)          → service restart or mem flush
      R5  Error rate in logs exceeds 20 %                       → critical alert

    Returns a list of dicts describing each triggered rule and the action taken.
    """
    triggered = []
    healing   = is_healing_enabled()

    anomaly   = report.get("anomaly_detection", {})
    forecasts = report.get("forecasts",         {})
    log_info  = report.get("log_analysis",      {})

    # ── Rule 1: Live anomaly ──────────────────────────────────────────────
    if anomaly.get("status") == "critical":
        cpu  = anomaly.get("latest_cpu",    0)
        mem  = anomaly.get("latest_memory", 0)
        desc = f"Isolation Forest anomaly — CPU={cpu}%, Memory={mem}%"
        rca  = f"Unsupervised model flagged abnormal CPU/Memory joint pattern"

        action = ""
        if healing:
            action = action_flush_memory_cache() if mem > 75 else (
                f"Escalated monitoring for anomaly (CPU={cpu}%)"
            )

        send_alert("CRITICAL", "Metric Anomaly", desc, rca=rca, action_taken=action)
        log_incident("CRITICAL", "Metric Anomaly", desc, rca=rca, action_taken=action)
        triggered.append({"rule": "R1_metric_anomaly", "action": action})

    # ── Rule 2: Memory forecast breach ───────────────────────────────────
    mem_fc = forecasts.get("memory", {})
    if mem_fc.get("will_breach"):
        mins    = mem_fc.get("minutes_to_breach", "?")
        current = mem_fc.get("current_value",     0)
        desc    = (f"Memory predicted to breach 90 % in ≈{mins} min "
                   f"(current {current}%)")

        action = ""
        if healing:
            action = action_flush_memory_cache()

        send_alert("WARNING", "Memory Forecast Breach", desc,
                   rca="Linear/exponential trend analysis", action_taken=action)
        log_incident("WARNING", "Memory Forecast", desc,
                     rca="Trend-based prediction", action_taken=action)
        triggered.append({"rule": "R2_memory_forecast", "action": action})

    # ── Rule 3: Disk forecast breach ─────────────────────────────────────
    disk_fc = forecasts.get("disk", {})
    if disk_fc.get("will_breach"):
        mins    = disk_fc.get("minutes_to_breach", "?")
        current = disk_fc.get("current_value",     0)
        desc    = (f"Disk predicted to breach 90 % in ≈{mins} min "
                   f"(current {current}%)")

        action = ""
        if healing:
            action = action_disk_cleanup() + " | " + action_rotate_logs()

        send_alert("WARNING", "Disk Forecast Breach", desc,
                   rca="Disk usage velocity trend", action_taken=action)
        log_incident("WARNING", "Disk Forecast", desc,
                     rca="Disk trend prediction", action_taken=action)
        triggered.append({"rule": "R3_disk_forecast", "action": action})

    # ── Rule 4: Recurring log error clusters ─────────────────────────────
    clusters = log_info.get("clusters", [])
    for cluster in clusters[:2]:   # evaluate top-2 clusters only
        if cluster.get("count", 0) < 3:
            continue

        root_cause = cluster.get("root_cause", "unknown error")
        rca_text   = cluster.get("rca", root_cause)
        services   = cluster.get("affected_services", [])
        primary_svc = services[0] if services else "unknown-service"
        svc_label   = ", ".join(services[:3])
        desc = (f"Recurring error in [{svc_label}]: '{root_cause}' "
                f"× {cluster['count']}")

        action = ""
        if healing:
            msg_lower = root_cause.lower()
            if "memory" in msg_lower or "oom" in msg_lower or "heap" in msg_lower:
                action = action_flush_memory_cache()
            else:
                action = action_restart_service(primary_svc)

        send_alert("WARNING", "Recurring Log Error", desc,
                   rca=rca_text, action_taken=action)
        log_incident("WARNING", "Log Error Pattern", desc,
                     rca=rca_text, action_taken=action)
        triggered.append({
            "rule":       "R4_recurring_error",
            "root_cause": root_cause,
            "action":     action,
        })

    # ── Rule 5: High error rate ───────────────────────────────────────────
    total_logs = log_info.get("total_count",  0)
    error_logs = log_info.get("error_count",  0)
    if total_logs > 0 and (error_logs / total_logs) > 0.20:
        rate = error_logs / total_logs
        desc = (f"Log error rate {rate:.1%} exceeds 20 % "
                f"({error_logs}/{total_logs} entries)")
        send_alert("CRITICAL", "High Error Rate", desc,
                   rca="Log sampling ratio exceeds critical threshold")
        log_incident("CRITICAL", "High Error Rate", desc,
                     rca=f"Error rate: {rate:.1%}")
        triggered.append({"rule": "R5_high_error_rate"})

    return triggered


# ===========================================================================
# Single-cycle runner and daemon loop
# ===========================================================================

def run_remediation_cycle() -> list:
    """Run one full analysis + rule evaluation cycle. Returns triggered rules."""
    _log_action("=== Remediation cycle start ===")
    try:
        report    = run_full_analysis()
        triggered = evaluate_rules(report)
        if triggered:
            _log_action(f"  {len(triggered)} rule(s) triggered")
        else:
            _log_action("  All clear — no rules triggered")
        return triggered
    except Exception as exc:
        _log_action(f"  Cycle error: {exc}", "ERROR")
        return []


def remediation_daemon(interval: int = 30) -> None:
    """
    Blocking daemon: run a remediation cycle every *interval* seconds.
    Designed to be started as a standalone process.
    """
    init_incident_db()
    print("=" * 60)
    print("  AI-DevOps Remediation Daemon  —  starting up")
    print("=" * 60)
    print(f"[Remediation] Cycle interval : {interval}s")
    print(f"[Remediation] Self-healing   : {'ENABLED' if is_healing_enabled() else 'DISABLED'}")
    print(f"[Remediation] Log file       : {REMEDIATION_LOG}")
    print()

    while True:
        try:
            run_remediation_cycle()
        except Exception as exc:
            _log_action(f"Daemon top-level error: {exc}", "ERROR")
        time.sleep(interval)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    remediation_daemon(interval=30)
