"""
collector.py - Simulated Ingestion & Data Collector
=====================================================
Background worker #1: Collects real system metrics via psutil every 5 seconds
                       and writes them to a local SQLite database.

Background worker #2: Continuously generates realistic JSON-formatted application
                       log lines, randomly injecting error anomalies every few minutes.

Run standalone:
    python collector.py
"""

import os
import sys
import json
import time
import random
import sqlite3
import threading
import platform
from datetime import datetime

import psutil

# ---------------------------------------------------------------------------
# Path configuration — all data lives under a 'data/' sub-folder next to
# this script, so the tool works regardless of the current working directory.
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
DB_PATH   = os.path.join(DATA_DIR, "metrics.db")
LOG_PATH  = os.path.join(DATA_DIR, "app.log")

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
COLLECTION_INTERVAL   = 5    # seconds between metric snapshots
LOG_LINE_INTERVAL     = 3    # seconds between synthetic log lines
ANOMALY_PROBABILITY   = 0.05 # 5 % base chance a log line is an error

# Root disk path differs between Windows and Linux/macOS
DISK_PATH = "C:\\" if platform.system() == "Windows" else "/"

# Synthetic application topology
SERVICES = [
    "auth-service",
    "payment-service",
    "api-gateway",
    "user-service",
    "notification-service",
]

# Weighted level list so INFO dominates
LOG_LEVELS_NORMAL = ["INFO", "INFO", "INFO", "DEBUG", "WARNING"]

NORMAL_MESSAGES = [
    "Request processed successfully in {}ms",
    "Cache hit for key user:{}",
    "User {} authenticated via OAuth2",
    "Health check passed — all upstreams reachable",
    "Database query executed in {}ms",
    "Message published to queue (partition {})",
    "Configuration hot-reload completed",
    "Session created for user:{}; TTL=3600s",
    "Rate-limit counter reset for IP 10.0.0.{}",
    "Outbound HTTP 200 to downstream service in {}ms",
]

# Anomaly patterns — duplicates intentionally weight certain errors higher
ANOMALY_MESSAGES = [
    "Error 500: Database connection timeout after 30s",
    "Error 500: Database connection timeout after 30s",   # higher weight
    "Out of Memory Exception: Java heap space exhausted",
    "Out of Memory Exception: GC overhead limit exceeded",
    "CRITICAL: Service unresponsive after 3 consecutive retries",
    "Error 500: Upstream gateway returned 503 Service Unavailable",
]


# ===========================================================================
# Database helpers
# ===========================================================================

def init_db() -> None:
    """Create the data directory and the metrics table if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                cpu_percent     REAL,
                memory_percent  REAL,
                memory_used_mb  REAL,
                disk_percent    REAL,
                disk_read_mb    REAL,
                disk_write_mb   REAL,
                net_sent_mb     REAL,
                net_recv_mb     REAL
            )
        """)
        conn.commit()
    finally:
        conn.close()
    print(f"[Collector] SQLite DB ready → {DB_PATH}")


def insert_metrics(row: dict) -> None:
    """Insert a single metrics row; each call opens its own connection for thread-safety."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("""
            INSERT INTO metrics
                (timestamp, cpu_percent, memory_percent, memory_used_mb,
                 disk_percent, disk_read_mb, disk_write_mb, net_sent_mb, net_recv_mb)
            VALUES
                (:timestamp, :cpu_percent, :memory_percent, :memory_used_mb,
                 :disk_percent, :disk_read_mb, :disk_write_mb, :net_sent_mb, :net_recv_mb)
        """, row)
        conn.commit()
    finally:
        conn.close()


# ===========================================================================
# Metrics collection
# ===========================================================================

def _safe_disk_io_delta() -> tuple:
    """
    Returns (read_mb, write_mb) deltas over a 1-second window.
    Returns (0.0, 0.0) safely if disk I/O counters are unavailable
    (e.g., certain VM configurations).
    """
    try:
        before = psutil.disk_io_counters()
        time.sleep(1)
        after  = psutil.disk_io_counters()
        if before is None or after is None:
            return 0.0, 0.0
        read_mb  = (after.read_bytes  - before.read_bytes)  / 1_048_576
        write_mb = (after.write_bytes - before.write_bytes) / 1_048_576
        return max(read_mb, 0.0), max(write_mb, 0.0)
    except Exception:
        time.sleep(1)   # keep the 1-second cadence even on error
        return 0.0, 0.0


def _safe_net_io_delta() -> tuple:
    """Returns (sent_mb, recv_mb) deltas. Falls back to 0.0 on any error."""
    try:
        before = psutil.net_io_counters()
        after  = psutil.net_io_counters()   # called immediately after disk delta
        if before is None or after is None:
            return 0.0, 0.0
        sent_mb = (after.bytes_sent - before.bytes_sent) / 1_048_576
        recv_mb = (after.bytes_recv - before.bytes_recv) / 1_048_576
        return max(sent_mb, 0.0), max(recv_mb, 0.0)
    except Exception:
        return 0.0, 0.0


def collect_one_snapshot() -> None:
    """Collect a single metrics snapshot and persist it to SQLite."""
    # Disk I/O delta includes an internal 1-second sleep for measurement
    disk_read_mb, disk_write_mb = _safe_disk_io_delta()
    net_before = psutil.net_io_counters()

    # CPU measured over the 1-second window that already elapsed inside disk delta
    cpu_pct = psutil.cpu_percent(interval=None)

    # Memory
    mem       = psutil.virtual_memory()
    mem_pct   = mem.percent
    mem_used  = mem.used / 1_048_576

    # Disk usage
    try:
        disk_pct = psutil.disk_usage(DISK_PATH).percent
    except Exception:
        disk_pct = 0.0

    # Network delta — compare now vs. the snapshot taken before disk I/O
    try:
        net_after   = psutil.net_io_counters()
        net_sent_mb = max((net_after.bytes_sent - net_before.bytes_sent) / 1_048_576, 0.0)
        net_recv_mb = max((net_after.bytes_recv - net_before.bytes_recv) / 1_048_576, 0.0)
    except Exception:
        net_sent_mb = 0.0
        net_recv_mb = 0.0

    row = {
        "timestamp":      datetime.utcnow().isoformat(),
        "cpu_percent":    round(cpu_pct,    2),
        "memory_percent": round(mem_pct,    2),
        "memory_used_mb": round(mem_used,   2),
        "disk_percent":   round(disk_pct,   2),
        "disk_read_mb":   round(disk_read_mb,  4),
        "disk_write_mb":  round(disk_write_mb, 4),
        "net_sent_mb":    round(net_sent_mb,   4),
        "net_recv_mb":    round(net_recv_mb,   4),
    }
    insert_metrics(row)


def metrics_worker() -> None:
    """
    Background thread: collect one snapshot every COLLECTION_INTERVAL seconds.
    The function itself sleeps 1 second inside collect_one_snapshot() for I/O
    delta measurement, so the outer sleep is reduced accordingly.
    """
    print(f"[Collector] Metrics worker started (interval: {COLLECTION_INTERVAL}s)")
    while True:
        try:
            collect_one_snapshot()
        except Exception as exc:
            print(f"[Collector][ERROR] Metrics snapshot failed: {exc}")
        # COLLECTION_INTERVAL minus the 1 s already consumed by disk I/O delta
        time.sleep(max(COLLECTION_INTERVAL - 1, 1))


# ===========================================================================
# Synthetic log generation
# ===========================================================================

def _build_log_line(is_anomaly: bool) -> str:
    """Build and return one JSON-formatted log line as a string."""
    service = random.choice(SERVICES)

    if is_anomaly:
        level   = "ERROR"
        message = random.choice(ANOMALY_MESSAGES)
    else:
        level   = random.choice(LOG_LEVELS_NORMAL)
        tmpl    = random.choice(NORMAL_MESSAGES)
        # Fill up to two {} placeholders with random integers
        message = tmpl.replace("{}", str(random.randint(1, 9999)), 1)
        message = message.replace("{}", str(random.randint(1, 9999)), 1)

    entry = {
        "timestamp":    datetime.utcnow().isoformat() + "Z",
        "log_level":    level,
        "service_name": service,
        "message":      message,
        "trace_id":     f"trace-{random.randint(100_000, 999_999)}",
        "host":         f"node-{random.randint(1, 5):02d}",
    }
    return json.dumps(entry)


def log_worker() -> None:
    """
    Background thread: append realistic JSON log lines to app.log.
    Every ~2 minutes a small burst of consecutive error lines is injected
    to simulate a real outage pattern that the NLP engine can cluster.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"[Collector] Log worker started → {LOG_PATH}")

    burst_remaining = 0   # lines left in current error burst

    while True:
        try:
            # Decide whether this line is an anomaly
            if burst_remaining > 0:
                is_anomaly     = True
                burst_remaining -= 1
            elif random.random() < ANOMALY_PROBABILITY:
                is_anomaly      = True
                burst_remaining = random.randint(2, 6)  # start a burst
            else:
                is_anomaly = False

            line = _build_log_line(is_anomaly)

            with open(LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

        except Exception as exc:
            print(f"[Collector][ERROR] Log write failed: {exc}")

        time.sleep(LOG_LINE_INTERVAL)


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    """Initialise storage, start both worker threads, then block until Ctrl-C."""
    print("=" * 60)
    print("  AI-DevOps Collector  —  starting up")
    print("=" * 60)

    init_db()

    t_metrics = threading.Thread(
        target=metrics_worker, daemon=True, name="MetricsWorker"
    )
    t_logs = threading.Thread(
        target=log_worker, daemon=True, name="LogWorker"
    )

    t_metrics.start()
    t_logs.start()

    print("[Collector] Both workers running.  Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Collector] Shutdown requested — exiting cleanly.")
        sys.exit(0)


if __name__ == "__main__":
    main()
