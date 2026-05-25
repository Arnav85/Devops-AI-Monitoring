# AI-Driven DevOps Monitoring & Self-Healing Tool

A fully Python-based, zero-infrastructure monitoring platform.  
No Docker, no databases, no admin rights required — runs entirely inside a standard `venv`.

---

## Project Structure

```
AI-Devops/
├── collector.py        # Worker 1: real system metrics (psutil → SQLite)
│                       # Worker 2: synthetic JSON app-log with injected errors
├── analytics.py        # AI engine: Isolation Forest · forecasting · TF-IDF log NLP
├── remediation.py      # Rules engine · mock self-healing · Slack/Teams webhooks
├── app.py              # Streamlit dashboard (real-time charts, incidents, RCA)
├── requirements.txt    # pip-installable dependencies only
├── remediation_log.txt # Auto-created: all remediation/alert audit entries
└── data/               # Auto-created at runtime
    ├── metrics.db      # SQLite — raw system metrics (psutil snapshots)
    ├── incidents.db    # SQLite — triggered incidents & auto-actions
    ├── app.log         # JSON application logs (synthetic + injected errors)
    ├── healing_enabled.flag   # Presence = self-healing ON
    ├── temp_logs/      # Dummy temp files (used by rotate-logs action)
    └── log_archive/    # Dummy archive markers (used by rotate-logs action)
```

---

## Quick-Start

### Step 1 — Create & activate a virtual environment

```powershell
# Windows PowerShell
python -m venv venv
venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> Installation takes ~2-3 minutes; no admin rights needed.

---

### Step 3 — Run the three components

Open **three separate terminals**, all with the venv activated and the project folder as CWD.

**Terminal 1 — Data Collector** *(start this first)*

```bash
python collector.py
```

Starts two background threads:
- Collects real CPU / Memory / Disk / Network metrics every **5 seconds** → `data/metrics.db`
- Generates synthetic JSON log lines every **3 seconds** → `data/app.log`  
  Errors are injected randomly (~5 % probability, in short bursts) to trigger the AI engine.

---

**Terminal 2 — Remediation Daemon** *(start after ~30 s of data)*

```bash
python remediation.py
```

Runs an analysis + rule evaluation cycle every **30 seconds**.  
Self-healing is **disabled** by default; enable it from the dashboard sidebar.

---

**Terminal 3 — Dashboard**

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**  
The dashboard auto-refreshes every 10 seconds (configurable via sidebar).

---

## Dashboard Feature Map

| Panel | Description |
|---|---|
| **Status Banner** | System health (Healthy / Anomaly) and self-healing toggle indicator |
| **KPI Cards** | Live CPU · Memory · Disk · Network with delta vs. 10-point rolling average |
| **Metric Charts** | Plotly dark-theme time-series; red ✕ markers appear at AI-detected anomalies |
| **Anomaly Score** | Isolation Forest decision score sparkline — values below 0 = anomalous |
| **Forecast Panel** | Per-metric mini-charts showing predicted trajectory; warns when breach is imminent |
| **Log Clusters** | TF-IDF + KMeans groups similar error messages; surfaces extracted root-cause string |
| **Incident History** | Filterable table of every triggered rule with RCA text and auto-action taken; CSV export |
| **Recent Incidents** | Raw ERROR/CRITICAL log entries from the last hour with RCA annotation |
| **Sidebar** | Enable/disable self-healing · manual action buttons · refresh controls |

---

## Enabling Slack / Teams Alerts

Set the relevant environment variable **before** starting `remediation.py`:

```powershell
# Windows PowerShell
$env:SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/T.../B.../..."
$env:TEAMS_WEBHOOK_URL = "https://your-org.webhook.office.com/webhookb2/..."
```

```bash
# Linux / macOS
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/T.../B.../..."
export TEAMS_WEBHOOK_URL="https://your-org.webhook.office.com/webhookb2/..."
```

If neither variable is set, alerts are written locally to `remediation_log.txt`.  
Duplicate alerts within a **5-minute window** are automatically suppressed.

---

## AI / ML Components

| Component | Technology | Purpose |
|---|---|---|
| Anomaly detection | `IsolationForest` (scikit-learn) | Flags abnormal CPU/Memory/Disk combinations |
| Threshold forecasting | `LinearRegression` + optional `SimpleExpSmoothing` (statsmodels) | Predicts minutes-to-breach for each metric |
| Log clustering | `TfidfVectorizer` + `KMeans` (scikit-learn) | Groups similar errors, extracts root-cause string |
| RCA extraction | Regex pattern matching | Converts raw error text to structured RCA strings |

> The Isolation Forest requires **≥ 20 data points** (~2 minutes of collection) before training.  
> The model retrains automatically on each analysis cycle using the latest 200 snapshots.

---

## Self-Healing Rules

| Rule | Trigger Condition | Simulated Action |
|---|---|---|
| R1 | Isolation Forest flags current snapshot as anomaly | Flush memory cache (if Mem > 75%) |
| R2 | Memory predicted to breach 90% within 1 minute | Flush memory cache |
| R3 | Disk predicted to breach 90% within 1 minute | Disk cleanup + log rotation |
| R4 | Recurring error cluster with ≥ 3 occurrences | Service restart or memory flush |
| R5 | Log error rate exceeds 20% | CRITICAL alert (no auto-action) |

All remediation actions are **fully simulated** — they write audit entries to `remediation_log.txt` and operate only inside the project's `data/` directory.  No real processes or system resources are modified.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` | Ensure venv is activated and `pip install -r requirements.txt` completed |
| Charts empty / "Waiting for metrics…" | Start `collector.py` first and wait ~30 s |
| "Need ≥ 20 data points" in anomaly panel | Wait ~2 minutes for the collector to accumulate enough rows |
| Dashboard not opening | Check that port 8501 is free; use `streamlit run app.py --server.port 8502` to change it |
| Disk I/O shows 0.0 | Normal on some VMs — the collector handles this gracefully |
| `statsmodels` import warning | Safe to ignore; forecasting falls back to `LinearRegression` automatically |
