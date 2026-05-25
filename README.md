cat > README.md << 'EOF'
# AI-Driven DevOps Monitoring & Self-Healing Tool

A fully Python-based, zero-infrastructure monitoring platform running entirely in a Python virtual environment — no Docker, no databases, no admin rights required.

---

## Project Structure

\`\`\`
ai-devops-monitor/
├── collector.py        ← psutil metrics + synthetic log generator
├── analytics.py        ← AI engine (Isolation Forest, forecasting, TF-IDF NLP)
├── remediation.py      ← Rules engine, self-healing, webhook alerts
├── app.py              ← Streamlit dashboard
├── requirements.txt    ← pip-only dependencies
└── data/               ← Auto-created at runtime
    ├── metrics.db      # SQLite metrics
    ├── incidents.db    # SQLite incidents
    └── app.log         # JSON application logs
\`\`\`

---

## Quick Start (GitHub Codespaces)

\`\`\`bash
# Step 1 — Install dependencies
pip install -r requirements.txt

# Step 2 — Terminal 1: Start collector
python collector.py

# Step 3 — Terminal 2: Start remediation daemon
python remediation.py

# Step 4 — Terminal 3: Launch dashboard
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
\`\`\`

Open the forwarded port 8501 in your browser.

---

## AI/ML Components

| Component | Technology | Purpose |
|---|---|---|
| Anomaly Detection | IsolationForest (scikit-learn) | Flags abnormal CPU/Memory/Disk patterns |
| Forecasting | LinearRegression + SimpleExpSmoothing | Predicts minutes-to-breach per metric |
| Log Clustering | TF-IDF + KMeans (scikit-learn) | Groups similar errors, extracts RCA |
| RCA Extraction | Regex patterns | Converts raw errors to structured root cause |

---

## Self-Healing Rules

| Rule | Trigger | Action |
|---|---|---|
| R1 | Isolation Forest anomaly | Flush memory cache |
| R2 | Memory breach predicted | Flush memory cache |
| R3 | Disk breach predicted | Disk cleanup + log rotation |
| R4 | Recurring error cluster (≥3) | Service restart or memory flush |
| R5 | Error rate > 20% | Critical alert |

All actions are **fully simulated** — no real system resources are modified.

---

## Slack / Teams Alerts (Optional)

\`\`\`bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
export TEAMS_WEBHOOK_URL="https://your-org.webhook.office.com/..."
\`\`\`

If not set, alerts are written to \`remediation_log.txt\` locally.

---

## Tech Stack

- **Python 3.9+** — no admin install required
- **psutil** — real system metrics
- **scikit-learn** — Isolation Forest, TF-IDF, KMeans
- **statsmodels** — Exponential Smoothing forecasts
- **Streamlit + Plotly** — interactive dark-theme dashboard
- **SQLite** — zero-config structured storage
- **requests** — webhook delivery
EOF
