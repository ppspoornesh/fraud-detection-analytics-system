# 🛡️ Fraud Detection Analytics System

An end-to-end fraud detection and monitoring platform that identifies anomalous financial transactions and high-risk user patterns using rule-based heuristics, machine learning, and interactive dashboards.

---

## 📌 Project Overview

| Component | Description |
|-----------|-------------|
| **Data Pipeline** | Synthetic financial dataset generation + cleaning + validation |
| **Feature Engineering** | 14+ risk signals: velocity, z-score, time anomalies, country/category risk |
| **Risk Scoring** | Composite score from Rule Engine + Random Forest + Isolation Forest |
| **SQL Reporting** | Anomaly detection queries, high-risk user identification, trend reports |
| **Dashboard** | 5-page Streamlit app with Plotly charts, filters, and CSV exports |

---

## 🗂️ Project Structure

```
fraud-detection-analytics-system/
│
├── data/                          # Generated CSVs and SQLite DB
│   ├── users.csv
│   ├── transactions.csv           # 50,000 synthetic transactions
│   ├── scored_transactions.csv    # With risk scores + tiers
│   └── fraud_detection.db        # SQLite database
│
├── src/
│   ├── generate_data.py           # Synthetic data generation
│   ├── preprocessing.py           # Cleaning, validation, feature engineering
│   ├── risk_scoring.py            # Risk models + composite scoring
│   └── db_loader.py               # SQLite loader + report export
│
├── sql/
│   └── anomaly_detection.sql      # All SQL queries (DDL + reports + alerts)
│
├── dashboard/
│   └── app.py                     # Streamlit interactive dashboard
│
├── reports/                       # Auto-generated CSV reports
│   ├── monthly_fraud_summary.csv
│   ├── fraud_by_category.csv
│   ├── fraud_by_country.csv
│   ├── risk_tier_distribution.csv
│   ├── high_risk_users.csv
│   ├── hourly_fraud_heatmap.csv
│   ├── top_fraud_transactions.csv
│   └── summary_stats.json
│
├── run_pipeline.py                # ← Master runner (start here)
└── requirements.txt
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/fraud-detection-analytics-system.git
cd fraud-detection-analytics-system
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Option A: Full pipeline in one command
```bash
python run_pipeline.py
```
This runs all 4 steps automatically:
1. Generates 50,000 synthetic transactions (500 users)
2. Cleans, validates, and engineers features
3. Trains and applies risk scoring models
4. Loads data into SQLite and exports all reports

### Option B: Step by step
```bash
python src/generate_data.py    # Step 1: Generate data
python src/risk_scoring.py     # Step 2: Score transactions
python src/db_loader.py        # Step 3: Load DB + export reports
```

### Launch the dashboard
```bash
streamlit run dashboard/app.py
```
Opens at `http://localhost:8501`

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| **Executive Overview** | KPI cards, risk tier donut, monthly trend, category/country fraud maps |
| **Transaction Monitor** | Filterable scatter plot + high-risk transaction table with CSV export |
| **User Risk Profiles** | Credit score vs risk scatter, account age analysis, user ID lookup |
| **Trend Analysis** | Hourly fraud heatmap, device-type breakdown, monthly exposure area chart |
| **Model Insights** | Score distributions, rule vs ML score comparison, isolation flag overlap |

---

## 🤖 Scoring Architecture

### Feature Engineering (14 signals)
| Feature | Description |
|---------|-------------|
| `amount_z_score` | Standard deviations from user's personal spending mean |
| `txn_count_24h` | Rolling 24-hour transaction velocity |
| `txn_amt_24h` | Rolling 24-hour spend velocity |
| `is_odd_hour` | Transactions between 10 PM – 6 AM |
| `is_high_risk_category` | Wire Transfer, Crypto, Gambling, ATM, Jewelry |
| `is_high_risk_country` | Nigeria, Russia, China |
| `ip_flag` | Suspicious IP indicator |
| `is_flagged_account` | Compliance-flagged user |
| `credit_score` | User credit score (300–850) |
| `account_age_days` | Account seniority |
| `spend_ratio` | Amount / user's avg monthly spend |

### Composite Score (weighted ensemble)
```
composite_score = 0.35 × rule_score  +  0.45 × ml_score  +  0.20 × isolation_score
```

### Risk Tiers
| Tier | Score Range | Action |
|------|-------------|--------|
| 🟢 LOW | 0.00 – 0.30 | Auto-approve |
| 🟡 MEDIUM | 0.30 – 0.55 | Soft review |
| 🟠 HIGH | 0.55 – 0.75 | Manual review |
| 🔴 CRITICAL | 0.75 – 1.00 | Block + escalate |

---

## 🗄️ Key SQL Queries

Located in `sql/anomaly_detection.sql`:

- **Anomaly detection**: high-risk country + large amount, odd-hour + high amount, 24h velocity
- **High-risk user identification**: CRITICAL tier aggregations, low credit + flagged accounts
- **Trend reports**: Monthly fraud summary, category breakdown, country map, hourly heatmap
- **Alert system**: Auto-insert CRITICAL transactions into `fraud_alerts` table

---

## 📈 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Transactions | 50,000 |
| Total Users | 500 |
| Fraud Rate | ~2.2% |
| Fraud Scenarios | High-risk country, flagged user, odd-hour, velocity |
| Date Range | Jan 2023 – Dec 2024 |

---

## 🛠️ Tech Stack

- **Python**: pandas, NumPy, scikit-learn
- **ML Models**: Random Forest Classifier, Isolation Forest
- **Database**: SQLite (via Python `sqlite3`)
- **Dashboard**: Streamlit + Plotly
- **SQL**: SQLite-compatible (portable to PostgreSQL/MySQL)

---

## 📄 License

MIT License — free to use, modify, and distribute.
