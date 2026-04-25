"""
db_loader.py
Loads generated CSV data into a local SQLite database
and runs the anomaly detection SQL queries, saving results to /reports.
"""

import sqlite3
import pandas as pd
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
SQL_DIR   = os.path.join(BASE_DIR, "sql")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DB_PATH   = os.path.join(DATA_DIR, "fraud_detection.db")

os.makedirs(REPORTS_DIR, exist_ok=True)


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn


def load_csv_to_db(conn: sqlite3.Connection):
    """Load all CSVs into SQLite tables."""
    tables = {
        "users":               os.path.join(DATA_DIR, "users.csv"),
        "transactions":        os.path.join(DATA_DIR, "transactions.csv"),
        "scored_transactions": os.path.join(DATA_DIR, "scored_transactions.csv"),
    }
    for table, path in tables.items():
        if not os.path.exists(path):
            print(f"[WARN] {path} not found — skipping {table}.")
            continue
        df = pd.read_csv(path)
        df.to_sql(table, conn, if_exists="replace", index=False)
        print(f"[DB] Loaded {len(df):,} rows → '{table}'")


def run_report_query(conn, query: str, report_name: str):
    """Execute a SELECT query and save result as CSV."""
    try:
        df = pd.read_sql_query(query, conn)
        out_path = os.path.join(REPORTS_DIR, f"{report_name}.csv")
        df.to_csv(out_path, index=False)
        print(f"[Report] {report_name}: {len(df):,} rows → {out_path}")
        return df
    except Exception as e:
        print(f"[ERROR] {report_name}: {e}")
        return pd.DataFrame()


def run_all_reports(conn):
    """Run all key reporting SQL queries and export to CSVs."""

    # Monthly fraud summary
    run_report_query(conn, """
        SELECT SUBSTR(txn_date,1,7) AS year_month,
               COUNT(*) AS total_txns,
               SUM(is_fraud) AS fraud_txns,
               ROUND(SUM(is_fraud)*100.0/COUNT(*),2) AS fraud_rate_pct,
               ROUND(SUM(CASE WHEN is_fraud=1 THEN amount ELSE 0 END),2) AS fraud_exposure
        FROM transactions GROUP BY SUBSTR(txn_date,1,7) ORDER BY year_month
    """, "monthly_fraud_summary")

    # Fraud by category
    run_report_query(conn, """
        SELECT merchant_category,
               COUNT(*) AS total_txns, SUM(is_fraud) AS fraud_txns,
               ROUND(SUM(is_fraud)*100.0/COUNT(*),2) AS fraud_rate_pct,
               ROUND(AVG(amount),2) AS avg_txn_amount,
               ROUND(SUM(CASE WHEN is_fraud=1 THEN amount ELSE 0 END),2) AS fraud_exposure
        FROM transactions GROUP BY merchant_category ORDER BY fraud_rate_pct DESC
    """, "fraud_by_category")

    # Fraud by country
    run_report_query(conn, """
        SELECT transaction_country,
               COUNT(*) AS total_txns, SUM(is_fraud) AS fraud_txns,
               ROUND(SUM(is_fraud)*100.0/COUNT(*),2) AS fraud_rate_pct,
               ROUND(SUM(CASE WHEN is_fraud=1 THEN amount ELSE 0 END),2) AS fraud_exposure
        FROM transactions GROUP BY transaction_country ORDER BY fraud_rate_pct DESC
    """, "fraud_by_country")

    # Risk tier distribution
    run_report_query(conn, """
        SELECT risk_tier, COUNT(*) AS txn_count,
               ROUND(SUM(amount),2) AS total_exposure,
               ROUND(AVG(amount),2) AS avg_amount,
               SUM(is_fraud) AS confirmed_frauds
        FROM scored_transactions GROUP BY risk_tier
        ORDER BY CASE risk_tier WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 WHEN 'MEDIUM' THEN 3 ELSE 4 END
    """, "risk_tier_distribution")

    # High-risk users
    run_report_query(conn, """
        SELECT s.user_id, COUNT(*) AS critical_txn_count,
               ROUND(SUM(s.amount),2) AS total_exposure,
               ROUND(AVG(s.composite_score),4) AS avg_risk_score,
               u.credit_score, u.account_age_days, u.is_flagged_account
        FROM scored_transactions s
        JOIN users u ON s.user_id = u.user_id
        WHERE s.risk_tier IN ('CRITICAL','HIGH')
        GROUP BY s.user_id ORDER BY critical_txn_count DESC LIMIT 100
    """, "high_risk_users")

    # Velocity anomalies
    run_report_query(conn, """
        SELECT user_id, txn_date, COUNT(*) AS txn_count,
               SUM(amount) AS total_amount
        FROM transactions GROUP BY user_id, txn_date
        HAVING COUNT(*) > 10 ORDER BY txn_count DESC
    """, "velocity_anomalies")

    # Hourly fraud heatmap
    run_report_query(conn, """
        SELECT CAST(SUBSTR(txn_time,1,2) AS INTEGER) AS hour_of_day,
               COUNT(*) AS total_txns, SUM(is_fraud) AS fraud_txns,
               ROUND(SUM(is_fraud)*100.0/COUNT(*),2) AS fraud_rate_pct
        FROM transactions
        GROUP BY CAST(SUBSTR(txn_time,1,2) AS INTEGER) ORDER BY hour_of_day
    """, "hourly_fraud_heatmap")

    # Top fraud transactions
    run_report_query(conn, """
        SELECT transaction_id, user_id, txn_datetime, amount,
               merchant_category, transaction_country, fraud_reason
        FROM transactions WHERE is_fraud=1 ORDER BY amount DESC LIMIT 50
    """, "top_fraud_transactions")


def generate_summary_stats(conn) -> dict:
    """Generate a JSON summary for the dashboard."""
    cur = conn.cursor()

    def q(sql):
        return cur.execute(sql).fetchone()[0]

    stats = {
        "total_transactions": q("SELECT COUNT(*) FROM transactions"),
        "total_fraud":        q("SELECT SUM(is_fraud) FROM transactions"),
        "fraud_rate_pct":     round(q("SELECT SUM(is_fraud)*100.0/COUNT(*) FROM transactions"), 3),
        "total_fraud_exposure": round(q("SELECT SUM(CASE WHEN is_fraud=1 THEN amount ELSE 0 END) FROM transactions"), 2),
        "critical_txns":      q("SELECT COUNT(*) FROM scored_transactions WHERE risk_tier='CRITICAL'"),
        "high_txns":          q("SELECT COUNT(*) FROM scored_transactions WHERE risk_tier='HIGH'"),
        "high_risk_users":    q("SELECT COUNT(DISTINCT user_id) FROM scored_transactions WHERE risk_tier IN ('CRITICAL','HIGH')"),
    }

    out = os.path.join(REPORTS_DIR, "summary_stats.json")
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[Summary] {stats}")
    return stats


def main():
    print("=== Fraud Detection DB Loader ===\n")
    conn = get_connection()

    print("── Loading CSVs into SQLite ──")
    load_csv_to_db(conn)
    conn.commit()

    print("\n── Running Reports ──")
    run_all_reports(conn)

    print("\n── Generating Summary Stats ──")
    generate_summary_stats(conn)

    conn.close()
    print(f"\n✓ All done. DB: {DB_PATH} | Reports: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
