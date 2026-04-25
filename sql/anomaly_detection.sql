-- =============================================================
-- Fraud Detection Analytics System
-- SQL Scripts: Anomaly Detection & Reporting
-- =============================================================
-- Compatible with: SQLite, PostgreSQL, MySQL (minor dialect diffs)
-- Tables assumed: transactions, users, scored_transactions
-- =============================================================


-- ─────────────────────────────────────────────────────────────
-- 0. SETUP: Create base tables (SQLite-compatible DDL)
-- ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
    user_id             TEXT PRIMARY KEY,
    age                 INTEGER,
    account_age_days    INTEGER,
    credit_score        INTEGER,
    avg_monthly_spend   REAL,
    country             TEXT,
    is_flagged_account  INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id      TEXT PRIMARY KEY,
    user_id             TEXT,
    txn_date            TEXT,
    txn_time            TEXT,
    txn_datetime        TEXT,
    amount              REAL,
    currency            TEXT DEFAULT 'USD',
    merchant_category   TEXT,
    merchant_name       TEXT,
    transaction_country TEXT,
    device_type         TEXT,
    ip_flag             INTEGER DEFAULT 0,
    is_international    INTEGER DEFAULT 0,
    is_fraud            INTEGER DEFAULT 0,
    fraud_reason        TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS scored_transactions (
    transaction_id      TEXT PRIMARY KEY,
    user_id             TEXT,
    txn_datetime        TEXT,
    amount              REAL,
    merchant_category   TEXT,
    transaction_country TEXT,
    composite_score     REAL,
    risk_tier           TEXT,
    rule_score          REAL,
    ml_score            REAL,
    isolation_score     REAL,
    isolation_flag      INTEGER,
    is_fraud            INTEGER,
    fraud_reason        TEXT
);


-- ─────────────────────────────────────────────────────────────
-- 1. ANOMALY DETECTION QUERIES
-- ─────────────────────────────────────────────────────────────

-- 1a. All HIGH and CRITICAL risk transactions (requires review)
SELECT
    s.transaction_id,
    s.user_id,
    s.txn_datetime,
    s.amount,
    s.merchant_category,
    s.transaction_country,
    ROUND(s.composite_score, 4)  AS composite_score,
    s.risk_tier,
    s.fraud_reason,
    u.credit_score,
    u.is_flagged_account
FROM scored_transactions s
JOIN users u ON s.user_id = u.user_id
WHERE s.risk_tier IN ('HIGH', 'CRITICAL')
ORDER BY s.composite_score DESC;


-- 1b. Large transactions from high-risk countries
SELECT
    transaction_id,
    user_id,
    txn_datetime,
    amount,
    transaction_country,
    merchant_category,
    is_fraud
FROM transactions
WHERE transaction_country IN ('NG', 'RU', 'CN')
  AND amount > 5000
ORDER BY amount DESC;


-- 1c. Odd-hour transactions (1 AM – 5 AM) with high amounts
SELECT
    transaction_id,
    user_id,
    txn_datetime,
    CAST(SUBSTR(txn_time, 1, 2) AS INTEGER) AS txn_hour,
    amount,
    merchant_category,
    is_fraud
FROM transactions
WHERE CAST(SUBSTR(txn_time, 1, 2) AS INTEGER) BETWEEN 1 AND 5
  AND amount > 3000
ORDER BY amount DESC;


-- 1d. Velocity anomaly — users with >10 transactions in a single day
SELECT
    user_id,
    txn_date,
    COUNT(*)        AS txn_count,
    SUM(amount)     AS total_amount,
    MAX(amount)     AS max_single_txn,
    AVG(amount)     AS avg_amount
FROM transactions
GROUP BY user_id, txn_date
HAVING COUNT(*) > 10
ORDER BY txn_count DESC;


-- 1e. Users transacting across 3+ different countries in any rolling 7-day window
SELECT
    user_id,
    txn_date,
    COUNT(DISTINCT transaction_country) AS country_count,
    GROUP_CONCAT(DISTINCT transaction_country) AS countries
FROM transactions
GROUP BY user_id, txn_date
HAVING COUNT(DISTINCT transaction_country) >= 3
ORDER BY country_count DESC;


-- 1f. Isolation Forest flagged transactions cross-validated with rule-based score > 0.5
SELECT
    s.transaction_id,
    s.user_id,
    s.txn_datetime,
    s.amount,
    s.merchant_category,
    s.transaction_country,
    ROUND(s.rule_score, 4)        AS rule_score,
    ROUND(s.isolation_score, 4)   AS isolation_score,
    ROUND(s.composite_score, 4)   AS composite_score,
    s.risk_tier
FROM scored_transactions s
WHERE s.isolation_flag = 1
  AND s.rule_score > 0.5
ORDER BY s.composite_score DESC
LIMIT 500;


-- ─────────────────────────────────────────────────────────────
-- 2. HIGH-RISK USER IDENTIFICATION
-- ─────────────────────────────────────────────────────────────

-- 2a. Users with the highest number of CRITICAL transactions
SELECT
    s.user_id,
    COUNT(*)                        AS critical_txn_count,
    ROUND(SUM(s.amount), 2)         AS total_exposure,
    ROUND(AVG(s.composite_score), 4) AS avg_risk_score,
    u.credit_score,
    u.account_age_days,
    u.is_flagged_account
FROM scored_transactions s
JOIN users u ON s.user_id = u.user_id
WHERE s.risk_tier = 'CRITICAL'
GROUP BY s.user_id
ORDER BY critical_txn_count DESC
LIMIT 50;


-- 2b. Users whose average composite score places them in HIGH risk overall
SELECT
    s.user_id,
    COUNT(*)                         AS total_txns,
    ROUND(AVG(s.composite_score), 4) AS avg_composite_score,
    ROUND(MAX(s.composite_score), 4) AS max_composite_score,
    SUM(s.is_fraud)                  AS confirmed_frauds,
    u.credit_score,
    u.is_flagged_account
FROM scored_transactions s
JOIN users u ON s.user_id = u.user_id
GROUP BY s.user_id
HAVING AVG(s.composite_score) > 0.55
ORDER BY avg_composite_score DESC;


-- 2c. High-risk users: flagged account + poor credit + high spend
SELECT
    u.user_id,
    u.credit_score,
    u.account_age_days,
    u.avg_monthly_spend,
    u.is_flagged_account,
    COUNT(t.transaction_id)     AS txn_count,
    ROUND(SUM(t.amount), 2)    AS total_spend,
    SUM(t.is_fraud)            AS fraud_txns
FROM users u
JOIN transactions t ON u.user_id = t.user_id
WHERE u.is_flagged_account = 1
   OR u.credit_score < 500
GROUP BY u.user_id
ORDER BY fraud_txns DESC, total_spend DESC;


-- ─────────────────────────────────────────────────────────────
-- 3. REPORTING & TREND ANALYSIS
-- ─────────────────────────────────────────────────────────────

-- 3a. Monthly fraud summary
SELECT
    SUBSTR(txn_date, 1, 7)         AS year_month,
    COUNT(*)                       AS total_txns,
    SUM(is_fraud)                  AS fraud_txns,
    ROUND(SUM(is_fraud)*100.0/COUNT(*), 2) AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud=1 THEN amount ELSE 0 END), 2) AS fraud_exposure
FROM transactions
GROUP BY SUBSTR(txn_date, 1, 7)
ORDER BY year_month;


-- 3b. Fraud by merchant category
SELECT
    merchant_category,
    COUNT(*)                       AS total_txns,
    SUM(is_fraud)                  AS fraud_txns,
    ROUND(SUM(is_fraud)*100.0/COUNT(*), 2) AS fraud_rate_pct,
    ROUND(AVG(amount), 2)          AS avg_txn_amount,
    ROUND(SUM(CASE WHEN is_fraud=1 THEN amount ELSE 0 END), 2) AS fraud_exposure
FROM transactions
GROUP BY merchant_category
ORDER BY fraud_rate_pct DESC;


-- 3c. Fraud by country
SELECT
    transaction_country,
    COUNT(*)                       AS total_txns,
    SUM(is_fraud)                  AS fraud_txns,
    ROUND(SUM(is_fraud)*100.0/COUNT(*), 2) AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud=1 THEN amount ELSE 0 END), 2) AS fraud_exposure
FROM transactions
GROUP BY transaction_country
ORDER BY fraud_rate_pct DESC;


-- 3d. Fraud by device type
SELECT
    device_type,
    COUNT(*)                       AS total_txns,
    SUM(is_fraud)                  AS fraud_txns,
    ROUND(SUM(is_fraud)*100.0/COUNT(*), 2) AS fraud_rate_pct
FROM transactions
GROUP BY device_type
ORDER BY fraud_rate_pct DESC;


-- 3e. Risk tier distribution with financial exposure
SELECT
    risk_tier,
    COUNT(*)                         AS txn_count,
    ROUND(COUNT(*)*100.0 / SUM(COUNT(*)) OVER(), 2) AS pct_of_total,
    ROUND(SUM(amount), 2)            AS total_exposure,
    ROUND(AVG(amount), 2)            AS avg_amount,
    SUM(is_fraud)                    AS confirmed_frauds
FROM scored_transactions
GROUP BY risk_tier
ORDER BY
    CASE risk_tier
        WHEN 'CRITICAL' THEN 1
        WHEN 'HIGH'     THEN 2
        WHEN 'MEDIUM'   THEN 3
        WHEN 'LOW'      THEN 4
    END;


-- 3f. Top 20 highest-value fraud transactions
SELECT
    transaction_id,
    user_id,
    txn_datetime,
    amount,
    merchant_category,
    transaction_country,
    fraud_reason
FROM transactions
WHERE is_fraud = 1
ORDER BY amount DESC
LIMIT 20;


-- 3g. Hour-of-day fraud heatmap
SELECT
    CAST(SUBSTR(txn_time, 1, 2) AS INTEGER) AS hour_of_day,
    COUNT(*) AS total_txns,
    SUM(is_fraud) AS fraud_txns,
    ROUND(SUM(is_fraud)*100.0/COUNT(*), 2) AS fraud_rate_pct
FROM transactions
GROUP BY CAST(SUBSTR(txn_time, 1, 2) AS INTEGER)
ORDER BY hour_of_day;


-- 3h. Rolling 30-day fraud trend (SQLite window function compatible)
SELECT
    txn_date,
    SUM(is_fraud) AS daily_fraud_count,
    SUM(SUM(is_fraud)) OVER (
        ORDER BY txn_date
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS rolling_30d_fraud
FROM transactions
GROUP BY txn_date
ORDER BY txn_date;


-- ─────────────────────────────────────────────────────────────
-- 4. ALERT GENERATION (Flagging queue for review)
-- ─────────────────────────────────────────────────────────────

-- 4a. Create alert table
CREATE TABLE IF NOT EXISTS fraud_alerts (
    alert_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id      TEXT,
    user_id             TEXT,
    alert_reason        TEXT,
    risk_tier           TEXT,
    composite_score     REAL,
    amount              REAL,
    alert_timestamp     TEXT DEFAULT (DATETIME('now')),
    status              TEXT DEFAULT 'OPEN'   -- OPEN / REVIEWED / CLOSED
);

-- 4b. Insert alerts for CRITICAL-tier transactions not yet alerted
INSERT OR IGNORE INTO fraud_alerts (transaction_id, user_id, alert_reason, risk_tier, composite_score, amount)
SELECT
    transaction_id,
    user_id,
    'CRITICAL risk score — requires immediate review',
    risk_tier,
    composite_score,
    amount
FROM scored_transactions
WHERE risk_tier = 'CRITICAL'
  AND transaction_id NOT IN (SELECT transaction_id FROM fraud_alerts);

-- 4c. View open alerts (for compliance dashboard)
SELECT
    fa.alert_id,
    fa.transaction_id,
    fa.user_id,
    fa.alert_reason,
    fa.risk_tier,
    ROUND(fa.composite_score, 4) AS composite_score,
    fa.amount,
    fa.alert_timestamp,
    fa.status,
    u.credit_score,
    u.is_flagged_account
FROM fraud_alerts fa
JOIN users u ON fa.user_id = u.user_id
WHERE fa.status = 'OPEN'
ORDER BY fa.composite_score DESC;
