"""
preprocessing.py
Data cleaning, transformation, validation, and feature engineering
for the Fraud Detection Analytics System.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_data(data_dir: str = "../data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load transactions and users CSVs."""
    txn_path = os.path.join(data_dir, "transactions.csv")
    usr_path = os.path.join(data_dir, "users.csv")

    if not os.path.exists(txn_path):
        raise FileNotFoundError(f"transactions.csv not found at {txn_path}. Run src/generate_data.py first.")

    transactions = pd.read_csv(txn_path, parse_dates=["txn_datetime"])
    users = pd.read_csv(usr_path)
    print(f"[load] {len(transactions):,} transactions | {len(users):,} users loaded.")
    return transactions, users


# ─────────────────────────────────────────────
# 2. CLEANING & VALIDATION
# ─────────────────────────────────────────────

def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate schema, remove duplicates, fix nulls,
    and enforce data-type constraints.
    """
    initial = len(df)
    report = {}

    # Drop exact duplicates
    df = df.drop_duplicates(subset=["transaction_id"])
    report["duplicate_rows_removed"] = initial - len(df)

    # Enforce non-negative amounts
    neg_mask = df["amount"] <= 0
    report["negative_amount_rows"] = neg_mask.sum()
    df = df[~neg_mask]

    # Cap extreme outliers (>99.9th percentile) — flag but keep
    cap = df["amount"].quantile(0.999)
    df["amount_capped"] = df["amount"].clip(upper=cap)
    report["outlier_amount_capped"] = (df["amount"] > cap).sum()

    # Null handling
    df["fraud_reason"] = df["fraud_reason"].fillna("none")
    report["nulls_filled"] = df.isnull().sum().to_dict()

    # Parse datetime safely
    df["txn_datetime"] = pd.to_datetime(df["txn_datetime"], errors="coerce")
    df = df.dropna(subset=["txn_datetime"])

    # Type enforcement
    df["is_fraud"] = df["is_fraud"].astype(int)
    df["is_international"] = df["is_international"].astype(int)
    df["ip_flag"] = df["ip_flag"].astype(int)

    print(f"[clean] {len(df):,} rows after cleaning | Report: {report}")
    return df


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(txn: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """Create risk-relevant features for scoring and analysis."""

    df = txn.copy()
    df = df.sort_values(["user_id", "txn_datetime"])

    # ── Time-based features ──
    df["txn_hour"]       = df["txn_datetime"].dt.hour
    df["txn_dow"]        = df["txn_datetime"].dt.dayofweek       # 0=Mon
    df["txn_month"]      = df["txn_datetime"].dt.month
    df["txn_quarter"]    = df["txn_datetime"].dt.quarter
    df["is_weekend"]     = (df["txn_dow"] >= 5).astype(int)
    df["is_odd_hour"]    = df["txn_hour"].apply(lambda h: 1 if h < 6 or h >= 22 else 0)

    # ── Velocity features (rolling 24h per user) ──
    df = df.set_index("txn_datetime")
    df["txn_count_24h"] = (
        df.groupby("user_id")["amount"]
        .transform(lambda x: x.rolling("24h", min_periods=1).count())
    )
    df["txn_amt_24h"] = (
        df.groupby("user_id")["amount"]
        .transform(lambda x: x.rolling("24h", min_periods=1).sum())
    )
    df = df.reset_index()

    # ── Amount deviation from user mean ──
    user_avg = df.groupby("user_id")["amount"].mean().rename("user_avg_amount")
    user_std = df.groupby("user_id")["amount"].std().rename("user_std_amount")
    df = df.merge(user_avg, on="user_id", how="left")
    df = df.merge(user_std, on="user_id", how="left")
    df["user_std_amount"] = df["user_std_amount"].fillna(1)
    df["amount_z_score"] = (
        (df["amount"] - df["user_avg_amount"]) / df["user_std_amount"].replace(0, 1)
    )

    # ── Category risk flag ──
    HIGH_RISK_CATS = {"Wire Transfer", "Crypto Exchange", "Gambling", "ATM Withdrawal", "Jewelry"}
    df["is_high_risk_category"] = df["merchant_category"].apply(
        lambda c: 1 if c in HIGH_RISK_CATS else 0
    )

    # ── Country risk flag ──
    HIGH_RISK_COUNTRIES = {"NG", "RU", "CN"}
    df["is_high_risk_country"] = df["transaction_country"].apply(
        lambda c: 1 if c in HIGH_RISK_COUNTRIES else 0
    )

    # ── Merge user profile ──
    df = df.merge(
        users[["user_id", "credit_score", "account_age_days", "avg_monthly_spend", "is_flagged_account"]],
        on="user_id", how="left"
    )

    # ── Spend ratio vs user baseline ──
    df["spend_ratio"] = df["amount"] / df["avg_monthly_spend"].replace(0, 1)

    print(f"[features] Feature engineering complete. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 4. FULL PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(data_dir: str = "../data") -> pd.DataFrame:
    """End-to-end data preparation pipeline."""
    transactions, users = load_data(data_dir)
    clean_txn = validate_and_clean(transactions)
    featured = engineer_features(clean_txn, users)
    return featured


if __name__ == "__main__":
    df = run_pipeline(data_dir=os.path.join(os.path.dirname(__file__), "..", "data"))
    print(df.head())
    print(df.dtypes)
