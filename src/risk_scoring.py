"""
risk_scoring.py
Risk scoring engine: rule-based + ML-based scoring,
classifying transactions into LOW / MEDIUM / HIGH / CRITICAL risk tiers.
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_fscore_support
)

# ─────────────────────────────────────────────
# RISK TIER THRESHOLDS
# ─────────────────────────────────────────────
RISK_TIERS = {
    "LOW":      (0.00, 0.30),
    "MEDIUM":   (0.30, 0.55),
    "HIGH":     (0.55, 0.75),
    "CRITICAL": (0.75, 1.01),
}

FEATURES = [
    "amount",
    "amount_z_score",
    "txn_count_24h",
    "txn_amt_24h",
    "is_odd_hour",
    "is_weekend",
    "is_international",
    "is_high_risk_category",
    "is_high_risk_country",
    "ip_flag",
    "is_flagged_account",
    "credit_score",
    "account_age_days",
    "spend_ratio",
]


# ─────────────────────────────────────────────
# 1. RULE-BASED SCORING
# ─────────────────────────────────────────────

def rule_based_score(row: pd.Series) -> float:
    """
    Assigns a heuristic risk score in [0, 1] based on domain rules.
    Higher score = higher fraud likelihood.
    """
    score = 0.0

    # Large transaction amount
    if row["amount"] > 10000:
        score += 0.20
    elif row["amount"] > 5000:
        score += 0.12
    elif row["amount"] > 2000:
        score += 0.06

    # Statistical deviation from user's own history
    z = abs(row.get("amount_z_score", 0))
    if z > 4:
        score += 0.18
    elif z > 2.5:
        score += 0.10
    elif z > 1.5:
        score += 0.05

    # High velocity
    if row.get("txn_count_24h", 0) > 15:
        score += 0.15
    elif row.get("txn_count_24h", 0) > 8:
        score += 0.08

    # Unusual hour
    if row.get("is_odd_hour", 0):
        score += 0.08

    # High-risk merchant category
    if row.get("is_high_risk_category", 0):
        score += 0.12

    # High-risk country
    if row.get("is_high_risk_country", 0):
        score += 0.12

    # International transaction
    if row.get("is_international", 0):
        score += 0.05

    # Suspicious IP flag
    if row.get("ip_flag", 0):
        score += 0.10

    # Account flagged by compliance
    if row.get("is_flagged_account", 0):
        score += 0.15

    # Low credit score
    cs = row.get("credit_score", 700)
    if cs < 450:
        score += 0.08
    elif cs < 600:
        score += 0.04

    # Young account
    age = row.get("account_age_days", 365)
    if age < 30:
        score += 0.10
    elif age < 90:
        score += 0.05

    # High spend ratio vs monthly baseline
    sr = row.get("spend_ratio", 1)
    if sr > 5:
        score += 0.12
    elif sr > 2:
        score += 0.06

    return min(score, 1.0)


def apply_rule_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Apply rule-based scoring to entire dataframe."""
    df = df.copy()
    df["rule_score"] = df.apply(rule_based_score, axis=1)
    return df


# ─────────────────────────────────────────────
# 2. ML-BASED SCORING (Random Forest)
# ─────────────────────────────────────────────

def train_rf_model(df: pd.DataFrame):
    """Train a Random Forest classifier on labeled fraud data."""
    available_features = [f for f in FEATURES if f in df.columns]
    X = df[available_features].fillna(0)
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_sc, y_train)

    y_pred = clf.predict(X_test_sc)
    y_prob = clf.predict_proba(X_test_sc)[:, 1]

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "feature_importances": dict(
            zip(available_features, clf.feature_importances_.round(4).tolist())
        )
    }
    print(f"[RF Model] ROC-AUC: {metrics['roc_auc']}")
    print(classification_report(y_test, y_pred))
    return clf, scaler, metrics, available_features


def apply_ml_scores(df: pd.DataFrame, clf, scaler, features) -> pd.DataFrame:
    """Apply trained RF model probabilities as ml_score."""
    df = df.copy()
    X = df[features].fillna(0)
    X_sc = scaler.transform(X)
    df["ml_score"] = clf.predict_proba(X_sc)[:, 1]
    return df


# ─────────────────────────────────────────────
# 3. ISOLATION FOREST (Unsupervised Anomaly)
# ─────────────────────────────────────────────

def apply_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    """Flag anomalies using Isolation Forest (unsupervised)."""
    available_features = [f for f in FEATURES if f in df.columns]
    X = df[available_features].fillna(0)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    iso = IsolationForest(contamination=0.025, random_state=42, n_jobs=-1)
    df = df.copy()
    df["isolation_flag"] = (iso.fit_predict(X_sc) == -1).astype(int)
    df["isolation_score"] = -iso.score_samples(X_sc)  # higher = more anomalous
    # Normalize to [0, 1]
    mn, mx = df["isolation_score"].min(), df["isolation_score"].max()
    df["isolation_score"] = (df["isolation_score"] - mn) / (mx - mn + 1e-9)

    print(f"[IsoForest] Flagged {df['isolation_flag'].sum():,} anomalies.")
    return df


# ─────────────────────────────────────────────
# 4. COMPOSITE SCORE & RISK TIER
# ─────────────────────────────────────────────

def compute_composite_score(df: pd.DataFrame,
                             w_rule=0.35, w_ml=0.45, w_iso=0.20) -> pd.DataFrame:
    """
    Weighted composite fraud risk score.
    Weights: rule_score=35%, ml_score=45%, isolation_score=20%
    """
    df = df.copy()

    # Handle missing component scores gracefully
    rule = df.get("rule_score", pd.Series(0.0, index=df.index))
    ml   = df.get("ml_score",   pd.Series(0.0, index=df.index))
    iso  = df.get("isolation_score", pd.Series(0.0, index=df.index))

    df["composite_score"] = (w_rule * rule + w_ml * ml + w_iso * iso).clip(0, 1)
    return df


def assign_risk_tier(score: float) -> str:
    for tier, (lo, hi) in RISK_TIERS.items():
        if lo <= score < hi:
            return tier
    return "CRITICAL"


def apply_risk_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """Assign risk tier labels based on composite score."""
    df = df.copy()
    df["risk_tier"] = df["composite_score"].apply(assign_risk_tier)
    return df


# ─────────────────────────────────────────────
# 5. FULL SCORING PIPELINE
# ─────────────────────────────────────────────

def run_scoring_pipeline(df: pd.DataFrame):
    """
    Full scoring pipeline:
    1. Rule-based scoring
    2. Isolation Forest (unsupervised)
    3. Random Forest (supervised)
    4. Composite score & risk tier
    Returns: scored DataFrame, model metrics dict
    """
    print("\n── Rule-Based Scoring ──")
    df = apply_rule_scores(df)

    print("\n── Isolation Forest ──")
    df = apply_isolation_forest(df)

    print("\n── Random Forest Classifier ──")
    clf, scaler, metrics, features = train_rf_model(df)
    df = apply_ml_scores(df, clf, scaler, features)

    print("\n── Composite Score + Risk Tiers ──")
    df = compute_composite_score(df)
    df = apply_risk_tiers(df)

    tier_counts = df["risk_tier"].value_counts().to_dict()
    print(f"[Tiers] {tier_counts}")
    return df, metrics


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocessing import run_pipeline

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    df = run_pipeline(data_dir=data_dir)
    scored_df, metrics = run_scoring_pipeline(df)

    out = os.path.join(data_dir, "scored_transactions.csv")
    scored_df.to_csv(out, index=False)
    print(f"\n[Done] Scored data saved to {out}")
    print(scored_df[["transaction_id", "amount", "composite_score", "risk_tier", "is_fraud"]].head(10))
