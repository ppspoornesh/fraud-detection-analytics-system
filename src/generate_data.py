"""
generate_data.py
Generates a realistic synthetic financial transactions dataset for fraud detection.
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

N_USERS = 500
N_TRANSACTIONS = 50000
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)

MERCHANT_CATEGORIES = [
    "Grocery", "Electronics", "Restaurants", "Travel", "Healthcare",
    "Utilities", "Entertainment", "Online Shopping", "Gas Station",
    "Retail", "ATM Withdrawal", "Wire Transfer", "Crypto Exchange",
    "Gambling", "Jewelry"
]

HIGH_RISK_CATEGORIES = {"Wire Transfer", "Crypto Exchange", "Gambling", "ATM Withdrawal", "Jewelry"}

COUNTRIES = ["US", "US", "US", "US", "US", "GB", "CA", "IN", "NG", "RU", "BR", "CN"]
HIGH_RISK_COUNTRIES = {"NG", "RU", "CN"}

def random_date(start, end):
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

def generate_users(n=N_USERS):
    users = []
    for i in range(1, n + 1):
        users.append({
            "user_id": f"U{i:05d}",
            "age": random.randint(18, 75),
            "account_age_days": random.randint(30, 3650),
            "credit_score": random.randint(300, 850),
            "avg_monthly_spend": round(random.uniform(500, 8000), 2),
            "country": random.choice(["US", "US", "US", "GB", "CA", "IN"]),
            "is_flagged_account": random.choices([0, 1], weights=[0.96, 0.04])[0]
        })
    return pd.DataFrame(users)

def generate_transactions(users_df, n=N_TRANSACTIONS):
    user_ids = users_df["user_id"].tolist()
    flagged_users = set(users_df[users_df["is_flagged_account"] == 1]["user_id"])

    transactions = []
    for i in range(1, n + 1):
        user_id = random.choice(user_ids)
        is_flagged_user = user_id in flagged_users
        category = random.choice(MERCHANT_CATEGORIES)
        country = random.choice(COUNTRIES)
        txn_date = random_date(START_DATE, END_DATE)

        # Base amount
        if category in HIGH_RISK_CATEGORIES:
            amount = round(np.random.lognormal(6, 1.5), 2)
        else:
            amount = round(np.random.lognormal(4, 1.2), 2)

        # Fraud signal: inject anomalies
        is_fraud = 0
        fraud_reason = None

        # Scenario 1: flagged user + high-risk category
        if is_flagged_user and category in HIGH_RISK_CATEGORIES and random.random() < 0.35:
            is_fraud = 1
            fraud_reason = "flagged_user_high_risk_category"
            amount = round(amount * random.uniform(3, 10), 2)

        # Scenario 2: high amount + high risk country
        elif country in HIGH_RISK_COUNTRIES and amount > 5000 and random.random() < 0.45:
            is_fraud = 1
            fraud_reason = "high_risk_country_large_amount"

        # Scenario 3: unusual hour (1am - 4am) + large amount
        elif txn_date.hour in range(1, 5) and amount > 3000 and random.random() < 0.30:
            is_fraud = 1
            fraud_reason = "unusual_hour_large_amount"

        # Scenario 4: velocity – random inject
        elif random.random() < 0.015:
            is_fraud = 1
            fraud_reason = "random_anomaly"

        transactions.append({
            "transaction_id": f"TXN{i:08d}",
            "user_id": user_id,
            "txn_date": txn_date.strftime("%Y-%m-%d"),
            "txn_time": txn_date.strftime("%H:%M:%S"),
            "txn_datetime": txn_date.strftime("%Y-%m-%d %H:%M:%S"),
            "amount": amount,
            "currency": "USD",
            "merchant_category": category,
            "merchant_name": f"Merchant_{random.randint(1, 200):03d}",
            "transaction_country": country,
            "device_type": random.choice(["Mobile", "Web", "POS", "ATM"]),
            "ip_flag": random.choices([0, 1], weights=[0.92, 0.08])[0],
            "is_international": 1 if country != "US" else 0,
            "is_fraud": is_fraud,
            "fraud_reason": fraud_reason if is_fraud else "none"
        })

    return pd.DataFrame(transactions)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)

    print("Generating users...")
    users = generate_users()
    users.to_csv(os.path.join(out_dir, "users.csv"), index=False)
    print(f"  → {len(users)} users saved.")

    print("Generating transactions...")
    transactions = generate_transactions(users)
    transactions.to_csv(os.path.join(out_dir, "transactions.csv"), index=False)
    print(f"  → {len(transactions)} transactions saved.")
    print(f"  → Fraud rate: {transactions['is_fraud'].mean()*100:.2f}%")
    print("Done. Data saved to /data/")


if __name__ == "__main__":
    main()
