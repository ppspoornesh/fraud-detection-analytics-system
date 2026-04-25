"""
run_pipeline.py
Master script — runs the full fraud detection pipeline end to end:
  1. Generate synthetic data
  2. Preprocess + feature engineering
  3. Risk scoring (rule + ML)
  4. Load into SQLite + generate reports
"""

import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

def banner(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def main():
    t0 = time.time()

    banner("STEP 1/4 — Generating Synthetic Data")
    from generate_data import main as gen_main
    gen_main()

    banner("STEP 2/4 — Preprocessing & Feature Engineering")
    from preprocessing import run_pipeline
    df = run_pipeline(data_dir=os.path.join(BASE_DIR, "data"))

    banner("STEP 3/4 — Risk Scoring (Rule + Isolation Forest + Random Forest)")
    from risk_scoring import run_scoring_pipeline
    scored_df, metrics = run_scoring_pipeline(df)

    out = os.path.join(BASE_DIR, "data", "scored_transactions.csv")
    scored_df.to_csv(out, index=False)
    print(f"\n[Saved] scored_transactions.csv → {out}")

    banner("STEP 4/4 — Loading to SQLite & Generating Reports")
    from db_loader import main as db_main
    db_main()

    elapsed = round(time.time() - t0, 1)
    banner(f"✅ Pipeline Complete in {elapsed}s")
    print("Next step: Launch dashboard with:")
    print("  streamlit run dashboard/app.py\n")


if __name__ == "__main__":
    main()
