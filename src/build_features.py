"""
Build Features Pipeline - Home Credit Default Risk

Runs the FULL feature engineering pipeline on ALL data (no sampling)
and saves the result to data/train_featured.parquet and data/test_featured.parquet.

Usage:
    python -m src.build_features

This script orchestrates all functions from src/feature_engineering.py
in the correct order. Each step prints progress so you can monitor
the pipeline.
"""

import os
import sys
import time

import numpy as np
import pandas as pd

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import reduce_memory_usage
from src.feature_engineering import (
    engineer_application_features,
    fix_days_employed_anomaly,
    aggregate_bureau,
    aggregate_previous_application,
    aggregate_installments,
    aggregate_credit_card,
    aggregate_pos_cash,
    aggregate_bureau_balance,
)

DATA_DIR = os.path.join(PROJECT_ROOT, "docs", "dataset")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV from the dataset directory with memory optimization."""
    path = os.path.join(DATA_DIR, filename)
    print(f"\n{'='*60}")
    print(f"Loading: {filename}")
    df = pd.read_csv(path)
    df = reduce_memory_usage(df, verbose=True)
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def build_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Apply the full feature engineering pipeline to an application DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        application_train or application_test.
    is_train : bool
        If True, prints TARGET distribution.

    Returns
    -------
    pd.DataFrame
        DataFrame with all engineered features merged.
    """
    pipeline_start = time.time()
    client_ids = set(df["SK_ID_CURR"].unique())
    label = "TRAIN" if is_train else "TEST"

    print(f"\n{'#'*60}")
    print(f"# FEATURE ENGINEERING PIPELINE - {label}")
    print(f"# Clients: {len(client_ids):,}")
    print(f"{'#'*60}")

    if is_train:
        print(f"\nTARGET distribution:")
        print(df["TARGET"].value_counts().to_string())

    # -- Step 1: Application-level derived features ----------------------------
    print(f"\n--- Step 1: Application derived features ---")
    original_cols = df.shape[1]
    df = engineer_application_features(df)
    print(f"  +{df.shape[1] - original_cols} features -> {df.shape[1]} columns")

    # -- Step 2: DAYS_EMPLOYED anomaly cleanup ---------------------------------
    print(f"\n--- Step 2: DAYS_EMPLOYED anomaly cleanup ---")
    df = fix_days_employed_anomaly(df)
    print(f"  -> {df.shape[1]} columns")

    # -- Step 3: Bureau aggregations -------------------------------------------
    print(f"\n--- Step 3: Bureau aggregations ---")
    bureau = _load_csv("bureau.csv")
    bureau = bureau[bureau["SK_ID_CURR"].isin(client_ids)].reset_index(drop=True)
    print(f"  Filtered to {len(bureau):,} rows")
    bureau_agg = aggregate_bureau(bureau)
    df = df.merge(bureau_agg, on="SK_ID_CURR", how="left")
    bureau_fill = [c for c in bureau_agg.columns if c != "SK_ID_CURR" and
                   any(k in c for k in ["COUNT", "HAS_", "SUM", "PROLONGATION"])]
    df[bureau_fill] = df[bureau_fill].fillna(0)
    print(f"  -> {df.shape[1]} columns")

    # -- Step 4: Previous application aggregations -----------------------------
    print(f"\n--- Step 4: Previous application aggregations ---")
    prev = _load_csv("previous_application.csv")
    prev = prev[prev["SK_ID_CURR"].isin(client_ids)].reset_index(drop=True)
    print(f"  Filtered to {len(prev):,} rows")
    prev_agg = aggregate_previous_application(prev)
    df = df.merge(prev_agg, on="SK_ID_CURR", how="left")
    prev_fill = [c for c in prev_agg.columns if c != "SK_ID_CURR" and
                 any(k in c for k in ["COUNT", "RATE"])]
    df[prev_fill] = df[prev_fill].fillna(0)
    print(f"  -> {df.shape[1]} columns")
    del prev; import gc; gc.collect()

    # -- Step 5: Installments payments aggregations ----------------------------
    print(f"\n--- Step 5: Installments aggregations ---")
    inst = _load_csv("installments_payments.csv")
    inst = inst[inst["SK_ID_CURR"].isin(client_ids)].reset_index(drop=True)
    print(f"  Filtered to {len(inst):,} rows")
    inst_agg = aggregate_installments(inst)
    df = df.merge(inst_agg, on="SK_ID_CURR", how="left")
    inst_fill = [c for c in inst_agg.columns if c != "SK_ID_CURR" and
                 any(k in c for k in ["COUNT", "RATIO", "SUM", "LATE", "UNDERPAID"])]
    df[inst_fill] = df[inst_fill].fillna(0)
    print(f"  -> {df.shape[1]} columns")
    del inst; gc.collect()

    # -- Step 6: Credit card balance aggregations ------------------------------
    print(f"\n--- Step 6: Credit card aggregations ---")
    cc = _load_csv("credit_card_balance.csv")
    cc = cc[cc["SK_ID_CURR"].isin(client_ids)].reset_index(drop=True)
    print(f"  Filtered to {len(cc):,} rows")
    cc_agg = aggregate_credit_card(cc)
    df = df.merge(cc_agg, on="SK_ID_CURR", how="left")
    cc_fill = [c for c in cc_agg.columns if c != "SK_ID_CURR" and
               any(k in c for k in ["COUNT", "HAS_"])]
    df[cc_fill] = df[cc_fill].fillna(0)
    print(f"  -> {df.shape[1]} columns")
    del cc; gc.collect()

    # -- Step 7: POS_CASH balance aggregations ---------------------------------
    print(f"\n--- Step 7: POS_CASH aggregations ---")
    pos = _load_csv("POS_CASH_balance.csv")
    pos = pos[pos["SK_ID_CURR"].isin(client_ids)].reset_index(drop=True)
    print(f"  Filtered to {len(pos):,} rows")
    pos_agg = aggregate_pos_cash(pos)
    df = df.merge(pos_agg, on="SK_ID_CURR", how="left")
    pos_fill = [c for c in pos_agg.columns if c != "SK_ID_CURR" and
                any(k in c for k in ["COUNT", "HAS_", "RATIO"])]
    df[pos_fill] = df[pos_fill].fillna(0)
    print(f"  -> {df.shape[1]} columns")
    del pos; gc.collect()

    # -- Step 8: Bureau balance aggregations -----------------------------------
    print(f"\n--- Step 8: Bureau balance aggregations ---")
    bb = _load_csv("bureau_balance.csv")
    # Need bureau table for SK_ID_BUREAU -> SK_ID_CURR mapping
    bureau_key = bureau[["SK_ID_BUREAU", "SK_ID_CURR"]].drop_duplicates()
    valid_bureau_ids = set(bureau_key["SK_ID_BUREAU"].unique())
    bb = bb[bb["SK_ID_BUREAU"].isin(valid_bureau_ids)].reset_index(drop=True)
    print(f"  Pre-filtered to {len(bb):,} rows")
    bb_agg = aggregate_bureau_balance(bb, bureau_key)
    df = df.merge(bb_agg, on="SK_ID_CURR", how="left")
    bb_fill = [c for c in bb_agg.columns if c != "SK_ID_CURR" and
               any(k in c for k in ["COUNT", "HAS_", "RATIO"])]
    df[bb_fill] = df[bb_fill].fillna(0)
    print(f"  -> {df.shape[1]} columns")
    del bb, bureau, bureau_key; gc.collect()

    # -- Final memory optimization ---------------------------------------------
    df = reduce_memory_usage(df, verbose=True)

    elapsed = time.time() - pipeline_start
    print(f"\n{'#'*60}")
    print(f"# PIPELINE COMPLETE - {label}")
    print(f"# Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"# Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'#'*60}")

    return df


def main():
    total_start = time.time()

    # ===== TRAIN =====
    train = _load_csv("application_train.csv")
    train = build_features(train, is_train=True)

    train_path = os.path.join(OUTPUT_DIR, "train_featured.parquet")
    train.to_parquet(train_path, index=False)
    size_mb = os.path.getsize(train_path) / 1024 ** 2
    print(f"\nSaved: {train_path} ({size_mb:.1f} MB)")

    # ===== TEST =====
    test = _load_csv("application_test.csv")
    test = build_features(test, is_train=False)

    test_path = os.path.join(OUTPUT_DIR, "test_featured.parquet")
    test.to_parquet(test_path, index=False)
    size_mb = os.path.getsize(test_path) / 1024 ** 2
    print(f"\nSaved: {test_path} ({size_mb:.1f} MB)")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"ALL DONE - Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Train: {train.shape}")
    print(f"  Test:  {test.shape}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
