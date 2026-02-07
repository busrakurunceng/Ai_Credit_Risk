"""
Utility functions for memory optimization and data loading.
Home Credit Default Risk - AI Credit Risk Project
"""

import numpy as np
import pandas as pd
import os
import time


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types.

    Applies the following conversions:
        - float64 -> float32
        - int64   -> int8 / int16 / int32 (based on value range)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to optimize.
    verbose : bool
        If True, print memory reduction summary.

    Returns
    -------
    pd.DataFrame
        Memory-optimized DataFrame.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and col_type.name != "category":
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type).startswith("int"):
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            elif str(col_type).startswith("float"):
                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2

    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem
        print(f"  Memory usage: {start_mem:.1f} MB -> {end_mem:.1f} MB ({reduction:.1f}% reduction)")

    return df


def load_dataset(data_dir: str, sample_frac: float = None, random_state: int = 42) -> dict:
    """
    Load all Home Credit dataset tables and apply memory optimization.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing CSV files.
    sample_frac : float, optional
        Fraction of data to sample (0-1). If None, loads full data.
    random_state : int
        Random seed for reproducible sampling.

    Returns
    -------
    dict
        Mapping of table name -> DataFrame.
    """
    tables = {
        "application_train": "application_train.csv",
        "application_test": "application_test.csv",
        "bureau": "bureau.csv",
        "bureau_balance": "bureau_balance.csv",
        "previous_application": "previous_application.csv",
        "installments_payments": "installments_payments.csv",
        "credit_card_balance": "credit_card_balance.csv",
        "POS_CASH_balance": "POS_CASH_balance.csv",
    }

    datasets = {}

    for name, filename in tables.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"  [!] {filename} not found, skipping...")
            continue

        print(f"\n{'='*60}")
        print(f"Loading: {filename}")
        start = time.time()

        df = pd.read_csv(filepath)

        if sample_frac is not None and sample_frac < 1.0:
            # Directly sample main application tables
            if name in ("application_train", "application_test"):
                df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
                print(f"  Sampled: {len(df):,} rows ({sample_frac*100:.0f}%)")
            # Filter auxiliary tables by sampled train IDs for consistency
            elif "application_train" in datasets:
                train_ids = set(datasets["application_train"]["SK_ID_CURR"].unique())
                if "SK_ID_CURR" in df.columns:
                    df = df[df["SK_ID_CURR"].isin(train_ids)].reset_index(drop=True)
                    print(f"  Filtered: {len(df):,} rows (by train IDs)")

        df = reduce_memory_usage(df, verbose=True)

        elapsed = time.time() - start
        print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns ({elapsed:.1f}s)")

        datasets[name] = df

    return datasets


def get_missing_info(df: pd.DataFrame, table_name: str = "") -> pd.DataFrame:
    """
    Compute missing value statistics for each column in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyze.
    table_name : str
        Label for the source table (used in output).

    Returns
    -------
    pd.DataFrame
        Per-column missing value report sorted by missing percentage (desc).
    """
    total = df.isnull().sum()
    percent = (total / len(df)) * 100
    dtypes = df.dtypes

    missing_df = pd.DataFrame({
        "Table": table_name,
        "Column": total.index,
        "Missing_Count": total.values,
        "Missing_Pct": percent.values,
        "Dtype": dtypes.values,
    })

    # Keep only columns that have at least one missing value
    missing_df = missing_df[missing_df["Missing_Count"] > 0]
    missing_df = missing_df.sort_values("Missing_Pct", ascending=False).reset_index(drop=True)

    return missing_df
