"""
Feature engineering functions for Home Credit Default Risk project.

Each function takes a DataFrame (or dict of DataFrames) and returns
a new DataFrame with engineered features, ready to merge on SK_ID_CURR.
"""

import numpy as np
import pandas as pd


# ====================================================================
# STEP 1 — Application table: derived features
# ====================================================================

def engineer_application_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from the main application_train / application_test table.

    All transformations are row-level (no joins needed), so this is fast
    and memory-friendly.

    New features created:
        - CREDIT_INCOME_RATIO:   loan amount relative to income
        - ANNUITY_INCOME_RATIO:  annual payment burden
        - CREDIT_TERM:           estimated repayment duration in months
        - GOODS_CREDIT_DIFF:     difference between goods price and credit
        - INCOME_PER_CHILD:      income divided by number of children
        - AGE_YEARS:             client age in years
        - EMPLOYED_YEARS:        employment duration in years
        - REGISTRATION_YEARS:    years since ID registration
        - HAS_CAR:               binary flag derived from OWN_CAR_AGE
        - INCOME_PER_FAMILY:     income per family member

    Parameters
    ----------
    df : pd.DataFrame
        application_train or application_test DataFrame.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with new columns appended.
    """
    df = df.copy()

    # -- Financial ratios --------------------------------------------------
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["CREDIT_TERM"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1)
    df["GOODS_CREDIT_DIFF"] = df["AMT_GOODS_PRICE"] - df["AMT_CREDIT"]

    # Income per child (0 children -> use 1 to avoid division by zero)
    df["INCOME_PER_CHILD"] = df["AMT_INCOME_TOTAL"] / (df["CNT_CHILDREN"].clip(lower=1))
    # Clients with 0 children get their full income as the value
    df.loc[df["CNT_CHILDREN"] == 0, "INCOME_PER_CHILD"] = df["AMT_INCOME_TOTAL"]

    # Income per family member
    df["INCOME_PER_FAMILY"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"].clip(lower=1))

    # -- Time-based features (DAYS columns are negative; negate to get positive) --
    df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365.25).round(1)
    df["EMPLOYED_YEARS"] = (-df["DAYS_EMPLOYED"] / 365.25).round(1)
    df["REGISTRATION_YEARS"] = (-df["DAYS_REGISTRATION"] / 365.25).round(1)

    # -- Binary flags -------------------------------------------------------
    # has_car: 1 if OWN_CAR_AGE has a value (i.e. client owns a car), else 0
    df["HAS_CAR"] = df["OWN_CAR_AGE"].notna().astype(np.int8)

    return df


# ====================================================================
# STEP 2 — DAYS_EMPLOYED anomaly cleanup
# ====================================================================

# Home Credit uses 365243 as a sentinel value in DAYS_EMPLOYED
# for clients who are not employed (e.g. retired, unemployed).
# This maps to ~1000 years which is clearly not real data.
_DAYS_EMPLOYED_ANOMALY = 365243


def fix_days_employed_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the known anomaly in DAYS_EMPLOYED where the value 365243
    is used as a placeholder for unemployed / retired clients.

    Creates:
        - DAYS_EMPLOYED_ANOMALY: binary flag (1 = had the anomalous value)
        - DAYS_EMPLOYED: anomalous values replaced with NaN, then filled
          with median of non-anomalous records.
        - EMPLOYED_YEARS: recalculated after cleanup.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing DAYS_EMPLOYED column.

    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned DAYS_EMPLOYED and new flag column.
    """
    df = df.copy()

    # Flag rows with the anomalous sentinel value
    df["DAYS_EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"] == _DAYS_EMPLOYED_ANOMALY).astype(np.int8)

    anomaly_count = df["DAYS_EMPLOYED_ANOMALY"].sum()
    anomaly_pct = anomaly_count / len(df) * 100
    print(f"  DAYS_EMPLOYED anomaly: {anomaly_count:,} rows ({anomaly_pct:.1f}%) replaced with NaN")

    # Replace anomaly with NaN
    df.loc[df["DAYS_EMPLOYED"] == _DAYS_EMPLOYED_ANOMALY, "DAYS_EMPLOYED"] = np.nan

    # Fill NaN with median of non-anomalous values
    median_val = df["DAYS_EMPLOYED"].median()
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].fillna(median_val)
    print(f"  DAYS_EMPLOYED NaN filled with median: {median_val:.0f} days")

    # Recalculate EMPLOYED_YEARS after cleanup
    df["EMPLOYED_YEARS"] = (-df["DAYS_EMPLOYED"] / 365.25).round(1)

    return df
