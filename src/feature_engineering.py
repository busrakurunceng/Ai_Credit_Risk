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


# ====================================================================
# STEP 3 — Bureau table aggregations
# ====================================================================

def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the bureau table (external credit history from other
    financial institutions) to one row per SK_ID_CURR.

    Why this matters:
        A client's behavior with *other* lenders is one of the strongest
        predictors of future default.  Someone who already has overdue
        credits elsewhere is riskier than someone with a clean record.

    Features created (all prefixed BUREAU_):
        Counts:
            - CREDIT_COUNT          : total number of external credits
            - ACTIVE_COUNT          : credits still open
            - CLOSED_COUNT          : credits already closed
            - ACTIVE_RATIO          : proportion of credits that are active
        Overdue signals:
            - OVERDUE_MAX           : worst-case overdue days across all credits
            - OVERDUE_MEAN          : average overdue days
            - HAS_OVERDUE           : binary flag — any overdue > 0?
            - OVERDUE_CREDIT_SUM    : total overdue amount across credits
        Credit amounts:
            - CREDIT_SUM_TOTAL      : total external credit amount
            - CREDIT_SUM_MEAN       : average credit size
            - CREDIT_SUM_MAX        : largest single credit
            - DEBT_SUM_TOTAL        : total outstanding debt
            - DEBT_CREDIT_RATIO     : debt / credit ratio (leverage)
        Timing:
            - DAYS_CREDIT_MEAN      : average "age" of credits (how long ago opened)
            - DAYS_CREDIT_MIN       : most recent credit opened
            - PROLONGATION_SUM      : total number of credit prolongations

    Parameters
    ----------
    bureau : pd.DataFrame
        Raw bureau table with one row per external credit.

    Returns
    -------
    pd.DataFrame
        One row per SK_ID_CURR with aggregated features.
    """
    # --- Numeric aggregations -------------------------------------------------
    agg_spec = {
        "SK_ID_BUREAU":          ["count"],                          # total credits
        "CREDIT_DAY_OVERDUE":    ["max", "mean"],                    # overdue signals
        "AMT_CREDIT_SUM":        ["sum", "mean", "max"],             # credit amounts
        "AMT_CREDIT_SUM_DEBT":   ["sum"],                            # outstanding debt
        "AMT_CREDIT_SUM_OVERDUE":["sum"],                            # overdue amounts
        "DAYS_CREDIT":           ["mean", "min"],                    # credit timing
        "CNT_CREDIT_PROLONG":    ["sum"],                            # prolongations
    }

    bureau_agg = bureau.groupby("SK_ID_CURR").agg(agg_spec)

    # Flatten multi-level column names: ('col', 'stat') -> BUREAU_COL_STAT
    bureau_agg.columns = [
        f"BUREAU_{col}_{stat}".upper()
        for col, stat in bureau_agg.columns
    ]
    bureau_agg = bureau_agg.reset_index()

    # Rename for readability
    bureau_agg = bureau_agg.rename(columns={
        "BUREAU_SK_ID_BUREAU_COUNT":            "BUREAU_CREDIT_COUNT",
        "BUREAU_CREDIT_DAY_OVERDUE_MAX":        "BUREAU_OVERDUE_MAX",
        "BUREAU_CREDIT_DAY_OVERDUE_MEAN":       "BUREAU_OVERDUE_MEAN",
        "BUREAU_AMT_CREDIT_SUM_SUM":            "BUREAU_CREDIT_SUM_TOTAL",
        "BUREAU_AMT_CREDIT_SUM_MEAN":           "BUREAU_CREDIT_SUM_MEAN",
        "BUREAU_AMT_CREDIT_SUM_MAX":            "BUREAU_CREDIT_SUM_MAX",
        "BUREAU_AMT_CREDIT_SUM_DEBT_SUM":       "BUREAU_DEBT_SUM_TOTAL",
        "BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM":    "BUREAU_OVERDUE_CREDIT_SUM",
        "BUREAU_DAYS_CREDIT_MEAN":              "BUREAU_DAYS_CREDIT_MEAN",
        "BUREAU_DAYS_CREDIT_MIN":               "BUREAU_DAYS_CREDIT_MIN",
        "BUREAU_CNT_CREDIT_PROLONG_SUM":        "BUREAU_PROLONGATION_SUM",
    })

    # --- Active / Closed counts (from categorical CREDIT_ACTIVE) ---------------
    active_counts = (
        bureau.groupby("SK_ID_CURR")["CREDIT_ACTIVE"]
        .apply(lambda x: (x == "Active").sum())
        .rename("BUREAU_ACTIVE_COUNT")
    )
    closed_counts = (
        bureau.groupby("SK_ID_CURR")["CREDIT_ACTIVE"]
        .apply(lambda x: (x == "Closed").sum())
        .rename("BUREAU_CLOSED_COUNT")
    )

    bureau_agg = bureau_agg.merge(active_counts, on="SK_ID_CURR", how="left")
    bureau_agg = bureau_agg.merge(closed_counts, on="SK_ID_CURR", how="left")

    # --- Derived ratios -------------------------------------------------------
    bureau_agg["BUREAU_ACTIVE_RATIO"] = (
        bureau_agg["BUREAU_ACTIVE_COUNT"]
        / bureau_agg["BUREAU_CREDIT_COUNT"].clip(lower=1)
    )
    bureau_agg["BUREAU_DEBT_CREDIT_RATIO"] = (
        bureau_agg["BUREAU_DEBT_SUM_TOTAL"]
        / bureau_agg["BUREAU_CREDIT_SUM_TOTAL"].clip(lower=1)
    )
    bureau_agg["BUREAU_HAS_OVERDUE"] = (bureau_agg["BUREAU_OVERDUE_MAX"] > 0).astype(np.int8)

    n_features = len(bureau_agg.columns) - 1  # exclude SK_ID_CURR
    print(f"  Bureau aggregation: {n_features} features for {len(bureau_agg):,} clients")

    return bureau_agg


# ====================================================================
# STEP 4 — Previous application aggregations
# ====================================================================

def aggregate_previous_application(prev: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the previous_application table (past loan applications
    at Home Credit) to one row per SK_ID_CURR.

    Why this matters:
        A client's *history with Home Credit itself* tells us how they
        behaved before.  If they were rejected multiple times, or if
        their previous loans were much smaller, that contextualizes
        the current application.  The approval/rejection ratio is
        especially telling — serial rejections hint at hidden risk.

    Features created (all prefixed PREV_):
        Application counts:
            - APP_COUNT              : total previous applications
            - APPROVED_COUNT         : how many were approved
            - REFUSED_COUNT          : how many were refused
            - APPROVAL_RATE          : approved / total ratio
        Financial:
            - AMT_APPLICATION_MEAN   : average requested amount
            - AMT_APPLICATION_MAX    : largest request
            - AMT_CREDIT_MEAN        : average granted credit
            - AMT_CREDIT_MAX         : largest granted credit
            - AMT_DOWN_PAYMENT_MEAN  : average down payment
            - APP_CREDIT_DIFF_MEAN   : mean gap between requested and granted
        Timing:
            - DAYS_DECISION_MEAN     : average days since decisions
            - DAYS_DECISION_MIN      : most recent decision
        Loan characteristics:
            - CNT_PAYMENT_MEAN       : average payment count (loan term)
            - CASH_LOAN_RATIO        : proportion of cash-type loans

    Parameters
    ----------
    prev : pd.DataFrame
        Raw previous_application table.

    Returns
    -------
    pd.DataFrame
        One row per SK_ID_CURR with aggregated features.
    """
    # --- Numeric aggregations -------------------------------------------------
    agg_spec = {
        "SK_ID_PREV":       ["count"],                    # total applications
        "AMT_APPLICATION":  ["mean", "max"],              # requested amounts
        "AMT_CREDIT":       ["mean", "max"],              # granted amounts
        "AMT_DOWN_PAYMENT": ["mean"],                     # down payments
        "DAYS_DECISION":    ["mean", "min"],              # decision timing
        "CNT_PAYMENT":      ["mean"],                     # loan term
    }

    prev_agg = prev.groupby("SK_ID_CURR").agg(agg_spec)

    # Flatten column names: ('col', 'stat') -> PREV_COL_STAT
    prev_agg.columns = [
        f"PREV_{col}_{stat}".upper()
        for col, stat in prev_agg.columns
    ]
    prev_agg = prev_agg.reset_index()

    # Rename for readability
    prev_agg = prev_agg.rename(columns={
        "PREV_SK_ID_PREV_COUNT":         "PREV_APP_COUNT",
        "PREV_AMT_APPLICATION_MEAN":     "PREV_AMT_APPLICATION_MEAN",
        "PREV_AMT_APPLICATION_MAX":      "PREV_AMT_APPLICATION_MAX",
        "PREV_AMT_CREDIT_MEAN":          "PREV_AMT_CREDIT_MEAN",
        "PREV_AMT_CREDIT_MAX":           "PREV_AMT_CREDIT_MAX",
        "PREV_AMT_DOWN_PAYMENT_MEAN":    "PREV_AMT_DOWN_PAYMENT_MEAN",
        "PREV_DAYS_DECISION_MEAN":       "PREV_DAYS_DECISION_MEAN",
        "PREV_DAYS_DECISION_MIN":        "PREV_DAYS_DECISION_MIN",
        "PREV_CNT_PAYMENT_MEAN":         "PREV_CNT_PAYMENT_MEAN",
    })

    # --- Approval / Refusal counts (from NAME_CONTRACT_STATUS) ----------------
    approved = (
        prev.groupby("SK_ID_CURR")["NAME_CONTRACT_STATUS"]
        .apply(lambda x: (x == "Approved").sum())
        .rename("PREV_APPROVED_COUNT")
    )
    refused = (
        prev.groupby("SK_ID_CURR")["NAME_CONTRACT_STATUS"]
        .apply(lambda x: (x == "Refused").sum())
        .rename("PREV_REFUSED_COUNT")
    )

    prev_agg = prev_agg.merge(approved, on="SK_ID_CURR", how="left")
    prev_agg = prev_agg.merge(refused,  on="SK_ID_CURR", how="left")

    # --- Cash loan ratio (from NAME_CONTRACT_TYPE) ----------------------------
    cash_ratio = (
        prev.groupby("SK_ID_CURR")["NAME_CONTRACT_TYPE"]
        .apply(lambda x: (x == "Cash loans").sum() / max(len(x), 1))
        .rename("PREV_CASH_LOAN_RATIO")
    )
    prev_agg = prev_agg.merge(cash_ratio, on="SK_ID_CURR", how="left")

    # --- Derived features -----------------------------------------------------
    prev_agg["PREV_APPROVAL_RATE"] = (
        prev_agg["PREV_APPROVED_COUNT"]
        / prev_agg["PREV_APP_COUNT"].clip(lower=1)
    )
    prev_agg["PREV_APP_CREDIT_DIFF_MEAN"] = (
        prev_agg["PREV_AMT_APPLICATION_MEAN"] - prev_agg["PREV_AMT_CREDIT_MEAN"]
    )

    n_features = len(prev_agg.columns) - 1  # exclude SK_ID_CURR
    print(f"  Previous application aggregation: {n_features} features for {len(prev_agg):,} clients")

    return prev_agg
