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


# ====================================================================
# STEP 5 — Installments payments aggregations
# ====================================================================

def aggregate_installments(inst: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the installments_payments table to one row per SK_ID_CURR.

    Why this is the most valuable table:
        Bureau tells us *what credits* a client has.  Previous application
        tells us *what they applied for*.  But installments_payments tells
        us **how they actually paid** — the ground truth of behavior.

        Each row is a single scheduled payment vs. actual payment.  By
        comparing the two we derive DPD (Days Past Due) and payment
        shortfall, the two strongest behavioral signals for default.

    Derived columns (computed before aggregation):
        - DPD: DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT
               Positive = paid late (past due), negative = paid early.
        - PAYMENT_DIFF: AMT_PAYMENT - AMT_INSTALMENT
               Positive = overpaid, negative = underpaid.
        - IS_LATE: binary, 1 if DPD > 0
        - IS_UNDERPAID: binary, 1 if PAYMENT_DIFF < 0

    Aggregated features (all prefixed INST_):
        Payment behavior:
            - DPD_MEAN             : average days past due across all payments
            - DPD_MAX              : worst single late payment
            - DPD_SUM              : total accumulated late days
            - LATE_COUNT           : how many payments were late
            - LATE_RATIO           : proportion of late payments
        Payment accuracy:
            - PAYMENT_DIFF_MEAN    : avg over/under-payment amount
            - PAYMENT_DIFF_MIN     : worst single underpayment
            - UNDERPAID_COUNT      : how many payments were less than required
            - UNDERPAID_RATIO      : proportion of underpaid installments
        Volume:
            - PAYMENT_COUNT        : total number of installment records
            - AMT_PAYMENT_MEAN     : average payment amount
            - AMT_PAYMENT_SUM      : total amount paid
            - AMT_INSTALMENT_SUM   : total amount that was due
            - PAYMENT_FULFILLMENT  : total paid / total due ratio

    Parameters
    ----------
    inst : pd.DataFrame
        Raw installments_payments table.

    Returns
    -------
    pd.DataFrame
        One row per SK_ID_CURR with aggregated features.
    """
    inst = inst.copy()

    # -- Derive row-level signals before aggregation ---------------------------
    # DPD: how many days late (positive = late, negative = early)
    inst["DPD"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
    inst["DPD"] = inst["DPD"].clip(lower=0)  # only care about lateness, not earliness

    # Payment difference: actual - expected (negative = underpaid)
    inst["PAYMENT_DIFF"] = inst["AMT_PAYMENT"] - inst["AMT_INSTALMENT"]

    # Binary flags
    inst["IS_LATE"] = (inst["DPD"] > 0).astype(np.int8)
    inst["IS_UNDERPAID"] = (inst["PAYMENT_DIFF"] < -1).astype(np.int8)  # 1 TL tolerance

    # -- GroupBy aggregation ---------------------------------------------------
    agg_spec = {
        "SK_ID_PREV":      ["count"],                         # total payments
        "DPD":             ["mean", "max", "sum"],             # lateness
        "IS_LATE":         ["sum", "mean"],                    # late count & ratio
        "PAYMENT_DIFF":    ["mean", "min"],                    # under/over payment
        "IS_UNDERPAID":    ["sum", "mean"],                    # underpaid count & ratio
        "AMT_PAYMENT":     ["mean", "sum"],                    # payment amounts
        "AMT_INSTALMENT":  ["sum"],                            # total due
    }

    inst_agg = inst.groupby("SK_ID_CURR").agg(agg_spec)

    # Flatten column names
    inst_agg.columns = [
        f"INST_{col}_{stat}".upper()
        for col, stat in inst_agg.columns
    ]
    inst_agg = inst_agg.reset_index()

    # Rename for readability
    inst_agg = inst_agg.rename(columns={
        "INST_SK_ID_PREV_COUNT":      "INST_PAYMENT_COUNT",
        "INST_DPD_MEAN":              "INST_DPD_MEAN",
        "INST_DPD_MAX":               "INST_DPD_MAX",
        "INST_DPD_SUM":               "INST_DPD_SUM",
        "INST_IS_LATE_SUM":           "INST_LATE_COUNT",
        "INST_IS_LATE_MEAN":          "INST_LATE_RATIO",
        "INST_PAYMENT_DIFF_MEAN":     "INST_PAYMENT_DIFF_MEAN",
        "INST_PAYMENT_DIFF_MIN":      "INST_PAYMENT_DIFF_MIN",
        "INST_IS_UNDERPAID_SUM":      "INST_UNDERPAID_COUNT",
        "INST_IS_UNDERPAID_MEAN":     "INST_UNDERPAID_RATIO",
        "INST_AMT_PAYMENT_MEAN":      "INST_AMT_PAYMENT_MEAN",
        "INST_AMT_PAYMENT_SUM":       "INST_AMT_PAYMENT_SUM",
        "INST_AMT_INSTALMENT_SUM":    "INST_AMT_INSTALMENT_SUM",
    })

    # -- Derived ratio: total paid / total due ---------------------------------
    inst_agg["INST_PAYMENT_FULFILLMENT"] = (
        inst_agg["INST_AMT_PAYMENT_SUM"]
        / inst_agg["INST_AMT_INSTALMENT_SUM"].clip(lower=1)
    )

    n_features = len(inst_agg.columns) - 1
    print(f"  Installments aggregation: {n_features} features for {len(inst_agg):,} clients")

    return inst_agg


# ====================================================================
# STEP 6 — Credit card balance aggregations
# ====================================================================

def aggregate_credit_card(cc: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the credit_card_balance table to one row per SK_ID_CURR.

    Why this matters:
        Credit card behavior reveals spending discipline.  A client who
        consistently maxes out their credit limit or carries a high
        balance-to-limit ratio is under financial stress — even if their
        loan application looks fine on paper.

    Derived columns (computed before aggregation):
        - UTILIZATION: AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL
          How much of the available limit is used (0-1+).
          >1 means over-limit.

    Aggregated features (all prefixed CC_):
        Usage patterns:
            - UTILIZATION_MEAN     : average utilization across all months
            - UTILIZATION_MAX      : peak utilization (worst month)
            - BALANCE_MEAN         : average outstanding balance
            - BALANCE_MAX          : highest balance ever carried
            - CREDIT_LIMIT_MEAN    : average credit limit
        Payment behavior:
            - AMT_PAYMENT_MEAN     : average monthly payment
            - MIN_PAYMENT_MEAN     : average minimum required payment
            - PAYMENT_MIN_DIFF     : avg(actual payment - minimum required)
        Delinquency:
            - SK_DPD_MAX           : worst days past due
            - SK_DPD_MEAN          : average days past due
            - SK_DPD_DEF_MAX       : worst DPD with tolerance (Home Credit's stricter measure)
            - HAS_DPD              : binary flag — any DPD > 0?
        Volume:
            - MONTH_COUNT          : total monthly records
            - DRAWINGS_MEAN        : average number of drawings (card usage frequency)

    Parameters
    ----------
    cc : pd.DataFrame
        Raw credit_card_balance table.

    Returns
    -------
    pd.DataFrame
        One row per SK_ID_CURR with aggregated features.
    """
    cc = cc.copy()

    # -- Derive utilization ratio before aggregation ---------------------------
    cc["UTILIZATION"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"].clip(lower=1)

    # -- GroupBy aggregation ---------------------------------------------------
    agg_spec = {
        "MONTHS_BALANCE":           ["count"],                    # total records
        "UTILIZATION":              ["mean", "max"],              # credit usage
        "AMT_BALANCE":              ["mean", "max"],              # balances
        "AMT_CREDIT_LIMIT_ACTUAL":  ["mean"],                    # limit
        "AMT_PAYMENT_CURRENT":      ["mean"],                    # payments
        "AMT_INST_MIN_REGULARITY":  ["mean"],                    # minimum payment due
        "CNT_DRAWINGS_CURRENT":     ["mean"],                    # card usage frequency
        "SK_DPD":                   ["max", "mean"],             # delinquency
        "SK_DPD_DEF":               ["max"],                     # strict delinquency
    }

    cc_agg = cc.groupby("SK_ID_CURR").agg(agg_spec)

    # Flatten column names
    cc_agg.columns = [
        f"CC_{col}_{stat}".upper()
        for col, stat in cc_agg.columns
    ]
    cc_agg = cc_agg.reset_index()

    # Rename for readability
    cc_agg = cc_agg.rename(columns={
        "CC_MONTHS_BALANCE_COUNT":          "CC_MONTH_COUNT",
        "CC_UTILIZATION_MEAN":              "CC_UTILIZATION_MEAN",
        "CC_UTILIZATION_MAX":               "CC_UTILIZATION_MAX",
        "CC_AMT_BALANCE_MEAN":              "CC_BALANCE_MEAN",
        "CC_AMT_BALANCE_MAX":               "CC_BALANCE_MAX",
        "CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN":  "CC_CREDIT_LIMIT_MEAN",
        "CC_AMT_PAYMENT_CURRENT_MEAN":      "CC_AMT_PAYMENT_MEAN",
        "CC_AMT_INST_MIN_REGULARITY_MEAN":  "CC_MIN_PAYMENT_MEAN",
        "CC_CNT_DRAWINGS_CURRENT_MEAN":     "CC_DRAWINGS_MEAN",
        "CC_SK_DPD_MAX":                    "CC_SK_DPD_MAX",
        "CC_SK_DPD_MEAN":                   "CC_SK_DPD_MEAN",
        "CC_SK_DPD_DEF_MAX":                "CC_SK_DPD_DEF_MAX",
    })

    # -- Derived features ------------------------------------------------------
    # How much more than the minimum does the client pay on average?
    cc_agg["CC_PAYMENT_MIN_DIFF"] = (
        cc_agg["CC_AMT_PAYMENT_MEAN"] - cc_agg["CC_MIN_PAYMENT_MEAN"]
    )
    # Binary: any delinquency?
    cc_agg["CC_HAS_DPD"] = (cc_agg["CC_SK_DPD_MAX"] > 0).astype(np.int8)

    n_features = len(cc_agg.columns) - 1
    print(f"  Credit card aggregation: {n_features} features for {len(cc_agg):,} clients")

    return cc_agg


# ====================================================================
# STEP 7 — POS_CASH_balance aggregations
# ====================================================================

def aggregate_pos_cash(pos: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the POS_CASH_balance table to one row per SK_ID_CURR.

    Why this matters:
        POS (point-of-sale) and cash loans are tracked monthly here.
        Each row is one month of one loan.  The key signal is whether
        the client completed their loans on time or fell behind.
        A pattern of early completion signals discipline; repeated
        delinquency signals trouble.

    Aggregated features (all prefixed POS_):
        Delinquency:
            - SK_DPD_MAX           : worst days past due
            - SK_DPD_MEAN          : average days past due
            - SK_DPD_DEF_MAX       : worst DPD (strict)
            - HAS_DPD              : binary — any late month?
        Loan progress:
            - MONTH_COUNT          : total monthly records
            - COMPLETED_COUNT      : how many loans reached "Completed" status
            - ACTIVE_COUNT         : how many are still "Active"
            - COMPLETED_RATIO      : completed / total distinct loans
        Installment info:
            - CNT_INSTALMENT_MEAN  : average total installment count
            - CNT_INSTALMENT_FUTURE_MEAN : average remaining installments
            - REMAINING_RATIO      : remaining / total installments

    Parameters
    ----------
    pos : pd.DataFrame
        Raw POS_CASH_balance table.

    Returns
    -------
    pd.DataFrame
        One row per SK_ID_CURR with aggregated features.
    """
    # -- GroupBy aggregation ---------------------------------------------------
    agg_spec = {
        "MONTHS_BALANCE":        ["count"],                   # total records
        "SK_DPD":                ["max", "mean"],             # delinquency
        "SK_DPD_DEF":            ["max"],                     # strict delinquency
        "CNT_INSTALMENT":        ["mean"],                    # loan length
        "CNT_INSTALMENT_FUTURE": ["mean"],                    # remaining payments
    }

    pos_agg = pos.groupby("SK_ID_CURR").agg(agg_spec)

    # Flatten column names
    pos_agg.columns = [
        f"POS_{col}_{stat}".upper()
        for col, stat in pos_agg.columns
    ]
    pos_agg = pos_agg.reset_index()

    # Rename for readability
    pos_agg = pos_agg.rename(columns={
        "POS_MONTHS_BALANCE_COUNT":         "POS_MONTH_COUNT",
        "POS_SK_DPD_MAX":                   "POS_SK_DPD_MAX",
        "POS_SK_DPD_MEAN":                  "POS_SK_DPD_MEAN",
        "POS_SK_DPD_DEF_MAX":               "POS_SK_DPD_DEF_MAX",
        "POS_CNT_INSTALMENT_MEAN":          "POS_CNT_INSTALMENT_MEAN",
        "POS_CNT_INSTALMENT_FUTURE_MEAN":   "POS_CNT_INSTALMENT_FUTURE_MEAN",
    })

    # -- Status counts (from NAME_CONTRACT_STATUS) -----------------------------
    completed = (
        pos.groupby("SK_ID_CURR")["NAME_CONTRACT_STATUS"]
        .apply(lambda x: (x == "Completed").sum())
        .rename("POS_COMPLETED_COUNT")
    )
    active = (
        pos.groupby("SK_ID_CURR")["NAME_CONTRACT_STATUS"]
        .apply(lambda x: (x == "Active").sum())
        .rename("POS_ACTIVE_COUNT")
    )

    pos_agg = pos_agg.merge(completed, on="SK_ID_CURR", how="left")
    pos_agg = pos_agg.merge(active,    on="SK_ID_CURR", how="left")

    # -- Derived features ------------------------------------------------------
    pos_agg["POS_HAS_DPD"] = (pos_agg["POS_SK_DPD_MAX"] > 0).astype(np.int8)

    pos_agg["POS_COMPLETED_RATIO"] = (
        pos_agg["POS_COMPLETED_COUNT"]
        / pos_agg["POS_MONTH_COUNT"].clip(lower=1)
    )

    pos_agg["POS_REMAINING_RATIO"] = (
        pos_agg["POS_CNT_INSTALMENT_FUTURE_MEAN"]
        / pos_agg["POS_CNT_INSTALMENT_MEAN"].clip(lower=1)
    )

    n_features = len(pos_agg.columns) - 1
    print(f"  POS_CASH aggregation: {n_features} features for {len(pos_agg):,} clients")

    return pos_agg
