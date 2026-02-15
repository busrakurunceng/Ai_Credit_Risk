# AI Credit Risk - Intelligent Risk Management System

An AI-powered credit risk scoring system built with machine learning. This project uses the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) dataset from Kaggle to develop a dynamic risk scoring engine that predicts loan default probability.

---

## Dataset Overview

The project uses the **Home Credit Default Risk** competition dataset, a multi-table relational dataset containing loan application data along with supplementary behavioral and financial history.

### Data Files

| File | Rows | Columns | Size | Description |
|------|------|---------|------|-------------|
| `application_train.csv` | 307,511 | 122 | 158 MB | Main training data with TARGET label (1 = default, 0 = no default) |
| `application_test.csv` | 48,744 | 121 | 25 MB | Test data (no TARGET column) |
| `bureau.csv` | 1,716,428 | 17 | 162 MB | Credit bureau records from other financial institutions |
| `bureau_balance.csv` | 27,299,925 | 3 | 358 MB | Monthly balance history for bureau credits |
| `previous_application.csv` | 1,670,214 | 37 | 386 MB | Previous loan applications at Home Credit |
| `installments_payments.csv` | 13,605,401 | 8 | 690 MB | Installment payment history (actual vs. expected) |
| `credit_card_balance.csv` | 3,840,312 | 23 | 405 MB | Monthly credit card balance snapshots |
| `POS_CASH_balance.csv` | 10,001,358 | 8 | 375 MB | Monthly POS and cash loan balance snapshots |

**Total size: ~2.56 GB across 10 files**

### Table Relationships

```
application_train/test (SK_ID_CURR)
├── bureau (SK_ID_CURR -> SK_ID_BUREAU)
│   └── bureau_balance (SK_ID_BUREAU)
├── previous_application (SK_ID_CURR -> SK_ID_PREV)
│   ├── installments_payments (SK_ID_PREV)
│   ├── POS_CASH_balance (SK_ID_PREV)
│   └── credit_card_balance (SK_ID_PREV)
```

### Target Variable

- **TARGET = 0**: Client repaid the loan (no payment difficulties)
- **TARGET = 1**: Client had payment difficulties (default)
- **Class imbalance**: ~8% default rate (ratio ~1:11)

---

## Project Structure

```
Ai_Credit_Risk/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore
├── app.py                             # Streamlit dashboard (risk scoring UI)
│
├── docs/                              # Research & raw data (not tracked in git)
│   ├── AI Risk Analizi.docx           #   Pre-research report
│   └── dataset/                       #   Raw CSV files from Kaggle
│
├── src/                               # Reusable source modules
│   ├── __init__.py
│   ├── utils.py                       #   Memory optimization & data loading utilities
│   ├── feature_engineering.py         #   All FE functions (8 steps)
│   └── build_features.py             #   Full pipeline orchestration script
│
├── data/                              # Feature & processed files (not tracked in git)
│   ├── train_featured.parquet         #   307,511 x 212 (64.3 MB) — after FE
│   ├── test_featured.parquet          #   48,744 x 211  (11.6 MB) — after FE
│   ├── train_processed.parquet        #   307,511 x 169 (54.7 MB) — after preprocessing
│   ├── test_processed.parquet         #   48,744 x 168  (10.1 MB) — after preprocessing
│   ├── oof_predictions.parquet        #   307,511 x 3 — Raw OOF predictions
│   ├── test_predictions.parquet       #   48,744 x 2 — Raw test ensemble predictions
│   ├── test_predictions_full.parquet  #   48,744 x 2 — Raw full-model test predictions
│   ├── oof_predictions_calibrated.parquet  #  Calibrated OOF (raw + Platt-scaled)
│   └── test_predictions_calibrated.parquet #  Calibrated test (ensemble + full, raw + scaled)
│
├── models/                            # Trained models & artifacts (not tracked in git)
│   ├── lgbm_best_fold.pkl            #   Best single fold LightGBM model
│   ├── lgbm_all_folds.pkl            #   All 5 fold models (ensemble)
│   ├── lgbm_full_model.pkl           #   Full-data single model (production)
│   ├── platt_calibrator.pkl          #   Platt scaling for probability calibration
│   ├── best_params.json              #   Tuned hyperparameters + CV scores + thresholds
│   ├── shap_values.npy               #   SHAP values (5K sample)
│   └── shap_sample.parquet           #   SHAP sample features
│
└── notebooks/                         # Jupyter analysis notebooks
    ├── 01_eda.ipynb                   #   EDA - pre-executed with all outputs
    ├── 02_feature_engineering.ipynb    #   Feature engineering - pre-executed
    ├── 03_preprocessing.ipynb         #   Data preprocessing - pre-executed
    ├── 04_modeling.ipynb              #   Model training, tuning, calibration & SHAP
    └── plots/                         #   Saved visualizations (not tracked in git)
```

> **Note:** `docs/`, `data/`, `models/`, and `notebooks/plots/` are excluded from git via `.gitignore`.
> The dataset must be downloaded separately from Kaggle (see [Setup](#setup--installation)).
> Notebooks are pre-executed — open them to view all outputs without re-running.

---

## Completed Steps

### 1. Exploratory Data Analysis (EDA)

**Notebook:** `notebooks/01_eda.ipynb` | **Notes:** `notebooks/01_EDA_PROJE_NOTLARI.txt`

Comprehensive analysis of all 8 tables using a 20% sample for fast iteration:

- **Memory optimization**: Downcasting float64/int64 types to reduce RAM usage (~30-50% reduction per table)
- **Target distribution**: Confirmed ~8% default rate with 1:11 class imbalance
- **Missing value analysis**: 41 columns >50% missing (drop candidates), 9 at 20-50%, 16 at <20%
- **Correlation analysis**: `EXT_SOURCE_1/2/3` are the strongest predictors (r = -0.16 to -0.17)
- **Categorical insights**: Highest default rates in low-skill laborers (17%), renters (13.3%)
- **Auxiliary table deep-dives**: DPD analysis, credit card utilization, bureau history, previous applications

### 2. Feature Engineering

**Notebook:** `notebooks/02_feature_engineering.ipynb` | **Notes:** `notebooks/02_FE_PROJE_NOTLARI.txt`  
**Pipeline:** `src/build_features.py` | **Functions:** `src/feature_engineering.py`

Engineered **90 new features** from all 6 auxiliary tables, expanding the dataset from 122 to 212 columns:

| Step | Source | Features | Technique |
|------|--------|----------|-----------|
| 1 | Application table | 10 | Row-level ratios: credit/income, annuity/income, age, employment years |
| 2 | DAYS_EMPLOYED cleanup | 1 | Anomaly detection: 365,243 sentinel replaced with median + flag |
| 3 | Bureau | 16 | GroupBy aggregation: external credit counts, overdue history, debt ratio |
| 4 | Previous application | 14 | GroupBy aggregation: approval rate, application amounts, decision timing |
| 5 | Installments payments | 14 | DPD & payment difference: late ratio, underpaid ratio, fulfillment |
| 6 | Credit card balance | 14 | Utilization ratio, DPD, payment vs minimum, drawing frequency |
| 7 | POS_CASH balance | 11 | Loan completion ratio, DPD, remaining installments |
| 8 | Bureau balance | 10 | Monthly DPD severity, status distribution (on-time/closed/unknown) |

**Full pipeline output** (run on all data, not sampled):

| File | Rows | Columns | Size |
|------|------|---------|------|
| `data/train_featured.parquet` | 307,511 | 212 | 64.3 MB |
| `data/test_featured.parquet` | 48,744 | 211 | 11.6 MB |

Pipeline runtime: ~5.4 minutes total.

### 3. Data Preprocessing

**Notebook:** `notebooks/03_preprocessing.ipynb`

Cleaned and transformed the 212-column featured dataset into a model-ready 169-column dataset:

| Step | Action | Details |
|------|--------|---------|
| Missing value drops | Dropped 40 columns | High-missing housing MODE/MEDI columns, weak CC features, categorical columns with >50% missing |
| Missing flags | Added `_MISSING` indicator columns | `EXT_SOURCE_1`, housing AVG columns, `OWN_CAR_AGE` — preserves missingness as signal |
| Imputation | Median (numeric) / Mode (categorical) | Train-derived medians applied to both train & test; count/flag NaNs filled with 0 |
| Outlier handling | Winsorize %1-%99 | 113 columns clipped; FLAG_* columns excluded to preserve binary integrity |
| Categorical encoding | Label (≤4 classes) + Frequency (>4) | 5 label-encoded (train-only fit, no leakage), 8 frequency-encoded with `_MISSING` flags |
| Multicollinearity | Drop one of pairs with \|corr\| > 0.95 | 30 pairs found, 25 columns dropped (kept the one with higher target correlation) |

**Preprocessing output** (212 → 169 columns, 0 missing, 0 inf, 0 categorical):

| File | Rows | Columns | Size |
|------|------|---------|------|
| `data/train_processed.parquet` | 307,511 | 169 | 54.7 MB |
| `data/test_processed.parquet` | 48,744 | 168 | 10.1 MB |

---

### 4. Model Training, Calibration & Explainability

**Notebook:** `notebooks/04_modeling.ipynb`

Full modeling pipeline with 12 steps:

| Step | Action | Details |
|------|--------|---------|
| Baseline LightGBM | Stratified 5-Fold CV + `scale_pos_weight=11.39` | AUC=0.7842, PR-AUC=0.2753 |
| Baseline XGBoost | Same CV setup, `tree_method='hist'` | AUC=0.7814, PR-AUC=0.2727 |
| Model Selection | AUC/PR-AUC comparison | **LightGBM selected** (higher AUC, faster) |
| Optuna AUC Tuning | 50 trials × 5-fold, 9 hyperparameters | Best AUC=0.7872 (+30 bps) |
| Optuna PR-AUC Tuning | 50 trials × 5-fold, `average_precision` objective | Best PR-AUC=0.2796 (+44 bps) |
| Study Comparison | AUC vs PR-AUC optimized params | **PR-AUC-optimized selected** (AUC loss only -1.7 bps) |
| Leakage & Stability | Correlation, dominance, fold variance checks | No leakage, stable folds (CV<0.5) |
| Final Model | PR-AUC-aligned early stopping + full-data model | **AUC=0.7868, PR-AUC=0.2802, +26 bps** |
| Calibration | Brier score, reliability diagram, Platt scaling | Brier Skill: -1.26 → **+0.11** |
| Threshold Optimization | Business cost matrix (FN:10×, FP:1×) on calibrated OOF | Cost-optimal: **0.090** (Recall 68%) |
| SHAP Analysis | Global summary + individual waterfalls | Top: EXT_SOURCE_2/3, CREDIT_TERM, GOODS_CREDIT_DIFF |
| Export | 12 artifact files | Models, predictions, calibrated outputs, SHAP values |

**Key results:**

| Metric | Baseline | Tuned | Improvement |
|--------|----------|-------|-------------|
| CV AUC | 0.78422 | 0.78682 | +26.0 bps |
| CV PR-AUC | 0.27528 | 0.28015 | +48.7 bps |
| Brier Score | 0.168 (raw) | 0.066 (calibrated) | -60.7% |
| Brier Skill | -1.264 | +0.109 | Business-ready |

**Output files (12 artifacts):**

| File | Description |
|------|-------------|
| `models/lgbm_best_fold.pkl` | Best single fold model |
| `models/lgbm_all_folds.pkl` | All 5 fold models (ensemble) |
| `models/lgbm_full_model.pkl` | Full-data single model (production) |
| `models/platt_calibrator.pkl` | Platt scaling for probability calibration |
| `models/best_params.json` | Hyperparameters + all metrics + thresholds |
| `models/shap_values.npy` | SHAP values (5K samples) |
| `data/oof_predictions.parquet` | Raw OOF predictions (307,511 rows) |
| `data/test_predictions.parquet` | Raw test ensemble (48,744 rows) |
| `data/test_predictions_full.parquet` | Raw full-model test predictions |
| `data/oof_predictions_calibrated.parquet` | Calibrated OOF predictions |
| `data/test_predictions_calibrated.parquet` | Calibrated test (ensemble + full model) |

---

### 5. Streamlit Dashboard

**App:** `app.py`

Interactive credit risk analysis dashboard built with Streamlit and Plotly, featuring a corporate finance theme with Turkish-language UI. Currently runs with mock data — ready for backend model integration.

| Component | Description |
|-----------|-------------|
| **Sidebar** | Customer profile selector (3 mock profiles), JSON file upload for custom data, project info |
| **KPI Panel** | 4 metric cards: Credit Amount, Annual Income, Current Debt, External Credit Score |
| **Risk Gauge** | Plotly gauge chart (0-100 scale) with color-coded zones: Green (≤30), Orange (31-60), Red (61-100) |
| **Simulation Panel** | Interactive scenario analysis — adjustable income, credit amount, credit term, employment years, late payment ratio with real-time risk recalculation |
| **SHAP Analysis** | Horizontal bar chart showing top feature contributions (red = risk increase, green = risk decrease) |
| **Payment History** | 12-month DPD (Days Past Due) timeline with color-coded markers |

**Mock customer profiles:**

| Profile | Risk Score | Risk Level |
|---------|-----------|------------|
| Customer #10234 | 18 | Low Risk (Green) |
| Customer #20567 | 47 | Medium Risk (Orange) |
| Customer #35891 | 82 | High Risk (Red) |

---

## Upcoming Steps

### 6. Backend API Integration
- FastAPI REST API with `/predict`, `/explain`, `/simulate` endpoints
- Replace mock data with real model inference (LightGBM + Platt calibration)
- Dynamic risk score (0-100): raw prediction → Platt calibration → score
- Top-3 contributing factors per prediction (SHAP-based explainability)
- Business threshold integration (cost-optimal: 0.090, F1-optimal: 0.178)
- Fairness analysis for `CODE_GENDER` (top-5 SHAP feature, sensitive attribute)

---

## Key Findings

### EDA Findings

| Finding | Detail |
|---------|--------|
| Class Imbalance | ~8% default rate, 1:11 ratio |
| Strongest Predictors | `EXT_SOURCE_3` (r=-0.174), `EXT_SOURCE_2` (r=-0.167), `EXT_SOURCE_1` (r=-0.160) |
| Age Effect | Younger clients (20-30) have highest default rates; risk decreases with age |
| Employment Effect | Longer employment duration correlates with lower default risk |
| Credit Card Utilization | >70% utilization -> 15.4% default vs. <30% -> 6.0% default |
| Payment Delays (DPD) | 8.6% of installments are late; strong signal for default prediction |
| Missing Values | 41 columns with >50% missing; careful imputation strategy needed |

### Feature Engineering Findings

| Finding | Detail |
|---------|--------|
| DAYS_EMPLOYED Anomaly | 18% of records had sentinel value 365,243 — cleaned with median + flag |
| Bureau Coverage | 14.3% of clients have no external credit history (NaN filled as 0 credits) |
| Credit Card Usage | Only 28.2% of clients have credit card records — sparse but informative |
| Installment Behavior | Average 46 payment records per client — richest behavioral signal |
| Approval History | Clients with low past approval rates show higher default risk |
| Bureau Balance | Monthly DPD severity (levels 1-5) adds temporal depth beyond bureau snapshot |

### Preprocessing Findings

| Finding | Detail |
|---------|--------|
| Column reduction | 212 → 169 columns (40 high-missing dropped, 25 multicollinear removed, missing flags added) |
| FLAG integrity | 27 FLAG_* columns preserved as binary [0,1] — excluded from winsorization |
| Outlier impact | Winsorization reduced highly skewed columns from 70 → 36 (17 excluding FLAGs) |
| Zero-inflated features | 17 remaining skewed columns are DPD/overdue features — expected, tree-safe |
| RATIO columns | All 14 ratio columns have non-negative min values after clipping |
| Multicollinearity | Key drops: `AGE_YEARS` (duplicate of `DAYS_BIRTH`), `AMT_CREDIT` (≈`AMT_GOODS_PRICE`) |

### Modeling Findings

| Finding | Detail |
|---------|--------|
| LightGBM > XGBoost | LightGBM wins both AUC (+28 bps) and PR-AUC (+26 bps), and runs 1.6× faster |
| Dual Optuna strategy | AUC-optimized → shallow trees (depth=4); PR-AUC-optimized → deep trees (depth=8) with stronger regularization |
| No leakage detected | Max feature-target correlation: 0.16 (EXT_SOURCE_2); no dominant feature (top-1 = 4.6%) |
| Calibration critical | `scale_pos_weight` inflates probabilities 5-8×; Platt scaling fixes Brier Skill from -1.26 to +0.11 |
| SHAP top features | EXT_SOURCE_2/3 (\|SHAP\|=0.31/0.28), GOODS_CREDIT_DIFF (0.13), CODE_GENDER (0.12), CREDIT_TERM (0.11) |
| Threshold sensitivity | Cost-optimal (FN:10×) = 0.090 catches 68% defaults; F1-optimal = 0.178 balances precision/recall |
| Full model correlation | Ensemble vs full-data single model: r=0.994 — nearly identical rankings |
| Fairness concern | `CODE_GENDER` in SHAP top-5 — requires fairness audit before production deployment |

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Ai_Credit_Risk.git
cd Ai_Credit_Risk

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle and place CSV files in docs/dataset/
# https://www.kaggle.com/competitions/home-credit-default-risk/data
```

### Running the Streamlit Dashboard

```bash
# Launch the interactive credit risk dashboard (opens in browser)
streamlit run app.py
```

### Running the Feature Engineering Pipeline

```bash
# Build features on full data (saves to data/ directory, ~5 min)
python -m src.build_features
```

### Viewing Notebooks

Notebooks ship pre-executed with all outputs embedded:

```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_feature_engineering.ipynb
jupyter notebook notebooks/03_preprocessing.ipynb
jupyter notebook notebooks/04_modeling.ipynb
```

---

## Tech Stack

- **Python 3.11+**
- **pandas / NumPy** — Data manipulation and analysis
- **matplotlib / seaborn** — Data visualization
- **Streamlit** — Interactive web dashboard
- **Plotly** — Interactive gauge & chart visualizations
- **scikit-learn** — Preprocessing and evaluation metrics
- **LightGBM / XGBoost** — Gradient boosting models
- **SHAP** — Model explainability
- **Optuna** — Bayesian hyperparameter optimization

---

## References

- [Home Credit Default Risk — Kaggle Competition](https://www.kaggle.com/competitions/home-credit-default-risk)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)
