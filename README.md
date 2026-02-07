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
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── .gitignore
│
├── docs/                       # Research & raw data (not tracked in git)
│   ├── AI Risk Analizi.docx    #   Pre-research report & requirements analysis
│   └── dataset/                #   Raw CSV files from Kaggle
│
├── src/                        # Reusable source modules
│   ├── __init__.py
│   └── utils.py                #   Memory optimization & data loading utilities
│
└── notebooks/                  # Jupyter analysis notebooks
    ├── 01_eda.ipynb            #   EDA - pre-executed with all outputs & plots
    └── plots/                  #   Saved visualizations (not tracked in git)
        ├── target_distribution.png
        ├── missing_values_train.png
        ├── numeric_distributions.png
        ├── age_analysis.png
        ├── income_credit_analysis.png
        ├── categorical_analysis.png
        ├── correlation_analysis.png
        ├── dpd_analysis.png
        ├── cc_utilization.png
        └── prev_app_analysis.png
```

> **Note:** `docs/` and `notebooks/plots/` are excluded from git via `.gitignore`.
> The dataset must be downloaded separately from Kaggle (see [Setup](#setup--installation)).
> The notebook `01_eda.ipynb` is pre-executed — open it to view all outputs and charts without re-running.

---

## Completed Steps

### 1. Exploratory Data Analysis (EDA) — `notebooks/01_eda.ipynb`

Comprehensive analysis of all 8 tables using a 20% sample for fast iteration:

- **Memory optimization**: Downcasting float64/int64 types to reduce RAM usage (~30-50% reduction per table)
- **Target distribution**: Confirmed ~8% default rate with 1:11 class imbalance
- **Missing value analysis**: 41 columns >50% missing (drop candidates), 9 at 20-50%, 16 at <20%
- **Correlation analysis**: `EXT_SOURCE_1/2/3` are the strongest predictors (r = -0.16 to -0.17)
- **Categorical insights**: Highest default rates in low-skill laborers (17%), renters (13.3%), lower secondary education (10.8%)
- **Auxiliary table deep-dives**:
  - `installments_payments` — 8.6% late payments; DPD is a strong default signal
  - `credit_card_balance` — High utilization (>70%) -> 15.4% default vs. low (<30%) -> 6.0%
  - `bureau` — Avg 5.5 external credits per client; 0.26% with overdue records
  - `previous_application` — 62.7% approved, 17.4% refused

---

## Upcoming Steps

### 2. Feature Engineering
- Aggregate features from auxiliary tables (mean/max/count per customer)
- DPD statistics, payment discipline metrics, credit utilization ratios
- Time-windowed features (last 3/6/12 months behavior)
- Interaction features (credit-to-income ratio, annuity-to-income ratio)

### 3. Data Preprocessing
- Missing value imputation strategy (threshold-based drop + median/mode fill)
- Categorical encoding (Label Encoding / Target Encoding)
- Outlier handling

### 4. Model Training
- LightGBM and XGBoost baseline models
- Stratified K-Fold cross-validation
- Hyperparameter tuning (Optuna)
- Class imbalance handling (SMOTE / class_weight / scale_pos_weight)

### 5. Model Evaluation & Explainability
- AUC-ROC, Precision-Recall curves
- SHAP values for feature importance and individual predictions
- Calibration of probability scores to 0-100 risk scale

### 6. Risk Scoring API
- Dynamic risk score (0-100) for each customer
- Top-3 contributing factors per prediction
- New transaction simulation endpoint
- Proactive alert system for sudden risk increases

---

## Key EDA Findings

| Finding | Detail |
|---------|--------|
| Class Imbalance | ~8% default rate, 1:11 ratio |
| Strongest Predictors | `EXT_SOURCE_3` (r=-0.174), `EXT_SOURCE_2` (r=-0.167), `EXT_SOURCE_1` (r=-0.160) |
| Age Effect | Younger clients (20-30) have highest default rates; risk decreases with age |
| Employment Effect | Longer employment duration correlates with lower default risk |
| Riskiest Occupation | Low-skill Laborers: 17.0% default rate |
| Riskiest Housing | Rented apartment: 13.3% default rate |
| Credit Card Utilization | >70% utilization -> 15.4% default vs. <30% -> 6.0% default |
| Payment Delays (DPD) | 8.6% of installments are late; strong signal for default prediction |
| Missing Values | 41 columns with >50% missing; careful imputation strategy needed |

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

### Viewing the EDA Notebook

The notebook ships pre-executed with all outputs embedded. Simply open it:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

To re-run the analysis from scratch (takes ~5 min due to 2.56 GB dataset):

```bash
cd notebooks
jupyter nbconvert --to notebook --execute 01_eda.ipynb --output 01_eda.ipynb
```

---

## Tech Stack

- **Python 3.11+**
- **pandas / NumPy** — Data manipulation and analysis
- **matplotlib / seaborn** — Data visualization
- **scikit-learn** — Preprocessing and evaluation metrics
- **LightGBM / XGBoost** — Gradient boosting models
- **SHAP** — Model explainability

---

## References

- [Home Credit Default Risk — Kaggle Competition](https://www.kaggle.com/competitions/home-credit-default-risk)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
