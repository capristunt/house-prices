# House Prices: Advanced Regression Techniques

End-to-end machine learning pipeline for the
[Ames Housing Kaggle competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques),
predicting residential property sale prices from 79 explanatory variables.

## Results

| Notebook | Model | CV RMSE (log) | Features Used |
|----------|-------|---------------|---------------|
| 03 | Linear Regression | 0.1267 | 205 |
| 03 | Ridge (L2) | 0.1215 | 205 |
| 03 | Lasso (L1) | 0.1190 | 87 |
| 04 | Decision Tree | 0.1947 | 205 |
| 04 | Random Forest | 0.1465 | 205 |
| 04 | Gradient Boosting | 0.1205 | 205 |
| 04 | XGBoost (tuned) | 0.1164 | 205 |
| 04 | Stacking (Lasso + XGBoost) | 0.1122 | 205 |
| 05 | Ridge | 0.1096 | 144 |
| 05 | CatBoost (Optuna) | 0.1063 | 144 |
| 05 | XGBoost (Optuna) | 0.1086 | 144 |
| 05 | LightGBM (Optuna) | 0.1100 | 144 |
| **05** | **Stacking (7 models)** | **0.1043** | **144** |

Best single model: CatBoost with Optuna TPE tuning (0.1063).
Best overall: 7-model stacking ensemble (Ridge, Lasso, ElasticNet, XGBoost, LightGBM, CatBoost, GBR) with Ridge meta-learner (0.1043).

## Project Structure

```
house-prices/
├── data/
│   ├── raw/                    # Original Kaggle train.csv and test.csv
│   └── processed/              # Cleaned datasets exported by 02_Features
│       ├── X_train.csv         # 1458 × 205
│       ├── X_test.csv          # 1459 × 205
│       ├── y_train.csv         # Log-transformed target
│       └── test_ids.csv        # Test set IDs for submission
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   ├── 02_Features.ipynb       # Feature Engineering Pipeline
│   ├── 03_Modeling.ipynb       # Linear Models (Ridge, Lasso, ElasticNet)
│   ├── 04_Modeling_Trees.ipynb # Tree-Based Models & Stacking
│   └── 05_Advanced_Modeling.ipynb # Advanced Ensemble & Hyperparameter Optimization
├── submissions/
│   ├── submission_lasso.csv
│   ├── submission_stacking.csv
│   ├── submission_blending.csv
│   └── submission_final.csv
└── README.md
```

## Pipeline Overview

### 01 — Exploratory Data Analysis

Full exploration of distributions, missing values, and feature relationships. Key findings
that drive all downstream decisions:

- Missing value taxonomy: 15 categorical and 7 numerical features with semantic NaN (absence of a
  feature, e.g. no garage) vs. true missing data requiring imputation
- Outlier detection: 2 observations with GrLivArea > 4,000 sq ft and SalePrice < $300k flagged
  for removal
- Multicollinearity clusters: garage features (r=0.88), surface features, and temporal features
  identified for consolidation
- Statistical validation: Kruskal-Wallis tests confirm which categorical features significantly
  discriminate SalePrice

### 02 — Feature Engineering

Nine-step transformation pipeline applied to the concatenated train+test set:

- Outlier removal (2 observations)
- Three-pass missing value imputation (semantic categorical → semantic numerical → true missing)
- Target transformation: `log1p(SalePrice)` to correct strong positive skewness
- 6 composite features: TotalSF, TotalBath, TotalPorchSF, HasRemodel, HouseAge, RemodAge
- Ordinal encoding (15 quality features) and one-hot encoding (28 nominal features)
- Skewness correction: `log1p` applied to 18 continuous features
- Redundancy removal: 11 features dropped based on EDA diagnostics

Output: 205 features, zero missing values, ready for modeling.

### 03 — Linear Models

Evaluation framework: 5-Fold CV with RMSE in log space (equivalent to Kaggle RMSLE).
All models wrapped in a StandardScaler pipeline to prevent data leakage.

- Baseline LinearRegression establishes reference performance (0.1267)
- RidgeCV and LassoCV with built-in alpha search via internal cross-validation
- Lasso emerges as the best linear model (0.1190), using only 87 of 205 features
- Residual analysis reveals heavy-tailed errors on atypical properties, motivating
  tree-based models

### 04 — Tree-Based Models & Stacking

Progressive exploration from a single decision tree to a stacking ensemble:

- Decision tree baseline demonstrates overfitting (0.195) and motivates ensemble methods
- Random Forest (bagging) reduces variance but plateaus at 0.147
- Gradient Boosting and XGBoost (boosting) approach Lasso-level performance
- `RandomizedSearchCV` tunes XGBoost to 0.1164, surpassing Lasso for the first time
- Stacking (Lasso + XGBoost tuned, Ridge meta-learner) achieves best score: 0.1122
- Residual std drops 25% compared to Lasso (0.105 → 0.078)

### 05 — Advanced Ensemble & Hyperparameter Optimization

Supersedes notebooks 03 and 04 by rebuilding the full pipeline from raw data with four
simultaneous improvements:

**Enhanced feature engineering (Part A):**
- 5 additional outliers removed (total: 7), identified via multi-feature residual analysis
  on observations flagged by 3+ key features simultaneously
- 6 interaction features capturing nonlinear relationships: Qual×TotalSF, Qual×GrLivArea,
  Qual×HouseAge, Garage×Cars, SF_per_Room, Has2ndFloor
- 1 composite quality index (QualityIndex) averaging the four most price-correlated quality features
- KFold target encoding with smoothing on 7 high-cardinality categoricals (Neighborhood,
  MSSubClass, Condition1, Exterior1st, Exterior2nd, SaleType, SaleCondition), reducing
  dimensionality from 205 to 144 features

**Advanced modeling (Part B):**
- 2 new model families: LightGBM and CatBoost join the model pool
- Bayesian hyperparameter tuning with Optuna TPE sampler (80 to 100 trials per model) replaces
  RandomizedSearchCV, finding better configurations by concentrating trials in promising regions
- 10-fold CV replaces 5-fold for more stable RMSE estimates
- 7-model stacking ensemble (3 linear + 4 boosting) with Ridge meta-learner: 0.1043
- Optimized weighted blending via scipy.optimize as complementary ensemble strategy: 0.1047
- Final prediction averages stacking and blending outputs

**Key insight from weight optimization:** CatBoost (42%) and Ridge (38%) capture 80% of the
blending weight, confirming that ensemble value comes from combining fundamentally different
model families whose errors are complementary, not from adding similar models.

## Tech Stack

- **Python**: NumPy, pandas, matplotlib, scipy
- **Machine Learning**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Hyperparameter Optimization**: Optuna (TPE sampler)
- **Environment**: Jupyter Notebooks, VS Code
- **Version Control**: Git, GitHub

## How to Reproduce

```bash
git clone https://github.com/<username>/house-prices.git
cd house-prices
pip install numpy pandas matplotlib scikit-learn xgboost lightgbm catboost optuna jupyter
```

Run the notebooks in order: `01_EDA.ipynb` → `02_Features.ipynb` → `03_Modeling.ipynb` →
`04_Modeling_Trees.ipynb` → `05_Advanced_Modeling.ipynb`.

Notebooks 03 and 04 depend on the processed data exported by 02. Notebook 05 is self-contained
and loads raw data directly from `data/raw/`.

The raw Kaggle data (`train.csv`, `test.csv`) should be placed in `data/raw/`.

## Key Takeaways

- Feature engineering matters more than model complexity: Lasso with well-prepared features
  outperforms Random Forest with 500 trees
- Regularization is essential when features outnumber the observations-to-features ratio (~7:1):
  Lasso eliminated 118 of 205 features as noise
- Target encoding high-cardinality categoricals reduces dimensionality while preserving signal:
  144 features outperform 205 across all model families
- Interaction features (quality × size, quality × age) inject nonlinear relationships that
  significantly boost linear model performance
- Bayesian hyperparameter tuning (Optuna TPE) finds better configurations than random search
  by learning which regions of the hyperparameter space are promising
- Stacking works best when combining fundamentally different approaches (linear + non-linear),
  not similar ones (XGBoost + Gradient Boosting)
- The "diagnostic then action" workflow (EDA → Features → Modeling) ensures every transformation
  is justified by data, not arbitrary choices
