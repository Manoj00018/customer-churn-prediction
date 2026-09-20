# Customer Churn Prediction 📉

Predicting which telecom customers are about to cancel their service ("churn"), so the business can step in **before** they leave. Keeping an existing customer is far cheaper than winning a new one, so even a rough early-warning model is genuinely valuable.

> 📺 **Video walkthrough:** _add your YouTube link here_

---

## 📊 The Dataset

The [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — **7,043 customers** described by **21 columns**:

- **Account info:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
- **Services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Target:** `Churn` (Yes / No)

About **27% of customers churn** and 73% stay — an *imbalanced* dataset, which shapes how we measure and train the models.

---

## 🗺️ Notebook Workflow

The notebook walks through a complete, beginner-friendly ML pipeline:

**1. Load & inspect** — read the data, check shapes, types, and the churn split.

**2. Data cleaning**
- `TotalCharges` was stored as text because of 11 blank entries (all customers with `tenure = 0`, i.e. brand new). Converted to numeric and filled those blanks with 0.
- Dropped `customerID` — a unique identifier carries no predictive signal and can cause a model to "memorise" rows.
- Encoded the target: `Churn` → 1 / 0.

**3. Exploratory Data Analysis (EDA)**
- **Tenure:** churn is highest in the first year (~47% in the first 12 months) and falls steadily after — the first year is the danger zone.
- **Contract:** month-to-month churns at ~43% vs ~3% for two-year contracts.
- **Internet:** fiber-optic customers churn far more (~42%) than DSL (~19%).
- **Interactions:** fiber **and** month-to-month together hit ~55% churn — the riskiest segment. Tech support roughly halves fiber churn.
- **Overlap check:** risky factors overlap (e.g. month-to-month customers lean toward electronic-check payment), a reminder that *EDA shows association, not causation.*
- **Linear-model checks:** a correlation heatmap flags multicollinearity, and a log-odds plot confirms the numeric features are roughly linear in the log-odds of churn.

**4. Preprocessing**
- Collapsed redundant `"No internet service"` / `"No phone service"` categories into `"No"`.
- Dropped `TotalCharges` — it's essentially `tenure × MonthlyCharges`, so it's redundant (multicollinearity) and hurts coefficient interpretability.
- **Binary encoding** for Yes/No columns; **one-hot encoding** (with `drop_first`) for multi-category columns.
- **Train/test split** (80/20) and **standard scaling** (z-scores), fit on the training data only to avoid leakage.

**5. Handling class imbalance** — `class_weight="balanced"` (and `scale_pos_weight` for XGBoost) so the models pay extra attention to the rare churn class, boosting **recall**.

**6. Modeling** — three models, scored the same way:

| Model | Recall | Precision | ROC-AUC |
|-------|:------:|:---------:|:-------:|
| **Logistic Regression** | **0.820** | 0.523 | **0.860** |
| XGBoost | 0.796 | 0.534 | 0.855 |
| Random Forest | 0.458 | 0.643 | 0.839 |

Because a missed churner is a lost customer, we prioritise **recall**. Note how Random Forest's decent accuracy hides poor recall — accuracy alone is misleading on imbalanced data.

**7. Tuning**
- **Hyperparameters:** `GridSearchCV` with 5-fold cross-validation on ROC-AUC. Logistic Regression and XGBoost tied within a hair, so we keep the simpler, more interpretable model.
- **Threshold:** lowering the decision cutoff from 0.50 toward ~0.35–0.40 catches noticeably more churners (higher recall) at the cost of more false alarms — a business trade-off.

**8. Save & reuse** — the final Logistic Regression model, the scaler, and the chosen threshold are saved together with `joblib`, then reloaded to score sample customers.

**9. Interpretability** — Logistic Regression coefficients are exponentiated into **odds ratios**, translating the model into plain business language (>1 raises churn odds, <1 lowers them). The top drivers line up with the EDA story.

---

## 🔑 Key Findings

**Raises churn:** month-to-month contracts · fiber-optic internet · electronic-check payment · high monthly charges · short tenure (new customers).

**Lowers churn:** one- and two-year contracts · longer tenure · add-ons like tech support and online security.

**Business takeaways:**
- Focus retention on customers in their first year.
- Nudge month-to-month customers toward longer contracts.
- Fiber customers without tech support churn heavily — bundling support could help.

---

## 🏆 Final Model

- **Model:** Logistic Regression (`class_weight="balanced"`), chosen for tying on performance while staying fully explainable.
- **Decision threshold:** 0.40
- **Catches ~82% of churners** on the test set.

Saved artifacts: `churn_model.pkl`, `scaler.pkl`, `threshold.pkl`.

---

## 🚀 How to Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/<your-username>/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Install the requirements**
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn xgboost joblib
   ```

3. **Get the dataset** from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and update the file path in the load cell:
   ```python
   df = pd.read_csv("customer_churn.csv")   # point this at your local copy
   ```

4. **Open the notebook** (Jupyter, VS Code, or Google Colab) and run the cells top to bottom.

---

## 🧰 Built With

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `seaborn` / `matplotlib` · `joblib`

---

## 📁 Repo Contents

```
customer_churn_prediction.ipynb   # the full annotated notebook
churn_model.pkl                   # saved Logistic Regression model
scaler.pkl                        # saved StandardScaler
threshold.pkl                     # saved decision threshold (0.40)
README.md
```
