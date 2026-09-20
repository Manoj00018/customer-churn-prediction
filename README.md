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

About **26.5% of customers churn** and 73.5% stay — an *imbalanced* dataset, which shapes how we measure and train the models.

---

## 🗺️ Notebook Workflow

The notebook walks through a complete, beginner-friendly ML pipeline:

**1. Load & inspect** — read the data, check shapes, types, and the churn split.

**2. Data cleaning**
- `TotalCharges` was stored as text because of 11 blank entries (all customers with `tenure = 0`, i.e. brand new). Converted to numeric and filled those blanks with 0.
- Dropped `customerID` — a unique identifier carries no predictive signal and can cause a model to "memorise" rows.
- Encoded the target: `Churn` → 1 / 0.

**3. Exploratory Data Analysis (EDA)**
- **Tenure:** churn is highest in the first year (**~47%** in the first 12 months) and falls steadily to **~7%** after five years — the first year is the danger zone.
- **Contract:** month-to-month churns at **~43%** vs **~3%** for two-year contracts.
- **Internet:** fiber-optic customers churn far more (**~42%**) than DSL (**~19%**).
- **Interactions:** fiber **and** month-to-month together hit **~55%** churn — the riskiest segment. Adding tech support roughly halves fiber churn (**~49% → ~23%**).
- **Overlap check:** risky factors overlap (e.g. ~48% of month-to-month customers use electronic check), a reminder that *EDA shows association, not causation.*
- **Linear-model checks:** a correlation heatmap flags multicollinearity, and a log-odds plot confirms the numeric features are roughly linear in the log-odds of churn.

**4. Preprocessing**
- Collapsed redundant `"No internet service"` / `"No phone service"` categories into `"No"`.
- Dropped `TotalCharges` — it's essentially `tenure × MonthlyCharges`, so it's redundant (multicollinearity) and hurts coefficient interpretability.
- **Binary encoding** for Yes/No columns; **one-hot encoding** (with `drop_first`) for multi-category columns.
- **Train/test split** (80/20) and **standard scaling** (z-scores), fit on the training data only to avoid leakage.

**5. Handling class imbalance** — `class_weight="balanced"` (and `scale_pos_weight` for XGBoost) so the models pay extra attention to the rare churn class, boosting **recall**.

**6. Modeling** — three models, scored the same way at the default 0.50 cutoff:

| Model | Recall | Precision | ROC-AUC |
|-------|:------:|:---------:|:-------:|
| **Logistic Regression** | **0.820** | 0.523 | **0.860** |
| XGBoost | 0.796 | 0.534 | 0.855 |
| Random Forest | 0.458 | 0.643 | 0.839 |

Because a missed churner is a lost customer, we prioritise **recall**. Note how Random Forest's decent accuracy hides poor recall — accuracy alone is misleading on imbalanced data.

**7. Tuning**
- **Hyperparameters:** `GridSearchCV` with 5-fold cross-validation on ROC-AUC. Logistic Regression (best `C = 2`, CV ROC-AUC ≈ 0.839) and XGBoost (best `max_depth = 3`, `learning_rate = 0.03`, CV ROC-AUC ≈ 0.841) tied within a hair, so we keep the simpler, more interpretable model.
- **Threshold:** lowering the decision cutoff from 0.50 catches more churners. At **0.40** the model catches **327 of 373** churners (recall ≈ 0.88) while missing only 46 — a deliberate recall-vs-precision trade-off.

| Threshold | Recall | Precision | Caught | Missed |
|:---------:|:------:|:---------:|:------:|:------:|
| 0.50 | 0.820 | 0.523 | 306 | 67 |
| **0.40** | **0.877** | 0.466 | **327** | **46** |
| 0.30 | 0.941 | 0.428 | 351 | 22 |

**8. Save & reuse** — the final Logistic Regression model, the scaler, and the chosen threshold are saved together with `joblib`, then reloaded to score sample customers.

**9. Interpretability** — Logistic Regression coefficients are exponentiated into **odds ratios**, translating the model into plain business language (>1 raises churn odds, <1 lowers them). The top drivers line up with the EDA story.

---

## 🔑 Key Findings

From the odds ratios (>1 raises churn odds, <1 lowers them):

**Raises churn:** fiber-optic internet (**2.25×**) · streaming services · multiple lines · electronic-check payment · paperless billing.

**Lowers churn:** high monthly charges within a contract (**0.40×**) · longer tenure (**0.47×**) · two-year and one-year contracts (**0.53× / 0.75×**) · online security and tech support.

**Business takeaways:**
- Focus retention on customers in their first year.
- Nudge month-to-month customers toward longer contracts.
- Fiber customers without tech support churn heavily — bundling support could help.

---

## 🏆 Final Model

- **Model:** Logistic Regression (`class_weight="balanced"`), chosen for tying on performance while staying fully explainable.
- **Decision threshold:** 0.40
- **Catches ~88% of churners** on the test set (327 of 373), trading some precision for that recall.

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

3. **Get the dataset** from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and update the file path in the load cell to point at your local copy:
   ```python
   df = pd.read_csv("customer_churn.csv")
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
