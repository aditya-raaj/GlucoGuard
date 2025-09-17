# 🩺 Diabetes Classification with Machine Learning

## 📌 Overview
This repository contains an end-to-end machine learning pipeline to classify diabetes-related outcomes using routine clinical features such as **Age, Urea, HbA1c, Cholesterol, Lipids, BMI, Kidney markers**, etc.  

The dataset was **imbalanced**, with most patients falling into one class, so **balancing + recall optimization** was critical. The final model leverages **SMOTE + XGBoost** with hyperparameter tuning to maximize **recall on the minority class** (diabetes-positive), while keeping strong overall performance.  

---

## 📊 Dataset
- **Source**: Clinical dataset (processed and saved as `polished_data.pkl`)  
- **Features**:
  - Demographics: `Gender`, `Age`
  - Clinical: `Urea`, `Cr`, `HbA1c`, `Chol`, `TG`, `HDL`, `LDL`, `VLDL`, `BMI`
- **Target**: `Class` (3 categories: encoded as 0, 1, 2)  
- **Imbalance observed**:
  - Class 2 → 671 samples
  - Class 0 → 82 samples
  - Class 1 → 47 samples
This imbalance made naive models biased toward the majority class.

---

## ⚙️ Preprocessing
1. **Cleaning & Standardization**:
 - Normalized categorical entries (`Gender`, `Class`)
 - Removed duplicates, dropped irrelevant IDs
2. **Encoding**:
 - `LabelEncoder` for categorical variables
3. **Scaling**:
 - `StandardScaler` applied to all numeric features  
 - Saved as `scaler01.pkl`
4. **Balancing**:
 - Applied **SMOTE** to oversample minority classes during training

---

## 🔬 Baseline Model Experiments

We tested several classifiers **without SMOTE** initially.  

| Model                | Accuracy | Macro-F1 | Recall (Minority Class 1) |
|-----------------------|----------|----------|---------------------------|
| Logistic Regression   | ~0.78    | ~0.62    | Very low                  |
| Random Forest         | ~0.91    | ~0.85    | Poor recall on class 1    |
| SVM (RBF Kernel)      | ~0.89    | ~0.80    | Misclassified minority    |
| XGBoost (default)     | ~0.93    | ~0.88    | Some improvement, but still biased |

👉 **Observation**: Most models achieved **high accuracy** due to dominance of Class 2, but **failed recall on minority class (1)**.  

This motivated use of **SMOTE + model tuning**.

---

## ⚡ XGBoost with SMOTE (Final Model)

We built an **imbalanced-learn pipeline**:

(SMOTE → StandardScaler → XGBoost)


---


### 🎯 Hyperparameter Tuning
- Performed **RandomizedSearchCV** over 25 configs  
- Custom scorer = **Recall on Class 1**  
- Best Params:
  ```python
  {
    'clf__subsample': 1.0,
    'clf__n_estimators': 300,
    'clf__min_child_weight': 5,
    'clf__max_depth': 4,
    'clf__learning_rate': 0.03,
    'clf__gamma': 5,
    'clf__colsample_bytree': 0.8
  }


---

## ✅ Results (Test Set)

- **Accuracy**: 0.965
- **Macro F1**: 0.945
- **Minority Class Recall (Class 1)**: 1.0 🎉

### Confusion Matrix:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.75      | 1.00   | 0.86     | 21      |
| 1     | 1.00      | 1.00   | 1.00     | 6       |
| 2     | 1.00      | 0.96   | 0.98     | 173     |



---

👉 Why we selected XGBoost + SMOTE?

Outperformed baselines in both accuracy and macro-F1

Achieved perfect recall on minority class, critical in healthcare applications (false negatives = dangerous)

Balanced tradeoff: high specificity + high sensitivity


---

