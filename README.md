# 🩺 Gluco Guard: Diabetes Classification with Machine Learning




## Overview
This repository contains an end-to-end machine learning pipeline to classify diabetes-related outcomes using routine clinical features such as **Age, Urea, HbA1c, Cholesterol, Lipids, BMI, Kidney markers**, etc.  

The dataset was **imbalanced**, with most patients falling into one class, so **balancing + recall optimization** was critical. The final model leverages **SMOTE + XGBoost** with hyperparameter tuning to maximize **recall on the minority class** (diabetes-positive), while keeping strong overall performance.  

---


## Dataset
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

## Preprocessing
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

## Baseline Model Experiments

We tested several classifiers **without SMOTE** initially.  

| Model                | Accuracy | Macro-F1 | Recall (Minority Class 1) |
|-----------------------|----------|----------|---------------------------|
| Logistic Regression   | ~0.78    | ~0.62    | Very low                  |
| Random Forest         | ~0.91    | ~0.85    | Poor recall on class 1    |
| SVM (RBF Kernel)      | ~0.89    | ~0.80    | Misclassified minority    |
| XGBoost (default)     | ~0.93    | ~0.88    | Some improvement, but still biased |

**Observation**: Most models achieved **high accuracy** due to dominance of Class 2, but **failed recall on minority class (1)**.  

This motivated use of **SMOTE + model tuning**.

---

## XGBoost with SMOTE (Final Model)

We built an **imbalanced-learn pipeline**:

(SMOTE → StandardScaler → XGBoost)


---


### Hyperparameter Tuning
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

## Results (Test Set)

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

## Why XGBoost + SMOTE?

### 1. Performance vs. Baseline Models
- Benchmarked Logistic Regression, SVM, Random Forest, and XGBoost.  
- Simpler models reached **80–93% accuracy** but struggled with the **imbalanced dataset**.  
- **XGBoost** captured non-linear feature interactions, achieving **96.5% accuracy** and highest F1-scores.

### 2. Handling Class Imbalance with SMOTE
- Dataset distribution was highly skewed (671 vs 82 vs 47).  
- Without balancing, models ignored minority classes → many **false negatives**.  
- **SMOTE** generated synthetic samples for underrepresented classes, boosting recall for minority class from **<40% → 100%**.

### 3. Recall is Critical in Healthcare
- In medical diagnosis, **recall (sensitivity)** is more important than plain accuracy.  
- A **false negative** (undiagnosed diabetes) can be life-threatening, while a false positive only triggers more testing.  
- Our pipeline ensured **no missed diabetic cases** while keeping high precision.

### 4. Balanced Trade-off
- XGBoost’s **regularization (L1/L2, tree pruning)** prevented overfitting to synthetic samples.  
- Final results:  
  - **Accuracy**: 96.5%  
  - **Precision**: High (few false alarms)  
  - **Recall**: Perfect on minority class (no missed cases)  


**Conclusion:**  
We selected **XGBoost + SMOTE** because it outperformed baselines **and** eliminated the critical issue of false negatives, making it highly reliable for real-world healthcare deployment.



---


## FastAPI Endpoint


![FastAPI Endpoint](https://github.com/aditya-raaj/GlucoGuard/blob/main/templates/fastApi.png)
*(Docs Endpoint preview showcasing different endpoints)*

We exposed the trained model as a REST API using **FastAPI**.  
- A `POST /predict/` endpoint accepts patient features in JSON, applies the same preprocessing (scaling + encoding), and returns the predicted diabetes class.  
- The root endpoint `/` serves a static page or a JSON welcome message.  




![FastAPI Endpoint](https://github.com/aditya-raaj/GlucoGuard/blob/main/templates/fastApi1.png)
*(Docs Endpoint preview showcasing predict endpoint)*

---
