
[credit_Card_fraud_README.md](https://github.com/user-attachments/files/21589844/credit_Card_fraud_README.md)
# Credit Card Fraud Detection – Imbalanced Data Modeling

This project focuses on detecting fraudulent credit card transactions using the well-known **Credit Card Fraud Detection** dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The dataset is highly imbalanced, making traditional modeling methods less effective.

---

## Problem Statement

The objective is to accurately classify transactions as **fraudulent** or **non-fraudulent**, minimizing false negatives (missed frauds) without oversampling or undersampling the data.

---

## Dataset

- Total samples: `284,807`
- Fraudulent transactions (class 1): ~`492`
- Non-fraudulent transactions (class 0): ~`284,315`
- Features: `V1` to `V28` (PCA-transformed), `Time`, and `Amount`

---

## Approach

Due to the extreme class imbalance, I opted for a method that **preserves the original dataset** instead of using:
-  Random over-sampling
-  Random under-sampling
-  SMOTE(Synthetic Minority Over-sampling Technique)

---

## Feature Engineering

- Removed features with **very low correlation** with the target label.
- Used `.corrwith()` to find correlation between features and target.
- Dropped low-contributing features to reduce noise and speed up training.
- Standardized `Time` and `Amount` using `StandardScaler`.

---

## Models Used

1. **XGBoost Classifier**  
   - Used with `scale_pos_weight` to address class imbalance.
   - Showed strong standalone performance.

2. **Ensemble (Voting Classifier)**  
   - Combined:
     - Logistic Regression (L1 regularization for feature selection)
     - Random Forest (balanced class weights)
     - XGBoost (for boosted performance)
   - Used **soft voting** to aggregate class probabilities.
   - Performed similarly to XGBoost but adds robustness.

3. **Support Vector Machine (SVM)** was tested but excluded from the final ensemble due to high training time.

---

## Model Tuning

- Attempted **GridSearchCV** for hyperparameter tuning.
- However, due to dataset size and multiple classifiers, it was **too time-consuming** on Colab resources.
- Opted for **manual tuning** and used `class_weight='balanced'` where applicable.

---

## Results

After training the XGBoost and ensemble models on the reduced feature set, I obtained the following metrics on the test set:

```
           precision    recall  f1-score   support

     0       1.00       1.00      1.00     56864
     1       0.89       0.83      0.85        98

accuracy                           1.00     56962
macro avg      0.94       0.90      0.92     56962
weighted avg   1.00       1.00      1.00     56962
```

- **Recall for fraud class (1)** is ~**0.80**, meaning 80% of frauds were correctly detected.
- Performance was similar between **XGBoost alone** and **VotingClassifier**.
- Ensemble adds reliability through model diversity.

---

## Conclusions

- Removing low-correlation features helped reduce noise and training time.
- XGBoost is a strong standalone model for this dataset.
- Ensemble VotingClassifier adds robustness with similar performance.
- Avoiding sampling let us preserve true data distribution and reduced risk of overfitting.

---

## Future Improvements

- Use `SelectFromModel` or SHAP for deeper feature importance analysis.
- Deploy optimized model with fast inference.
- Try **StackingClassifier** or **LightGBM** for improved performance.
- Run GridSearchCV using parallelization on better compute (like Kaggle Notebooks or AWS/GCP).

---

## Files

- `credit_card_fraud_detection.ipynb` – Full code
- `README.md` – Project overview
