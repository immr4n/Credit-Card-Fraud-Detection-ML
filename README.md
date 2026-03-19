# Credit Card Fraud Detection — Machine Learning

> A machine learning project built as part of my MSc AI coursework, tackling one of the most common real-world challenges in data science: highly imbalanced datasets.

---

## What this project does

This notebook builds a fraud detection system on real credit card transaction data. The main challenge here isn't the model itself — it's the data: only 0.17% of transactions are fraudulent. Getting a model to correctly catch fraud without flagging every legitimate transaction as suspicious requires careful handling of class imbalance.

---

## Key techniques

| Technique | Purpose |
|-----------|----------|
| SMOTE (Synthetic Oversampling) | Balance the dataset by generating synthetic fraud samples |
| Random Forest | Main classifier — robust to imbalanced data |
| Precision-Recall curve | Better evaluation metric than accuracy for imbalanced data |
| Confusion Matrix | Understand false positives vs. false negatives |

---

## Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions, 492 fraud cases (0.17%)
- **Features:** 28 PCA-transformed features + Amount + Time
- **Target:** 0 = Legitimate, 1 = Fraud

> Note: Dataset is too large for GitHub. Download `creditcard.csv` from Kaggle and update the file path in line 3 of the notebook.

---

## How to run

```bash
# Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn

# Open the notebook
jupyter notebook MLassignment.ipynb
```

---

## What I learned

- Why accuracy is a misleading metric on imbalanced datasets
- How SMOTE works and when to use it vs. undersampling
- The tradeoff between catching all fraud (recall) and avoiding false alarms (precision)
- How Random Forest handles feature importance in high-dimensional data

---

## Author

**Mohammed Imran Ibrahim**  
MSc Artificial Intelligence — Berlin School of Business and Innovation  
[LinkedIn](https://www.linkedin.com/in/imm4n)
