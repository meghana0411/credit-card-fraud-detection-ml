# Credit Card Fraud Detection using Machine Learning

This project implements a **Credit Card Fraud Detection system** using Machine Learning techniques. The model predicts whether a transaction is **Fraudulent** or **Normal** based on transaction-related features.

---

## Project Overview

Credit card fraud is a major challenge in financial systems due to the highly imbalanced nature of transaction data. This project applies **Logistic Regression** with appropriate data preprocessing techniques to effectively identify fraudulent transactions.

---

## Machine Learning Model

- Algorithm: Logistic Regression  
- Learning Type: Supervised Learning  
- Problem Type: Binary Classification  

### Classes
- 0 → Normal Transaction  
- 1 → Fraudulent Transaction  

---

## Dataset

The project uses a credit card transaction dataset containing both legitimate and fraudulent transactions.

- Target column: `is_fraud`
- Dataset format: CSV

### Features Include
- Transaction amount  
- Transaction hour  
- Merchant category  
- Foreign transaction indicator  
- Location mismatch  
- Device trust score  
- Transaction velocity  
- Cardholder age  

---

## Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

## Methodology

- Data loading and inspection  
- Handling missing values  
- Encoding categorical features using One-Hot Encoding  
- Feature scaling using StandardScaler  
- Train-test data split  
- Model training with class imbalance handling  
- Performance evaluation using classification metrics  
- Visualization using confusion matrix  

---

## Results

The Logistic Regression model demonstrates effective fraud detection performance. Due to the imbalanced dataset, **precision, recall, and F1-score** provide more meaningful insights than accuracy alone.

---

---

## Author

Sai Meghana Vadluri  
Computer Science Student  
Machine Learning Project
