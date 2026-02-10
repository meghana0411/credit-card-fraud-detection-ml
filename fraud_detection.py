import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ---------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------
df = pd.read_csv("creditcard.csv")

print("First 5 rows of dataset:")
print(df.head())

# ---------------------------------------------------
# 2. Check missing values
# ---------------------------------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# ---------------------------------------------------
# 3. Class distribution
# ---------------------------------------------------
print("\nClass Distribution (0 = Normal, 1 = Fraud):")
print(df["is_fraud"].value_counts())

# ---------------------------------------------------
# 4. Separate features and target
# ---------------------------------------------------
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

# ---------------------------------------------------
# 5. Identify categorical and numerical columns
# ---------------------------------------------------
categorical_cols = [
    "merchant_category",
    "foreign_transaction",
    "location_mismatch"
]

numerical_cols = [
    "amount",
    "transaction_hour",
    "device_trust_score",
    "velocity_last_24h",
    "cardholder_age"
]

# ---------------------------------------------------
# 6. One-hot encode categorical columns
# ---------------------------------------------------
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# ---------------------------------------------------
# 7. Scale numerical columns
# ---------------------------------------------------
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# ---------------------------------------------------
# 8. Train-test split
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------
# 9. Train Logistic Regression model
#    (class_weight used for imbalance)
# ---------------------------------------------------
model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train, y_train)

# ---------------------------------------------------
# 10. Predictions
# ---------------------------------------------------
y_pred = model.predict(X_test)

# ---------------------------------------------------
# 11. Evaluation metrics
# ---------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Accuracy : {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall   : {recall:.2f}")
print(f"F1 Score : {f1:.2f}")

# ---------------------------------------------------
# 12. Confusion Matrix
# ---------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Fraud"],
    yticklabels=["Normal", "Fraud"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
