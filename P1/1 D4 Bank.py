import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# FIND-S
# -----------------------------
def find_s(data, target):
    attrs = data.columns[:-1]
    S = ['∅'] * len(attrs)

    for _, row in data.iterrows():
        if row[target] == 1:
            for i, attr in enumerate(attrs):
                if S[i] == '∅':
                    S[i] = row[attr]
                elif S[i] != row[attr]:
                    S[i] = '?'
    return S


def predict(x, S):
    for i in range(len(S)):
        if S[i] != '?' and S[i] != x[i]:
            return 0
    return 1


# -----------------------------
# LOAD & CLEAN DATA (FIXED)
# -----------------------------
data = pd.read_excel("BankIP1000.xlsx")

# Rename needed columns
data = data.rename(columns={
    "Unnamed: 6": "ROCE",
    "Unnamed: 7": "ROA",
    "Unnamed: 10": "ROE",
    "Unnamed: 11": "NIM",
    "Unnamed: 17": "Price_to_Book"
})

# Keep only required columns
data = data[["ROCE", "ROA", "ROE", "NIM", "Price_to_Book"]]

# FORCE numeric conversion (THIS FIXES THE ERROR)
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Drop rows with non-numeric values
data = data.dropna()

# -----------------------------
# CREATE BINARY TARGET
# -----------------------------
median_pb = data["Price_to_Book"].median()
data["Target"] = data["Price_to_Book"].apply(lambda x: 1 if x >= median_pb else 0)
data = data.drop(columns=["Price_to_Book"])

# -----------------------------
# DISCRETIZE (required for FIND-S)
# -----------------------------
for col in data.columns[:-1]:
    data[col] = pd.qcut(data[col], q=3, labels=["Low", "Medium", "High"])

# -----------------------------
# TRAIN
# -----------------------------
hypothesis = find_s(data, "Target")
print("Learned Hypothesis (FIND-S):", hypothesis)

# -----------------------------
# PREDICT & EVALUATE
# -----------------------------
X = data.iloc[:, :-1].values
y_true = data["Target"].values
y_pred = [predict(x, hypothesis) for x in X]

print("\nEvaluation Metrics:")
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, pos_label=1))
print("Recall   :", recall_score(y_true, y_pred, pos_label=1))
print("F1 Score :", f1_score(y_true, y_pred, pos_label=1))
