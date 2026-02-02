import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# FIND-S ALGORITHM
# -----------------------------
def find_s(data, target):
    attrs = data.columns[:-1]
    S = ['∅'] * len(attrs)

    for _, row in data.iterrows():
        if row[target] == 1:          # positive example (AMI)
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
# LOAD DATA
# -----------------------------
data = pd.read_excel("HD1000.xlsx")

target_col = "group  (control:0, AMI:1)"

# Move target to last column
features = [c for c in data.columns if c != target_col]
data = data[features + [target_col]]

# -----------------------------
# CLEAN DATA
# -----------------------------
# Force numeric conversion (handles hidden strings safely)
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Remove invalid rows
data = data.dropna()

# -----------------------------
# DISCRETIZE FEATURES (FIND-S needs categorical)
# -----------------------------
for col in features:
    if data[col].nunique() > 3:
        data[col] = pd.qcut(
            data[col],
            q=3,
            labels=["Low", "Medium", "High"],
            duplicates="drop"
        )
    else:
        data[col] = data[col].astype(int).astype(str)  # treat as categorical

# -----------------------------
# TRAIN
# -----------------------------
hypothesis = find_s(data, target_col)
print("Learned Hypothesis (FIND-S):", hypothesis)

# -----------------------------
# PREDICT & EVALUATE
# -----------------------------
X = data.iloc[:, :-1].values
y_true = data[target_col].values
y_pred = [predict(x, hypothesis) for x in X]

print("\nEvaluation Metrics:")
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, pos_label=1))
print("Recall   :", recall_score(y_true, y_pred, pos_label=1))
print("F1 Score :", f1_score(y_true, y_pred, pos_label=1))
