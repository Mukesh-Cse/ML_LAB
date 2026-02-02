import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# FIND-S TRAINING
# -----------------------------
def find_s(data, target_col):
    attrs = data.columns[:-1]
    S = ['∅'] * len(attrs)

    for _, row in data.iterrows():
        if row[target_col] == 1:          # positive example
            for i, attr in enumerate(attrs):
                if S[i] == '∅':
                    S[i] = row[attr]
                elif S[i] != row[attr]:
                    S[i] = '?'
    return S


# -----------------------------
# PREDICTION
# -----------------------------
def predict(x, S):
    for i in range(len(S)):
        if S[i] != '?' and S[i] != x[i]:
            return 0
    return 1


# -----------------------------
# LOAD & PREPROCESS DATA
# ----------------------------
data = pd.read_csv("abalone1000.csv")
target_col = data.columns[-1]

# Convert multiclass target to binary (required for FIND-S)
data[target_col] = data[target_col].apply(lambda x: 1 if x >= 10 else 0)

# -----------------------------
# TRAIN
# -----------------------------
hypothesis = find_s(data, target_col)
print("Learned Hypothesis (FIND-S):", hypothesis)

# -----------------------------
# PREDICT
# -----------------------------
X = data.iloc[:, :-1].values
y_true = data[target_col].values
y_pred = [predict(x, hypothesis) for x in X]

# -----------------------------
# EVALUATION
# -----------------------------
print("\nEvaluation Metrics:")
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, pos_label=1))
print("Recall   :", recall_score(y_true, y_pred, pos_label=1))
print("F1 Score :", f1_score(y_true, y_pred, pos_label=1))
