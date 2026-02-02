import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("2dBreastCancer100.data", header=None)

# Drop ID column
data = data.drop(columns=[0])

# Rename target column
data = data.rename(columns={1: "Diagnosis"})

# Encode target
data["Diagnosis"] = data["Diagnosis"].map({"M": "Yes", "B": "No"})

# -----------------------------
# CLEAN & PREPARE FEATURES
# -----------------------------
features = data.columns[1:]

# Convert all features to numeric
for col in features:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna()

# Discretize numeric features
for col in features:
    data[col] = pd.qcut(
        data[col],
        q=3,
        labels=["Low", "Medium", "High"],
        duplicates="drop"
    )

X = data.iloc[:, 1:].values
y = data["Diagnosis"].values
n = X.shape[1]

# -----------------------------
# INITIALIZE S and G
# -----------------------------
S = ['∅'] * n
G = [['?' for _ in range(n)]]

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def is_consistent(h, x):
    return all(h[i] == '?' or h[i] == x[i] for i in range(len(h)))

def more_general(h1, h2):
    return all(h1[i] == '?' or h1[i] == h2[i] for i in range(len(h1)))

# -----------------------------
# CANDIDATE ELIMINATION
# -----------------------------
for x, label in zip(X, y):

    if label == "Yes":   # Malignant
        G = [g for g in G if is_consistent(g, x)]
        for i in range(n):
            if S[i] == '∅':
                S[i] = x[i]
            elif S[i] != x[i]:
                S[i] = '?'

    else:                # Benign
        new_G = []
        for g in G:
            if is_consistent(g, x):
                for i in range(n):
                    if g[i] == '?' and S[i] != x[i]:
                        h = g.copy()
                        h[i] = S[i]
                        new_G.append(h)
            else:
                new_G.append(g)
        G = new_G

    G = [g for g in G if more_general(g, S)]

# -----------------------------
# OUTPUT VERSION SPACE
# -----------------------------
print("Final Specific Hypothesis (S):")
print(S)

print("\nFinal General Hypotheses (G):")
for g in G:
    print(g)

# -----------------------------
# PREDICTION (STABLE)
# -----------------------------
def predict(x, S, G):
    if S.count('?') == len(S):
        return "Yes" if any(is_consistent(g, x) for g in G) else "No"
    return "Yes" if is_consistent(S, x) else "No"

y_pred = [predict(x, S, G) for x in X]

# -----------------------------
# EVALUATION METRICS
# -----------------------------
print("\nEvaluation Metrics:")
print("Accuracy :", accuracy_score(y, y_pred))
print("Precision:", precision_score(y, y_pred, pos_label="Yes", zero_division=0))
print("Recall   :", recall_score(y, y_pred, pos_label="Yes", zero_division=0))
print("F1 Score :", f1_score(y, y_pred, pos_label="Yes", zero_division=0))
