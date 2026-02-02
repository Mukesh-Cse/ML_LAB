import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("2eSP1000.csv")

# Drop ID column
data.drop(columns=["Student Id"], inplace=True)

# -----------------------------
# CREATE BINARY TARGET
# -----------------------------
data["Result"] = data["Categories"].isin(
    ["Pass", "Distinction"]
).map({True: "Yes", False: "No"})

data.drop(columns=["Categories"], inplace=True)

# -----------------------------
# PREPROCESS FEATURES (FAST + SAFE)
# -----------------------------
features = data.columns[:-1]

# Convert to numeric (vectorized)
data[features] = data[features].apply(pd.to_numeric, errors="coerce")

# Fill missing values instead of dropping (prevents empty data)
data[features] = data[features].fillna(
    data[features].median(numeric_only=True)
)

# Discretize numeric features safely
for col in features:
    data[col] = pd.qcut(
        data[col],
        q=3,
        duplicates="drop"
    ).astype(str)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
n = X.shape[1]

# -----------------------------
# INITIALIZE S & G
# -----------------------------
S = ['∅'] * n
G = [['?' for _ in range(n)]]

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def is_consistent(h, x):
    return all(h[i] == '?' or h[i] == x[i] for i in range(n))

def more_general(h1, h2):
    return all(h1[i] == '?' or h1[i] == h2[i] for i in range(n))

# -----------------------------
# CANDIDATE ELIMINATION
# -----------------------------
for x, label in zip(X, y):

    if label == "Yes":      # Positive example
        G = [g for g in G if is_consistent(g, x)]
        for i in range(n):
            if S[i] == '∅':
                S[i] = x[i]
            elif S[i] != x[i]:
                S[i] = '?'

    else:                   # Negative example
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
# PREDICTION (ROBUST)
# -----------------------------
def predict(x):
    if S.count('?') == n:
        return "Yes" if any(is_consistent(g, x) for g in G) else "No"
    return "Yes" if is_consistent(S, x) else "No"

y_pred = [predict(x) for x in X]

# -----------------------------
# EVALUATION METRICS (SAFE)
# -----------------------------
print("\nEvaluation Metrics:")
print("Accuracy :", accuracy_score(y, y_pred))
print("Precision:", precision_score(y, y_pred, pos_label="Yes", zero_division=0))
print("Recall   :", recall_score(y, y_pred, pos_label="Yes", zero_division=0))
print("F1 Score :", f1_score(y, y_pred, pos_label="Yes", zero_division=0))
