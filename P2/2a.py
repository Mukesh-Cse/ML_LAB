import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("2a1000.csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
n_attrs = X.shape[1]

# -----------------------------
# INITIALIZE S and G
# -----------------------------
S = ['∅'] * n_attrs
G = [['?' for _ in range(n_attrs)]]

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

    # Positive example
    if label == "Yes":
        G = [g for g in G if is_consistent(g, x)]

        for i in range(n_attrs):
            if S[i] == '∅':
                S[i] = x[i]
            elif S[i] != x[i]:
                S[i] = '?'

    # Negative example
    else:
        new_G = []
        for g in G:
            if is_consistent(g, x):
                for i in range(n_attrs):
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
# PREDICTION USING VERSION SPACE
# -----------------------------
def predict(x, S, G):
    if not is_consistent(S, x):
        return "No"
    for g in G:
        if is_consistent(g, x):
            return "Yes"
    return "No"

y_pred = [predict(x, S, G) for x in X]

# -----------------------------
# EVALUATION METRICS
# -----------------------------
print("\nEvaluation Metrics:")
print("Accuracy :", accuracy_score(y, y_pred))
print("Precision:", precision_score(y, y_pred, pos_label="Yes"))
print("Recall   :", recall_score(y, y_pred, pos_label="Yes"))
print("F1 Score :", f1_score(y, y_pred, pos_label="Yes"))
