import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("2bspam1000.csv", encoding="latin-1")
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# -----------------------------
# FEATURE EXTRACTION (CATEGORICAL)
# -----------------------------
data['free'] = data['message'].str.contains('free', case=False).map({True: 'Yes', False: 'No'})
data['win']  = data['message'].str.contains('win',  case=False).map({True: 'Yes', False: 'No'})
data['call'] = data['message'].str.contains('call', case=False).map({True: 'Yes', False: 'No'})

data['length'] = pd.cut(
    data['message'].str.len(),
    bins=[0, 50, 100, 500],
    labels=['Short', 'Medium', 'Long']
)

data['label'] = data['label'].map({'spam': 'Yes', 'ham': 'No'})
data = data[['free', 'win', 'call', 'length', 'label']]

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
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

    if label == "Yes":  # spam
        G = [g for g in G if is_consistent(g, x)]
        for i in range(n):
            if S[i] == '∅':
                S[i] = x[i]
            elif S[i] != x[i]:
                S[i] = '?'

    else:  # ham
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
# PREDICTION (FIXED)
# -----------------------------
def predict(x, S, G):
    if S.count('?') == len(S):          # fully general S
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
