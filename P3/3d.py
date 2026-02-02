# ID3 Decision Tree – Efficient, Concise, Optimized
# Dataset: 3dVoting.data

import pandas as pd
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# ENTROPY & INFORMATION GAIN
# -----------------------------
def entropy(y):
    probs = y.value_counts(normalize=True)
    return -(probs * probs.apply(math.log2)).sum()

def information_gain(data, feature, target):
    total_entropy = entropy(data[target])
    weighted_entropy = sum(
        (len(sub) / len(data)) * entropy(sub[target])
        for _, sub in data.groupby(feature)
    )
    return total_entropy - weighted_entropy


# -----------------------------
# ID3 ALGORITHM
# -----------------------------
def id3(data, features, target):
    if data[target].nunique() == 1:
        return data[target].iloc[0]

    if not features:
        return data[target].mode()[0]

    best_feature = max(features, key=lambda f: information_gain(data, f, target))
    tree = {best_feature: {}}

    for value, subset in data.groupby(best_feature):
        remaining = [f for f in features if f != best_feature]
        tree[best_feature][value] = id3(subset, remaining, target)

    return tree


# -----------------------------
# CLASSIFY FUNCTION
# -----------------------------
def classify(sample, tree, default):
    while isinstance(tree, dict):
        feature = next(iter(tree))
        tree = tree.get(feature, {}).get(sample.get(feature), default)
    return tree


# -----------------------------
# LOAD & PREPROCESS DATA
# -----------------------------
# Voting dataset (Congressional Voting Records)
columns = [
    "party","handicapped-infants","water-project-cost-sharing","adoption-of-the-budget-resolution",
    "physician-fee-freeze","el-salvador-aid","religious-groups-in-schools",
    "anti-satellite-test-ban","aid-to-nicaraguan-contras","mx-missile",
    "immigration","synfuels-corporation-cutback","education-spending",
    "superfund-right-to-sue","crime","duty-free-exports","export-administration-act-south-africa"
]

data = pd.read_csv("3dVoting1000.data", header=None, names=columns)

target_col = "party"

# Replace missing values
data.fillna("unknown", inplace=True)

features = [c for c in data.columns if c != target_col]

# -----------------------------
# TRAIN ID3
# -----------------------------
tree = id3(data, features, target_col)
print("Decision Tree:")
print(tree)

# -----------------------------
# EVALUATION
# -----------------------------
X = data[features].to_dict(orient="records")
y_true = data[target_col].tolist()
default_class = data[target_col].mode()[0]

y_pred = [classify(x, tree, default_class) for x in X]

print("\nEvaluation Metrics:")
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, average="weighted", zero_division=0))
print("Recall   :", recall_score(y_true, y_pred, average="weighted", zero_division=0))
print("F1 Score :", f1_score(y_true, y_pred, average="weighted", zero_division=0))

# -----------------------------
# CLASSIFY NEW SAMPLE
# -----------------------------
new_sample = X[0]   # example voting record
print("\nNew Sample Prediction:", classify(new_sample, tree, default_class))
