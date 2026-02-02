# ID3 Decision Tree – Efficient & Concise
# Dataset: 3cZoo.data

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
    weighted = sum(
        (len(sub) / len(data)) * entropy(sub[target])
        for _, sub in data.groupby(feature)
    )
    return total_entropy - weighted


# -----------------------------
# ID3 ALGORITHM
# -----------------------------
def id3(data, features, target):
    if data[target].nunique() == 1:
        return data[target].iloc[0]

    if not features:
        return data[target].mode()[0]

    best = max(features, key=lambda f: information_gain(data, f, target))
    tree = {best: {}}

    for val, subset in data.groupby(best):
        remaining = [f for f in features if f != best]
        tree[best][val] = id3(subset, remaining, target)

    return tree


# -----------------------------
# CLASSIFICATION
# -----------------------------
def classify(sample, tree, default):
    while isinstance(tree, dict):
        feature = next(iter(tree))
        tree = tree.get(feature, {}).get(sample.get(feature), default)
    return tree


# -----------------------------
# LOAD & PREPARE DATA
# -----------------------------
cols = [
    "animal_name","hair","feathers","eggs","milk","airborne","aquatic",
    "predator","toothed","backbone","breathes","venomous","fins",
    "legs","tail","domestic","catsize","type"
]

data = pd.read_csv("3cZoo1000.data", header=None, names=cols)

# Drop ID-like column
data.drop(columns=["animal_name"], inplace=True)

target_col = "type"

# Discretize numeric attribute
data["legs"] = data["legs"].astype(str)

features = [c for c in data.columns if c != target_col]

# -----------------------------
# TRAIN
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
new_sample = X[0]  # example sample
print("\nNew Sample Prediction:", classify(new_sample, tree, default_class))
