# ID3 Decision Tree – Efficient, Concise, Optimized
# Dataset: 3eMushroom.data

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
    base = entropy(data[target])
    weighted = sum(
        (len(sub) / len(data)) * entropy(sub[target])
        for _, sub in data.groupby(feature)
    )
    return base - weighted


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

    for val, sub in data.groupby(best):
        rem = [f for f in features if f != best]
        tree[best][val] = id3(sub, rem, target)

    return tree


# -----------------------------
# CLASSIFY
# -----------------------------
def classify(sample, tree, default):
    while isinstance(tree, dict):
        feat = next(iter(tree))
        tree = tree.get(feat, {}).get(sample.get(feat), default)
    return tree


# -----------------------------
# LOAD & PREPARE DATA
# -----------------------------
# Mushroom dataset columns (UCI)
cols = [
    "class","cap-shape","cap-surface","cap-color","bruises","odor",
    "gill-attachment","gill-spacing","gill-size","gill-color",
    "stalk-shape","stalk-root","stalk-surface-above-ring","stalk-surface-below-ring",
    "stalk-color-above-ring","stalk-color-below-ring","veil-type","veil-color",
    "ring-number","ring-type","spore-print-color","population","habitat"
]

data = pd.read_csv("3eMushroom1000.data", header=None, names=cols)

target_col = "class"   # e = edible, p = poisonous
features = [c for c in data.columns if c != target_col]

# Handle missing values (e.g., stalk-root = '?')
data.fillna("unknown", inplace=True)
data.replace("?", "unknown", inplace=True)

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
new_sample = X[0]  # example mushroom
print("\nNew Sample Prediction:", classify(new_sample, tree, default_class))
