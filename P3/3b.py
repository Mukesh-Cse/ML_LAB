import pandas as pd
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# ENTROPY
# -----------------------------
def entropy(data, target):
    counts = data[target].value_counts()
    ent = 0
    for c in counts:
        p = c / len(data)
        ent -= p * math.log2(p)
    return ent


# -----------------------------
# INFORMATION GAIN
# -----------------------------
def information_gain(data, feature, target):
    total_entropy = entropy(data, target)
    values = data[feature].unique()

    weighted_entropy = 0
    for v in values:
        subset = data[data[feature] == v]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset, target)

    return total_entropy - weighted_entropy


# -----------------------------
# ID3
# -----------------------------
def id3(data, features, target):
    if len(data[target].unique()) == 1:
        return data[target].iloc[0]

    if not features:
        return data[target].mode()[0]

    gains = {f: information_gain(data, f, target) for f in features}
    best_feature = max(gains, key=gains.get)

    tree = {best_feature: {}}

    for val in data[best_feature].unique():
        subset = data[data[best_feature] == val]
        remaining = [f for f in features if f != best_feature]
        tree[best_feature][val] = id3(subset, remaining, target)

    return tree


# -----------------------------
# CLASSIFY
# -----------------------------
def classify(sample, tree, default):
    if not isinstance(tree, dict):
        return tree

    feature = next(iter(tree))
    value = sample.get(feature)

    if value in tree[feature]:
        return classify(sample, tree[feature][value], default)
    else:
        return default


# -----------------------------
# LOAD & CLEAN DATA
# -----------------------------
data = pd.read_csv("3bHR1000.csv")

# DROP ID COLUMN (CRITICAL)
data.drop(columns=["EmployeeNumber"], inplace=True)

# TARGET
target_col = "Attrition"

# Fill missing values
data.fillna(data.mode().iloc[0], inplace=True)

# DISCRETIZE NUMERIC FEATURES
for col in data.columns:
    if col != target_col and data[col].dtype != object:
        data[col] = pd.qcut(data[col], 3, duplicates="drop").astype(str)

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
print("Precision:", precision_score(y_true, y_pred, pos_label="Yes", zero_division=0))
print("Recall   :", recall_score(y_true, y_pred, pos_label="Yes", zero_division=0))
print("F1 Score :", f1_score(y_true, y_pred, pos_label="Yes", zero_division=0))

# -----------------------------
# NEW SAMPLE (NO TARGET!)
# -----------------------------
new_sample = {k: v for k, v in X[0].items()}
print("\n New Sample :", new_sample)
print("\nPredicted Class for new sample:",
      classify(new_sample, tree, default_class))
