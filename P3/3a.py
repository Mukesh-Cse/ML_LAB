import pandas as pd
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# ENTROPY
# -----------------------------
def entropy(data, target):
    values = data[target].value_counts()
    ent = 0
    for v in values:
        p = v / len(data)
        ent -= p * math.log2(p)
    return ent


# -----------------------------
# INFORMATION GAIN
# -----------------------------
def information_gain(data, attribute, target):
    total_entropy = entropy(data, target)
    values = data[attribute].unique()

    weighted_entropy = 0
    for v in values:
        subset = data[data[attribute] == v]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset, target)

    return total_entropy - weighted_entropy


# -----------------------------
# ID3 ALGORITHM
# -----------------------------
def id3(data, features, target):
    if len(data[target].unique()) == 1:
        return data[target].iloc[0]

    if len(features) == 0:
        return data[target].mode()[0]

    gains = {f: information_gain(data, f, target) for f in features}
    best_feature = max(gains, key=gains.get)

    tree = {best_feature: {}}

    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        remaining = [f for f in features if f != best_feature]
        tree[best_feature][value] = id3(subset, remaining, target)

    return tree


# -----------------------------
# CLASSIFY FUNCTION
# -----------------------------
def classify(sample, tree):
    if not isinstance(tree, dict):
        return tree

    attribute = next(iter(tree))
    value = sample.get(attribute)

    if value in tree[attribute]:
        return classify(sample, tree[attribute][value])
    else:
        return "No"


# -----------------------------
# LOAD & PREPROCESS DATA
# -----------------------------
data = pd.read_csv("3aSport1000.csv")

# Drop ID column
data.drop(columns=["Day"], inplace=True)

# Handle missing values
data.fillna(data.mode().iloc[0], inplace=True)

target_col = "Play"
features = list(data.columns[:-1])

# -----------------------------
# TRAIN ID3
# -----------------------------
tree = id3(data, features, target_col)
print("Decision Tree:")
print(tree)

# -----------------------------
# PREDICTION ON TRAINING DATA
# -----------------------------
X = data[features].to_dict(orient="records")
y_true = data[target_col].tolist()
y_pred = [classify(x, tree) for x in X]

# -----------------------------
# EVALUATION METRICS
# -----------------------------
print("\nEvaluation Metrics:")
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, pos_label="Yes"))
print("Recall   :", recall_score(y_true, y_pred, pos_label="Yes"))
print("F1 Score :", f1_score(y_true, y_pred, pos_label="Yes"))

# -----------------------------
# CLASSIFY NEW SAMPLE
# -----------------------------
new_sample = {
    "Outlook": "Sunny",
    "Temperature": "Cool",
    "Humidity": "High",
    "Wind": "Strong"
}

prediction = classify(new_sample, tree)
print("\nNew Sample:", new_sample)
print("Predicted Class:", prediction)
