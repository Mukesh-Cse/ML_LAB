import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# FIND-S TRAINING FUNCTION
# -----------------------------
def find_s(training_data, target_col):
    attributes = training_data.columns[:-1]
    S = ['∅'] * len(attributes)

    for _, row in training_data.iterrows():
        if row[target_col] == "Pass":   # only positive examples
            for i, attr in enumerate(attributes):
                if S[i] == '∅':
                    S[i] = row[attr]
                elif S[i] != row[attr]:
                    S[i] = '?'
    return S


# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict(example, hypothesis):
    for i in range(len(hypothesis)):
        if hypothesis[i] != '?' and hypothesis[i] != example[i]:
            return "No"
    return "Yes"


# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("S1000.csv")   # change filename if needed
target_col = data.columns[-1]

# -----------------------------
# TRAIN FIND-S
# -----------------------------
hypothesis = find_s(data, target_col)
print("Learned Hypothesis (FIND-S):", hypothesis)

# -----------------------------
# MAKE PREDICTIONS
# -----------------------------
y_true = data[target_col].tolist()
y_pred = []

X = data.iloc[:, :-1].values
for row in X:
    y_pred.append(predict(row, hypothesis))

# -----------------------------
# EVALUATION METRICS
# -----------------------------
accuracy  = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, pos_label="Yes") # For d3 replace all yes =1
recall    = recall_score(y_true, y_pred, pos_label="Yes")
f1        = f1_score(y_true, y_pred, pos_label="Yes")

print("\nEvaluation Metrics:")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
