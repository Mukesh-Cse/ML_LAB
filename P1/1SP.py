import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ===============================
# 1. LOAD DATASET
# ===============================
df = pd.read_csv("SP500.csv")

# ===============================
# 2. DISCRETIZATION FUNCTION
# ===============================
def marks_to_category(marks):
    if marks >= 70:
        return "High"
    elif marks >= 40:
        return "Medium"
    else:
        return "Low"

# ===============================
# 3. CONVERT NUMERIC ATTRIBUTES
# ===============================
numeric_columns = [
    "Quiz", "Midterm", "Assignment_1", "Assignment_2",
    "Presentation", "Final_Exam_Marks"
]

for col in numeric_columns:
    df[col] = df[col].apply(marks_to_category)

# ===============================
# 4. CONVERT MULTICLASS → BINARY
# ===============================
# Concept: High Performer
df["Class"] = df["Categories"].apply(
    lambda x: "Yes" if x == "Distinction" else "No"
)

# Keep only required columns
df = df[numeric_columns + ["Class"]]

# ===============================
# 5. FIND-S ALGORITHM
# ===============================
def find_s(training_data):
    hypothesis = None

    # Initialize with first positive example
    for row in training_data:
        if row[-1] == "Yes":
            hypothesis = row[:-1]
            break

    if hypothesis is None:
        return []

    # Generalize hypothesis
    for row in training_data:
        if row[-1] == "Yes":
            for i in range(len(hypothesis)):
                if hypothesis[i] != row[i]:
                    hypothesis[i] = "?"

    return hypothesis

# ===============================
# 6. PREDICTION FUNCTION
# ===============================
def predict(hypothesis, instance):
    if not hypothesis:
        return "No"
    for h, x in zip(hypothesis, instance):
        if h != "?" and h != x:
            return "No"
    return "Yes"

# ===============================
# 7. TRAIN FIND-S
# ===============================
training_data = df.values.tolist()
hypothesis = find_s(training_data)

print("\nLearned Hypothesis (FIND-S):")
print(hypothesis)

# ===============================
# 8. PREDICTIONS
# ===============================
X = df.iloc[:, :-1].values.tolist()
y_true = df["Class"].tolist()

y_pred = [predict(hypothesis, x) for x in X]

# ===============================
# 9. EVALUATION METRICS (FIXED)
# ===============================
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, pos_label="Yes")
recall = recall_score(y_true, y_pred, pos_label="Yes")
f1 = f1_score(y_true, y_pred, pos_label="Yes")

print("\nEvaluation Metrics:")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)
