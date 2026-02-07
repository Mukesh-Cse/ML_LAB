import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("C:\\Users\\gmuke\\Desktop\\ML LAB R23\\FINAL MAM\\P7\\7bDiabetes10.csv")

# Detect target column
for col in ["Outcome", "target", "Class", "Label"]:
    if col in data.columns:
        target_col = col
        break

# -----------------------------
# SAFE DISCRETIZATION
# -----------------------------
for col in data.columns:
    if col != target_col and data[col].nunique() > 5:
        try:
            bins = pd.qcut(data[col], q=3, duplicates="drop")
            n_bins = bins.cat.categories.size
            data[col] = pd.qcut(
                data[col],
                q=n_bins,
                labels=list(range(n_bins)),
                duplicates="drop"
            )
        except:
            pass  # keep original if discretization fails

# -----------------------------
# TRAIN-TEST SPLIT
# -----------------------------
train_data, test_data = train_test_split(
    data,
    test_size=0.25,
    random_state=1,
    stratify=data[target_col]
)

# -----------------------------
# BUILD BAYESIAN NETWORK
# -----------------------------
features = [c for c in data.columns if c != target_col]
edges = [(f, target_col) for f in features]

model = DiscreteBayesianNetwork(edges)

# -----------------------------
# TRAIN MODEL
# -----------------------------
model.fit(train_data, estimator=MaximumLikelihoodEstimator)

# -----------------------------
# INFERENCE
# -----------------------------
infer = VariableElimination(model)

y_true = []
y_pred = []

for _, row in test_data.iterrows():
    evidence = row.drop(target_col).to_dict()
    result = infer.map_query(variables=[target_col], evidence=evidence)

    y_true.append(row[target_col])
    y_pred.append(result[target_col])

# -----------------------------
# EVALUATION METRICS
# -----------------------------
print("Bayesian Network Evaluation:")
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, zero_division=0))
print("Recall   :", recall_score(y_true, y_pred, zero_division=0))
print("F1 Score :", f1_score(y_true, y_pred, zero_division=0))

# -----------------------------
# PREDICT NEW PATIENT
# -----------------------------
sample = test_data.iloc[0]
evidence = sample.drop(target_col).to_dict()

prediction = infer.map_query(
    variables=[target_col],
    evidence=evidence
)

print("\nNew Patient Prediction:","Diabetic" if prediction[target_col] == 1 else "Non-Diabetic")
