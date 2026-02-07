import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("C:\\Users\\gmuke\\Desktop\\ML LAB R23\\FINAL MAM\\P7\\7eWeather1000.csv")

# Remove ID column if present
if "Day" in data.columns:
    data = data.drop(columns=["Day"])

# Detect target column
target_col = None
for col in ["Play", "PlayTennis", "target", "Label", "Class"]:
    if col in data.columns:
        target_col = col
        break

if target_col is None:
    raise ValueError("Target column not found")

# -----------------------------
# CONVERT CATEGORICAL TO NUMERIC
# -----------------------------
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = data[col].astype("category").cat.codes

# -----------------------------
# USE ALL FEATURES
# -----------------------------
features = [c for c in data.columns if c != target_col]

# -----------------------------
# HANDLE MISSING VALUES
# -----------------------------
data = data.fillna(0).astype(int)

# -----------------------------
# TRAIN-TEST SPLIT
# -----------------------------
train_data, test_data = train_test_split(
    data,
    test_size=0.25,
    random_state=1
)

# -----------------------------
# BUILD BAYESIAN NETWORK
# -----------------------------
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
# PREDICT NEW SAMPLE
# -----------------------------
sample = test_data.iloc[0]
evidence = sample.drop(target_col).to_dict()

prediction = infer.map_query(
    variables=[target_col],
    evidence=evidence
)

print("\nNew Sample Prediction:",  "Yes" if prediction[target_col] == 1 else "No")
