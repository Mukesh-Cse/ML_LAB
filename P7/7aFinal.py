import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("C:\\Users\\gmuke\\Desktop\\ML LAB R23\\FINAL MAM\\P7\\7aHeart100.csv")

# Drop ID column if present
if data.columns[0].lower() in ["id", "patientid"]:
    data = data.drop(columns=[data.columns[0]])

# Detect target column
target_col = None
for col in ["target", "HeartDisease", "output"]:
    if col in data.columns:
        target_col = col
        break

if target_col is None:
    raise ValueError("Target column not found")

# -----------------------------
# FORCE TARGET TO BINARY
# -----------------------------
data[target_col] = data[target_col].astype(int)
data[target_col] = data[target_col].apply(lambda x: 1 if x > 0 else 0)

# -----------------------------
# CONVERT EVERYTHING TO INT
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
features = [c for c in data.columns if c != target_col]
edges = [(f, target_col) for f in features]

model = DiscreteBayesianNetwork(edges)

# -----------------------------
# TRAIN MODEL
# -----------------------------
model.fit(train_data, estimator=MaximumLikelihoodEstimator)

# -----------------------------
# STORE VALID STATES
# -----------------------------
valid_states = {
    col: sorted(train_data[col].unique())
    for col in features
}

# -----------------------------
# INFERENCE
# -----------------------------
infer = VariableElimination(model)

y_true = []
y_pred = []

for _, row in test_data.iterrows():
    evidence = {}

    for col in features:
        val = int(row[col])
        # Clip value to known states
        if val not in valid_states[col]:
            val = valid_states[col][0]
        evidence[col] = val

    result = infer.map_query(variables=[target_col], evidence=evidence)

    y_true.append(int(row[target_col]))
    y_pred.append(int(result[target_col]))

# -----------------------------
# EVALUATION
# -----------------------------
print("Bayesian Network Evaluation:")
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, zero_division=0))
print("Recall   :", recall_score(y_true, y_pred, zero_division=0))
print("F1 Score :", f1_score(y_true, y_pred, zero_division=0))

# -----------------------------
# NEW PATIENT PREDICTION
# -----------------------------
sample = test_data.iloc[0]
evidence = {}

for col in features:
    val = int(sample[col])
    if val not in valid_states[col]:
        val = valid_states[col][0]
    evidence[col] = val

prediction = infer.map_query(
    variables=[target_col],
    evidence=evidence
)

print("\nNew Patient Prediction:",
      "Heart Disease" if prediction[target_col] == 1 else "No Heart Disease")
