import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# LOAD DATA
# -----------------------------
# Wisconsin Breast Cancer dataset (no header)
cols = [
    "ID","ClumpThickness","UniformCellSize","UniformCellShape","MarginalAdhesion",
    "SingleEpithelialCellSize","BareNuclei","BlandChromatin",
    "NormalNucleoli","Mitoses","Class"
]

data = pd.read_csv("C:\\Users\\gmuke\\Desktop\\ML LAB R23\\FINAL MAM\\P5\\5eBreastCancer1000.data", header=None, names=cols)

# Drop ID column
data.drop(columns=["ID"], inplace=True)

# Replace missing values ("?")
data.replace("?", pd.NA, inplace=True)
data.dropna(inplace=True)

# Convert all columns to numeric
data = data.astype(int)

# Separate features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values   # 2 = Benign, 4 = Malignant

# Convert target to binary (0/1)
y = (y == 4).astype(int)

# -----------------------------
# TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1, stratify=y
)

# -----------------------------
# FEATURE SCALING
# -----------------------------
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# TRAIN NAIVE BAYES
# -----------------------------
model = GaussianNB()
model.fit(X_train, y_train)

# -----------------------------
# TESTING
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# EVALUATION METRICS
# -----------------------------
print("Evaluation Metrics:")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall   :", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score :", f1_score(y_test, y_pred, zero_division=0))

# -----------------------------
# CLASSIFY NEW SAMPLE
# -----------------------------
new_sample = X_test[0].reshape(1, -1)
new_sample = scaler.transform(new_sample)

prediction = model.predict(new_sample)[0]
print("\nNew Sample Prediction:",
      "Malignant" if prediction == 1 else "Benign")
