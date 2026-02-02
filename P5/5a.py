import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("C:\\Users\\gmuke\\Desktop\\ML LAB R23\\FINAL MAM\\P5\\5aIris1000.csv")

# Drop ID column if present
if data.shape[1] == 6:
    data = data.drop(columns=[data.columns[0]])

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Encode class labels
le = LabelEncoder()
y = le.fit_transform(y)

# -----------------------------
# TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1, stratify=y
)

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
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall   :", recall_score(y_test, y_pred, average="weighted"))
print("F1 Score :", f1_score(y_test, y_pred, average="weighted"))

# -----------------------------
# CLASSIFY NEW SAMPLE
# -----------------------------
new_sample = X_test.iloc[0:1]
pred = model.predict(new_sample)

print("\nNew Sample Prediction:", le.inverse_transform(pred)[0])
