import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# LOAD DATASET
# -----------------------------
data = pd.read_csv("C:\\Users\\gmuke\\Desktop\\ML LAB R23\\FINAL MAM\\P9\\9aIris.csv")

# -----------------------------
# FEATURES & TARGET
# -----------------------------
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Convert species names to numbers
le = LabelEncoder()
y = le.fit_transform(y)

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------
# KNN MODEL
# -----------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# -----------------------------
# PREDICTIONS
# -----------------------------
y_pred = knn.predict(X_test)

print("\nPredictions:\n")

correct = 0
wrong = 0

for i in range(len(y_test)):
    actual = y_test[i]
    predicted = y_pred[i]

    if actual == predicted:
        correct += 1
        print(f"Correct  -> Actual:{actual} Predicted:{predicted}")
    else:
        wrong += 1
        print(f"Wrong    -> Actual:{actual} Predicted:{predicted}")

# -----------------------------
# RESULTS
# -----------------------------
print("\nTotal Correct Predictions :", correct)
print("Total Wrong Predictions   :", wrong)

print("\nEvaluation Metrics")

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="macro"))
print("Recall   :", recall_score(y_test, y_pred, average="macro"))
print("F1 Score :", f1_score(y_test, y_pred, average="macro"))