import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv(r"C:\Users\gmuke\Desktop\ML LAB R23\FINAL MAM\P9\9bCustomerPersonality.csv")

# -----------------------------
# REMOVE ID / DATE IF PRESENT
# -----------------------------
data.drop(columns=["ID", "Dt_Customer"], errors="ignore", inplace=True)

# -----------------------------
# CONVERT CATEGORICAL DATA
# -----------------------------
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))

# -----------------------------
# HANDLE MISSING VALUES
# -----------------------------
data = data.fillna(data.mean())

# -----------------------------
# SELECT NUMERIC COLUMNS
# -----------------------------
data = data.select_dtypes(include=['int64','float64'])

# -----------------------------
# CREATE TARGET CLASS
# (based on last column)
# -----------------------------
target_column = data.columns[-1]

# Convert target to categorical classes
data["Class"] = pd.qcut(data[target_column], q=3, labels=[0,1,2])

# -----------------------------
# FEATURES & TARGET
# -----------------------------
X = data.drop(columns=["Class"])
y = data["Class"]

# -----------------------------
# SCALE FEATURES
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

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
    actual = y_test.iloc[i]
    pred = y_pred[i]

    if actual == pred:
        correct += 1
        print(f"Correct -> Actual:{actual} Predicted:{pred}")
    else:
        wrong += 1
        print(f"Wrong   -> Actual:{actual} Predicted:{pred}")

# -----------------------------
# RESULTS
# -----------------------------
print("\nTotal Correct Predictions :", correct)
print("Total Wrong Predictions   :", wrong)

print("\nEvaluation Metrics")

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall   :", recall_score(y_test, y_pred, average='macro'))
print("F1 Score :", f1_score(y_test, y_pred, average='macro'))