import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("C:\\Users\\gmuke\\Desktop\\ML LAB R23\\FINAL MAM\\P4\\4eHeart10.csv")

# If dataset has an ID column, drop it
if data.columns[0].lower() in ["id", "patientid"]:
    data = data.drop(columns=[data.columns[0]])

# Identify target column (common names handled)
for col in ["target", "HeartDisease", "heart_disease", "output"]:
    if col in data.columns:
        target_col = col
        break

X = data.drop(columns=[target_col]).values
y = data[target_col].values.reshape(-1, 1)

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
# ACTIVATION FUNCTIONS
# -----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# -----------------------------
# INITIALIZE NETWORK
# -----------------------------
np.random.seed(1)
input_neurons = X_train.shape[1]
hidden_neurons = 8
output_neurons = 1

W1 = np.random.randn(input_neurons, hidden_neurons) * 0.1
W2 = np.random.randn(hidden_neurons, output_neurons) * 0.1

lr = 0.3
epochs = 6000

# -----------------------------
# TRAINING (BACK-PROPAGATION)
# -----------------------------
for _ in range(epochs):

    hidden_out = sigmoid(np.dot(X_train, W1))
    output_out = sigmoid(np.dot(hidden_out, W2))

    error = y_train - output_out

    d_out = error * sigmoid_derivative(output_out)
    d_hidden = d_out.dot(W2.T) * sigmoid_derivative(hidden_out)

    W2 += hidden_out.T.dot(d_out) * lr
    W1 += X_train.T.dot(d_hidden) * lr

# -----------------------------
# TESTING
# -----------------------------
test_hidden = sigmoid(np.dot(X_test, W1))
test_output = sigmoid(np.dot(test_hidden, W2))

y_pred = (test_output >= 0.5).astype(int)

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
pred = sigmoid(np.dot(sigmoid(np.dot(new_sample, W1)), W2))

print("\nNew Sample Prediction:",
      "Heart Disease" if pred >= 0.5 else "No Heart Disease")
