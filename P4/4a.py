import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# LOAD DATA (FIXED)
# -----------------------------
data = pd.read_csv("C:\\Users\\gmuke\\Desktop\\ML LAB R23\\FINAL MAM\\P4\\4aIris1000.csv")

# Drop ID column if present
if data.shape[1] == 6:
    data = data.drop(columns=[data.columns[0]])

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# -----------------------------
# ENCODE TARGET
# -----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# One-hot encoding
y_onehot = np.eye(len(np.unique(y_encoded)))[y_encoded]

# -----------------------------
# FEATURE SCALING
# -----------------------------
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

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
input_neurons = X.shape[1]      # 4
hidden_neurons = 6
output_neurons = y_onehot.shape[1]  # 3

W1 = np.random.randn(input_neurons, hidden_neurons)
W2 = np.random.randn(hidden_neurons, output_neurons)

lr = 0.5
epochs = 5000

# -----------------------------
# TRAINING (BACK-PROPAGATION)
# -----------------------------
for _ in range(epochs):

    hidden_out = sigmoid(np.dot(X, W1))
    output_out = sigmoid(np.dot(hidden_out, W2))

    error = y_onehot - output_out

    d_output = error * sigmoid_derivative(output_out)
    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden_out)

    W2 += hidden_out.T.dot(d_output) * lr
    W1 += X.T.dot(d_hidden) * lr

# -----------------------------
# TESTING
# -----------------------------
final_output = sigmoid(np.dot(sigmoid(np.dot(X, W1)), W2))
y_pred = np.argmax(final_output, axis=1)
y_true = y_encoded

# -----------------------------
# EVALUATION METRICS
# -----------------------------
print("Evaluation Metrics:")
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, average="weighted"))
print("Recall   :", recall_score(y_true, y_pred, average="weighted"))
print("F1 Score :", f1_score(y_true, y_pred, average="weighted"))

# -----------------------------
# CLASSIFY NEW SAMPLE (FIXED)
# -----------------------------
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])
new_sample = scaler.transform(new_sample)

hidden = sigmoid(np.dot(new_sample, W1))
output = sigmoid(np.dot(hidden, W2))

print("\nNew Sample Prediction:",
      le.inverse_transform([np.argmax(output)])[0])
