import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("C:\\Users\\gmuke\\Desktop\\ML LAB R23\\FINAL MAM\\P4\\4cDiabetes1000.csv")

# Separate features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

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
input_neurons = X.shape[1]
hidden_neurons = 8
output_neurons = 1

W1 = np.random.randn(input_neurons, hidden_neurons)
W2 = np.random.randn(hidden_neurons, output_neurons)

lr = 0.3
epochs = 5000

# -----------------------------
# TRAINING (BACK-PROPAGATION)
# -----------------------------
for _ in range(epochs):

    # Forward pass
    hidden_out = sigmoid(np.dot(X, W1))
    output_out = sigmoid(np.dot(hidden_out, W2))

    # Error
    error = y - output_out

    # Backward pass
    d_output = error * sigmoid_derivative(output_out)
    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden_out)

    # Update weights
    W2 += hidden_out.T.dot(d_output) * lr
    W1 += X.T.dot(d_hidden) * lr

# -----------------------------
# TESTING
# -----------------------------
final_output = sigmoid(np.dot(sigmoid(np.dot(X, W1)), W2))
y_pred = (final_output >= 0.5).astype(int)

# -----------------------------
# EVALUATION METRICS
# -----------------------------
print("Evaluation Metrics:")
print("Accuracy :", accuracy_score(y, y_pred))
print("Precision:", precision_score(y, y_pred))
print("Recall   :", recall_score(y, y_pred))
print("F1 Score :", f1_score(y, y_pred))

# -----------------------------
# CLASSIFY NEW SAMPLE
# -----------------------------
new_sample = X[0].reshape(1, -1)   # example patient
prediction = sigmoid(np.dot(sigmoid(np.dot(new_sample, W1)), W2))

print("\nNew Sample Prediction:",
      "Diabetic" if prediction >= 0.5 else "Non-Diabetic")
