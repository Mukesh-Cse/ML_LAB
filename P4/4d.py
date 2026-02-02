import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# LOAD DATA
# -----------------------------
# Wine dataset: first column = class label
data = pd.read_csv("C:\\Users\\gmuke\\Desktop\\ML LAB R23\\FINAL MAM\\P4\\4dWine1000.data", header=None)

X = data.iloc[:, 1:].values        # features
y = data.iloc[:, 0].values         # class labels (1,2,3)

# Convert labels to 0,1,2
y = y - 1

# One-hot encoding
y_onehot = np.eye(len(np.unique(y)))[y]

# -----------------------------
# TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.25, random_state=1, stratify=y
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
input_neurons = X_train.shape[1]   # 13
hidden_neurons = 10
output_neurons = y_train.shape[1]  # 3

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

    d_output = error * sigmoid_derivative(output_out)
    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden_out)

    W2 += hidden_out.T.dot(d_output) * lr
    W1 += X_train.T.dot(d_hidden) * lr

# -----------------------------
# TESTING
# -----------------------------
test_hidden = sigmoid(np.dot(X_test, W1))
test_output = sigmoid(np.dot(test_hidden, W2))

y_pred = np.argmax(test_output, axis=1)
y_true = np.argmax(y_test, axis=1)

# -----------------------------
# EVALUATION METRICS
# -----------------------------
print("Evaluation Metrics:")
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, average="weighted", zero_division=0))
print("Recall   :", recall_score(y_true, y_pred, average="weighted", zero_division=0))
print("F1 Score :", f1_score(y_true, y_pred, average="weighted", zero_division=0))

# -----------------------------
# CLASSIFY NEW SAMPLE
# -----------------------------
new_sample = X_test[0].reshape(1, -1)
new_sample = scaler.transform(new_sample)

hidden = sigmoid(np.dot(new_sample, W1))
output = sigmoid(np.dot(hidden, W2))

print("\nNew Sample Prediction: Class", np.argmax(output) + 1)
