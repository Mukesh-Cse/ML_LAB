import numpy as np

# -----------------------------
# ACTIVATION FUNCTIONS
# -----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# -----------------------------
# TRAINING DATA (XOR)
# -----------------------------
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# -----------------------------
# INITIALIZE WEIGHTS
# -----------------------------
np.random.seed(1)
W1 = np.random.uniform(size=(2,2))   # input → hidden
W2 = np.random.uniform(size=(2,1))   # hidden → output

lr = 0.5
epochs = 10000

# -----------------------------
# TRAINING (BACK-PROPAGATION)
# -----------------------------
for _ in range(epochs):

    # ---- FORWARD PROPAGATION ----
    hidden_input = np.dot(X, W1)
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2)
    y_pred = sigmoid(final_input)

    # ---- ERROR ----
    error = y - y_pred

    # ---- BACKWARD PROPAGATION ----
    d_output = error * sigmoid_derivative(y_pred)
    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden_output)

    # ---- UPDATE WEIGHTS ----
    W2 += hidden_output.T.dot(d_output) * lr
    W1 += X.T.dot(d_hidden) * lr

# -----------------------------
# TESTING
# -----------------------------
print("Final Predictions:")
for i in range(len(X)):
    print(X[i], "→", round(y_pred[i][0], 3))
