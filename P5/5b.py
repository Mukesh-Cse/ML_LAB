import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("C:\\Users\\gmuke\\Desktop\\ML LAB R23\\FINAL MAM\\P5\\5bspam1000.csv", encoding="latin-1")

# Keep only required columns (common spam dataset format)
data = data.iloc[:, :2]
data.columns = ["label", "message"]

# Convert labels: spam -> 1, ham -> 0
data["label"] = data["label"].map({"spam": 1, "ham": 0})

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(data["message"])
y = data["label"]

# -----------------------------
# TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1, stratify=y
)

# -----------------------------
# TRAIN NAIVE BAYES
# -----------------------------
model = MultinomialNB()
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
# CLASSIFY NEW MESSAGE
# -----------------------------
new_message = ["Congratulations! You have won a free prize"]
new_vec = vectorizer.transform(new_message)

prediction = model.predict(new_vec)[0]
print("\nNew Message Prediction:",
      "Spam" if prediction == 1 else "Not Spam")
