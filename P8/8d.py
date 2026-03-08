import os
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("C:\\Users\\gmuke\\Desktop\\ML LAB R23\\FINAL MAM\\P8\\8dObesity3000.csv")

# -----------------------------
# CONVERT CATEGORICAL COLUMNS
# -----------------------------
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = data[col].astype("category").cat.codes

# -----------------------------
# HANDLE MISSING VALUES
# -----------------------------
data = data.fillna(data.mean())

# -----------------------------
# SCALE DATA
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(data)

# -----------------------------
# NUMBER OF CLUSTERS
# -----------------------------
k = 4

# -----------------------------
# K-MEANS CLUSTERING
# -----------------------------
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X)

# -----------------------------
# EM CLUSTERING (GMM)
# -----------------------------
gmm = GaussianMixture(n_components=k, random_state=42)
gmm_labels = gmm.fit_predict(X)

# -----------------------------
# FAST SILHOUETTE SCORE
# -----------------------------
sample_size = min(500, len(X))

kmeans_score = silhouette_score(X, kmeans_labels, sample_size=sample_size)
gmm_score = silhouette_score(X, gmm_labels, sample_size=sample_size)

# -----------------------------
# OUTPUT
# -----------------------------
print("\nK-Means Cluster Labels (first 20):")
print(kmeans_labels[:20])

print("\nEM Cluster Labels (first 20):")
print(gmm_labels[:20])

print("\nClustering Quality Comparison")

print("K-Means Silhouette Score :", round(kmeans_score,3))
print("EM Silhouette Score      :", round(gmm_score,3))

if kmeans_score > gmm_score:
    print("\nK-Means produced better clustering.")
elif gmm_score > kmeans_score:
    print("\nEM algorithm produced better clustering.")
else:
    print("\nBoth algorithms produced similar clustering quality.")