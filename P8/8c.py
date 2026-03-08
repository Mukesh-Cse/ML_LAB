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
data = pd.read_csv("C:\\Users\\gmuke\\Desktop\\ML LAB R23\\FINAL MAM\\P8\\8cOnlineRetail3000.csv", encoding="latin1")

# -----------------------------
# DROP TEXT COLUMNS
# -----------------------------
drop_cols = ["InvoiceNo","StockCode","Description","InvoiceDate"]
for col in drop_cols:
    if col in data.columns:
        data = data.drop(columns=[col])

# -----------------------------
# CONVERT COUNTRY TO NUMERIC
# -----------------------------
if "Country" in data.columns:
    data["Country"] = data["Country"].astype("category").cat.codes

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
# CLUSTER COUNT
# -----------------------------
k = 3

# -----------------------------
# KMEANS
# -----------------------------
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X)

# -----------------------------
# EM (GMM)
# -----------------------------
gmm = GaussianMixture(n_components=k, random_state=42)
gmm_labels = gmm.fit_predict(X)

# -----------------------------
# SAMPLE DATA FOR SILHOUETTE
# (fast computation)
# -----------------------------
sample_size = min(500, len(X))

kmeans_score = silhouette_score(X, kmeans_labels, sample_size=sample_size)
gmm_score = silhouette_score(X, gmm_labels, sample_size=sample_size)

# -----------------------------
# OUTPUT
# -----------------------------
print("\nK-Means Cluster Labels:")
print(kmeans_labels[:20], "...")

print("\nEM Cluster Labels:")
print(gmm_labels[:20], "...")

print("\nClustering Quality Comparison")
print("K-Means Silhouette Score :", round(kmeans_score,3))
print("EM Silhouette Score      :", round(gmm_score,3))

if kmeans_score > gmm_score:
    print("\nK-Means produced better clustering.")
elif gmm_score > kmeans_score:
    print("\nEM algorithm produced better clustering.")
else:
    print("\nBoth algorithms produced similar clustering.")