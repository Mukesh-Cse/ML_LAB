import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("C:\\Users\\gmuke\\Desktop\\ML LAB R23\\FINAL MAM\\P8\\8aCC3000.csv")

# Keep numeric columns only
data = data.select_dtypes(include=['int64','float64'])

# -----------------------------
# HANDLE MISSING VALUES
# -----------------------------
data = data.fillna(data.mean())

# -----------------------------
# SCALE DATA
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(data)

# Number of clusters
k = 3

# -----------------------------
# K-MEANS
# -----------------------------
kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
kmeans_labels = kmeans.fit_predict(X)

# -----------------------------
# EM (Gaussian Mixture)
# -----------------------------
gmm = GaussianMixture(n_components=k, random_state=0)
gmm_labels = gmm.fit_predict(X)

# -----------------------------
# EVALUATION
# -----------------------------
kmeans_score = silhouette_score(X, kmeans_labels)
gmm_score = silhouette_score(X, gmm_labels)

# -----------------------------
# OUTPUT
# -----------------------------
print("\nK-Means Cluster Labels:")
print(kmeans_labels)

print("\nEM Cluster Labels:")
print(gmm_labels)

print("\nClustering Quality:")

print("K-Means Silhouette Score :", round(kmeans_score,3))
print("EM Silhouette Score      :", round(gmm_score,3))

if kmeans_score > gmm_score:
    print("\nK-Means produced better clustering.")
elif gmm_score > kmeans_score:
    print("\nEM algorithm produced better clustering.")
else:
    print("\nBoth algorithms produced similar clustering.")