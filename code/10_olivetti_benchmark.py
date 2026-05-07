# CS 432 Project — 10_olivetti_benchmark.py (ARI benchmark on real Olivetti faces dataset)

# INSTRUCTIONS:
# Run after 01_generate_data.py.
# Runs the full ARI comparison (Euclidean vs dc-dist, three clustering algorithms)
# on the Olivetti faces dataset, which is the one real-world benchmark available
# without an external download. Results extend the Fig 6 ARI table to real data.
# Working directory must be 25280058_Project when running this script.
# Saves figures/fig10_olivetti_ari.png.
# python 10_olivetti_benchmark.py

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import MDS
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, "repo") # Adding repo to path so all authors' modules are importable

from distance_metric import get_dc_dist_matrix
from DBSCAN import DBSCAN as DcDBSCAN

# Loading Olivetti faces (400 images, 4096 features, 40 classes)

X = np.load("data/synthetic/olivetti_X.npy")
y = np.load("data/synthetic/olivetti_y.npy")

n_clusters = 40
n_neighbors = 15

print(f"Olivetti dataset: X = {X.shape}, classes = {n_clusters}")
print("Note: Olivetti is high-dimensional (4096 features). MDS embedding may take a few minutes.\n")

# Deriving a good epsilon for DBSCAN using the k-nearest-neighbour distance plot
# For each point, we find the distance to its mu-th nearest neighbour and sort
# these values. The knee of this curve is the natural epsilon for DBSCAN.
# This avoids the flat-percentile approach which fails in high dimensions.

def find_epsilon_for_target_clusters(X_embedded, mu, target_k):

    nbrs = NearestNeighbors(n_neighbors = mu).fit(X_embedded)
    distances, _ = nbrs.kneighbors(X_embedded)
    kth_distances = np.sort(distances[:, -1])

    # Scanning candidate epsilons from the kNN distance distribution
    # and selecting the one that produces a cluster count closest to target_k

    candidates = np.percentile(kth_distances, np.linspace(10, 95, 50))
    best_eps = candidates[0]
    best_diff = float("inf")

    for eps_candidate in candidates:

        test_model = DcDBSCAN(eps = eps_candidate, min_pts = mu, cluster_type = "corepoints")
        test_model.fit(X_embedded)
        test_labels = test_model.labels_
        n_found = len(set(test_labels[test_labels >= 0]))
        diff = abs(n_found - target_k)

        if diff < best_diff:
            best_diff = diff
            best_eps = eps_candidate

    return float(best_eps)

# Running all three clustering algorithms on a given MDS-embedded distance matrix

def run_clusterings(dist_matrix, n_clusters, mu, mds_dim):

    print(f"  Running MDS to {mds_dim} dimensions...")
    mds_model = MDS(n_components = mds_dim, dissimilarity = "precomputed",
                    random_state = 42, n_init = 1, max_iter = 300)
    X_embedded = mds_model.fit_transform(dist_matrix)

    clustering_results = {}

    # Computing epsilon from the knee of the kNN distance curve on the embedded space

    eps_val = find_epsilon_for_target_clusters(X_embedded, mu, n_clusters)
    print(f"  Derived epsilon = {eps_val:.4f} for mu = {mu}, dim = {mds_dim}")

    dbscan_model = DcDBSCAN(eps = eps_val, min_pts = mu, cluster_type = "corepoints")
    dbscan_model.fit(X_embedded)
    dbscan_raw = dbscan_model.labels_
    n_found_clusters = len(set(dbscan_raw[dbscan_raw >= 0]))
    print(f"  DBSCAN found {n_found_clusters} clusters, {int(np.sum(dbscan_raw == -1))} noise points")
    clustering_results["DBSCAN"] = dbscan_raw


    # Checking if DBSCAN found any clusters; if all noise, relaxing epsilon by 20% and retrying

    n_found_clusters = len(set(dbscan_raw[dbscan_raw >= 0]))

    if n_found_clusters == 0:

        eps_val_relaxed = eps_val * 1.2
        print(f"  No clusters found, relaxing epsilon to {eps_val_relaxed:.4f} and retrying...")
        dbscan_retry = DcDBSCAN(eps = eps_val_relaxed, min_pts = mu, cluster_type = "corepoints")
        dbscan_retry.fit(X_embedded)
        dbscan_raw = dbscan_retry.labels_
        n_found_clusters = len(set(dbscan_raw[dbscan_raw >= 0]))

    kmeans_model = KMeans(n_clusters = n_clusters, random_state = 42, n_init = 5)
    clustering_results["kMeans"] = kmeans_model.fit_predict(X_embedded)

    spectral_model = SpectralClustering(n_clusters = n_clusters, affinity = "rbf", random_state = 42)
    clustering_results["Spectral"] = spectral_model.fit_predict(X_embedded)

    return clustering_results

# Computing ARI only on points that DBSCAN assigned to a cluster (excluding noise)
# Using all points for kMeans and Spectral since they do not produce noise labels

def compute_ari_scores(cluster_results, y_true):

    ari_scores = {}

    for method, labels in cluster_results.items():

        if method == "DBSCAN":
            mask = labels >= 0 # Excluding noise points from ARI computation
            if mask.sum() < 2:
                ari_scores[method] = 0.000
            else:
                ari_scores[method] = float(ari(y_true[mask], labels[mask]))
        else:
            ari_scores[method] = float(ari(y_true, labels))

    return ari_scores

# Distance configurations: name and mu value (None means Euclidean)
# Using mu = 3 as the minimum since Olivetti has 10 images per class

distance_configs = [("Euclidean", 3),  # mu = 3 used for DBSCAN even under Euclidean
                    ("dc mu=3",   3),
                    ("dc mu=5",   5),
                    ("dc mu=10",  10)]

mds_dims = [2, 10]

print(f"{'Distance':<12} {'dim':<5} {'DBSCAN':>8} {'kMeans':>8} {'Spectral':>10}")
print("-" * 48)

ari_rows = [] # Collecting all rows for the summary figure

for dist_name, mu in distance_configs:

    if dist_name == "Euclidean":
        print(f"\nComputing Euclidean distance matrix...")
        dist_matrix = cdist(X, X, metric = "euclidean")
    else:
        print(f"\nComputing dc-dist matrix (mu = {mu})...")
        dist_matrix = get_dc_dist_matrix(X, n_neighbors = n_neighbors, min_points = mu)

    for dim in mds_dims:

        cluster_results = run_clusterings(dist_matrix, n_clusters, mu, dim)
        ari_scores = compute_ari_scores(cluster_results, y)

        print(f"  {dist_name:<12} {dim:<5} "
              f"{ari_scores['DBSCAN']:>8.3f} "
              f"{ari_scores['kMeans']:>8.3f} "
              f"{ari_scores['Spectral']:>10.3f}")

        ari_rows.append({"distance": dist_name,
                         "dim":      dim,
                         "DBSCAN":   ari_scores["DBSCAN"],
                         "kMeans":   ari_scores["kMeans"],
                         "Spectral": ari_scores["Spectral"]})

# Saving a grouped bar chart summarising ARI by distance and method at dim = 2

dim2_rows = [row for row in ari_rows if row["dim"] == 2]
distance_names  = [row["distance"] for row in dim2_rows]
dbscan_scores   = [row["DBSCAN"]   for row in dim2_rows]
kmeans_scores   = [row["kMeans"]   for row in dim2_rows]
spectral_scores = [row["Spectral"] for row in dim2_rows]

x_positions = np.arange(len(distance_names))
bar_width = 0.25

fig, ax = plt.subplots(figsize = (10, 5))

ax.bar(x_positions - bar_width, dbscan_scores,   bar_width, label = "DBSCAN",   color = "#378ADD", alpha = 0.85)
ax.bar(x_positions,             kmeans_scores,   bar_width, label = "kMeans",   color = "#1D9E75", alpha = 0.85)
ax.bar(x_positions + bar_width, spectral_scores, bar_width, label = "Spectral", color = "#EF9F27", alpha = 0.85)

ax.set_xticks(x_positions)
ax.set_xticklabels(distance_names, fontsize = 9)
ax.set_ylabel("ARI vs ground truth")
ax.set_title("Fig 10 — Olivetti faces ARI by distance measure and algorithm (dim = 2)")
ax.set_ylim(0, 1)
ax.legend(fontsize = 9)
plt.tight_layout()
plt.savefig("figures/fig10_olivetti_ari.png", dpi = 150, bbox_inches = "tight")
plt.show()

print("\nSaved figures/fig10_olivetti_ari.png")
