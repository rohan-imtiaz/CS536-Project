# CS 432 Project — 06_ari_table.py (Reproducing Figure 6: ARI table across distance measures and datasets)

# INSTRUCTIONS:
# Run after 01_generate_data.py.
# Embeds each dataset using MDS on pairwise distance matrices (Euclidean and dc-dist),
# then runs DBSCAN, kMeans, and spectral clustering on each embedding.
# Reports adjusted rand index (ARI) for each combination.
# For the full paper table with all benchmark datasets, also run:
#   cd repo && python compare_clustering.py
# Working directory must be 25280058_Project when running this script.
# Saves figures/fig6_ari_table.png.
# python 06_ari_table.py

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import MDS
from sklearn.metrics import adjusted_rand_score as ari

sys.path.insert(0, "repo") # Adding repo to path so all authors' modules are importable

from distance_metric import get_dc_dist_matrix
from DBSCAN import DBSCAN as DcDBSCAN

# Running all three clustering algorithms on a given MDS embedding of a distance matrix

def run_clusterings(dist_matrix, n_clusters, mu, mds_dim):

    mds_model = MDS(n_components = mds_dim, dissimilarity = "precomputed",
                    random_state = 42, n_init = 1)
    X_embedded = mds_model.fit_transform(dist_matrix)

    clustering_results = {}

    # DBSCAN using authors' implementation with epsilon from percentile of non-zero distances

    eps_val = np.percentile(dist_matrix[dist_matrix > 0], 15)
    dbscan_model = DcDBSCAN(eps = eps_val, min_pts = mu, cluster_type = "corepoints")
    dbscan_model.fit(X_embedded)
    clustering_results["DBSCAN"] = dbscan_model.labels_

    # kMeans on the embedded space

    kmeans_model = KMeans(n_clusters = n_clusters, random_state = 42, n_init = 5)
    clustering_results["kMeans"] = kmeans_model.fit_predict(X_embedded)

    # Spectral clustering with RBF kernel on the embedded space (matching paper setup)

    spectral_model = SpectralClustering(n_clusters = n_clusters, affinity = "rbf",
                                        random_state = 42)
    clustering_results["Spectral"] = spectral_model.fit_predict(X_embedded)

    return clustering_results

# Dataset configurations: name -> (X path, y path, true number of clusters)

dataset_configs = {"d1":    ("data/synthetic/d1_X.npy",    "data/synthetic/d1_y.npy",    5),
                   "moons": ("data/synthetic/moons_X.npy", "data/synthetic/moons_y.npy", 2)}

distance_configs = [("Euclidean", None), # None means use cdist with Euclidean metric directly
                    ("dc mu=3",   3),
                    ("dc mu=5",   5),
                    ("dc mu=10",  10)]

mds_dims = [2, 10]

print(f"{'Dataset':<10} {'Distance':<12} {'dim':<5} {'DBSCAN':>8} {'kMeans':>8} {'Spectral':>10}")
print("-" * 58)

ari_rows = [] # Collecting rows for the summary bar chart

for dataset_name, (x_path, y_path, n_clusters) in dataset_configs.items():

    X = np.load(x_path)
    y_true = np.load(y_path)

    mask = y_true >= 0 # Removing noise points before computing ARI
    X = X[mask]
    y_true = y_true[mask]

    for dist_name, mu in distance_configs:

        if mu is None:
            dist_matrix = cdist(X, X, metric = "euclidean")
        else:
            dist_matrix = get_dc_dist_matrix(X, n_neighbors = 15, min_points = mu)

        for dim in mds_dims:

            cluster_results = run_clusterings(dist_matrix, n_clusters, mu if mu else 5, dim)

            ari_scores = {method: ari(y_true, labels)
                          for method, labels in cluster_results.items()} # Computing ARI for each method against ground truth

            print(f"{dataset_name:<10} {dist_name:<12} {dim:<5} "
                  f"{ari_scores['DBSCAN']:>8.3f} "
                  f"{ari_scores['kMeans']:>8.3f} "
                  f"{ari_scores['Spectral']:>10.3f}")

            ari_rows.append({"dataset":  dataset_name,
                             "distance": dist_name,
                             "dim":      dim,
                             "DBSCAN":   ari_scores["DBSCAN"],
                             "kMeans":   ari_scores["kMeans"],
                             "Spectral": ari_scores["Spectral"]})

    print()

# Saving a summary bar chart of ARI by distance measure averaged across methods and dims

distance_labels = [name for name, _ in distance_configs]
mean_ari_per_distance = []

for dist_name, _ in distance_configs:

    matching_rows = [row for row in ari_rows if row["distance"] == dist_name]
    all_ari_values = ([row["DBSCAN"]   for row in matching_rows] +
                      [row["kMeans"]   for row in matching_rows] +
                      [row["Spectral"] for row in matching_rows])
    mean_ari_per_distance.append(np.mean(all_ari_values))

plt.figure(figsize = (8, 5))
plt.bar(distance_labels, mean_ari_per_distance, color = "#378ADD", alpha = 0.85)
plt.ylabel("Mean ARI (across datasets, methods, dims)")
plt.title("Fig 6: Mean ARI by distance measure")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("figures/fig6_ari_table.png", dpi = 150, bbox_inches = "tight")
plt.show()

print("Saved figures/fig6_ari_table.png")
print("To reproduce the full paper table, run: cd repo && python compare_clustering.py")
