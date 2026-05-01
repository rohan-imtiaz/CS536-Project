# CS 432 Project — 03_equivalence.py (Reproducing Figure 3: algorithm equivalence under dc-dist)

# INSTRUCTIONS:
# Run after 01_generate_data.py and 02_verify_dcdist.py.
# Shows that DBSCAN*, k-center, and ultrametric spectral clustering
# produce identical clusterings when operating in dc-dist space.
# Working directory must be 25280058_Project when running this script.
# Saves figures/fig3_equivalence.png.
# python 03_equivalence.py

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "repo") # Adding repo to path so all authors' modules are importable

from density_tree import make_tree
from cluster_tree import dc_clustering
from DBSCAN import DBSCAN as DcDBSCAN
from distance_metric import get_dc_dist_matrix
from SpectralClustering import get_lambdas, get_sim_mx, run_spectral_clustering
from sklearn.metrics import normalized_mutual_info_score as nmi

# Loading the two moons dataset (clear two-cluster structure for clean visual)

X = np.load("data/synthetic/moons_X.npy")
y = np.load("data/synthetic/moons_y.npy")

min_pts = 5
k = 2
n_neighbors = 15

# Step 1: Building the ultrametric tree (also computes dc-dists internally via make_tree)

print("Building density tree...")
root, dc_dists = make_tree(X, y, min_points = min_pts, n_neighbors = n_neighbors)

# Step 2: Running k-center on dc-dist (Algorithm 1 from the paper, Tree-k-Center)

print("Running k-center on dc-dist...")
pred_labels, centers, epsilons = dc_clustering(root,
                                               num_points = len(y),
                                               k = k,
                                               min_points = min_pts)

eps = np.max(epsilons[np.where(epsilons > 0)]) + 1e-8 # Deriving epsilon from k-center solution for use in DBSCAN* and USC

# Step 3: Running DBSCAN* using the authors' implementation
# cluster_type = "corepoints" means border points are treated as noise (DBSCAN*)

print("Running DBSCAN*...")
dbscan_model = DcDBSCAN(eps = eps, min_pts = min_pts, cluster_type = "corepoints")
dbscan_model.fit(X)
dbscan_labels = dbscan_model.labels_

# Step 4: Running ultrametric spectral clustering (USC) on the dc-dist similarity matrix

print("Running ultrametric spectral clustering...")
no_lambdas = get_lambdas(root, eps)
sim_matrix = get_sim_mx(dc_dists)

sc_result, sc_labels = run_spectral_clustering(root,
                                               sim_matrix,
                                               dc_dists,
                                               eps = eps,
                                               it = no_lambdas,
                                               min_pts = min_pts,
                                               n_clusters = k,
                                               type_ = "it")

# Printing NMI scores between all three methods (all should be close to 1.0)

print(f"\nNMI k-center vs DBSCAN*: {nmi(pred_labels, dbscan_labels):.4f}")
print(f"NMI k-center vs USC:     {nmi(pred_labels, sc_labels):.4f}")
print(f"NMI DBSCAN*  vs USC:     {nmi(dbscan_labels, sc_labels):.4f}")

# Plotting all four panels side by side (ground truth + three equivalent methods)

fig, axes = plt.subplots(1, 4, figsize = (16, 4))

plot_configs = [(y,             "Ground truth"),
                (pred_labels,   "k-center (dc-dist)"),
                (dbscan_labels, "DBSCAN* (dc-dist)"),
                (sc_labels,     "USC (dc-dist)")]

for ax, (labels, title) in zip(axes, plot_configs):
    ax.scatter(X[:, 0], X[:, 1], c = labels, cmap = "tab10", s = 15, alpha = 0.8)
    ax.set_title(title, fontsize = 10)
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle("Fig 3: Equivalence under dc-dist (two moons)", fontsize = 11)
plt.tight_layout()
plt.savefig("figures/fig3_equivalence.png", dpi = 150, bbox_inches = "tight")
plt.show()

print("Saved figures/fig3_equivalence.png")
