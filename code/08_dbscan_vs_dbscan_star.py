# CS 432 Project — 08_dbscan_vs_dbscan_star.py (Comparing full DBSCAN to DBSCAN*)

# INSTRUCTIONS:
# Run after 01_generate_data.py and 03_equivalence.py.
# Demonstrates the difference between DBSCAN* (core points only) and full DBSCAN
# (core points plus border points), corresponding to Section 3.2 of the paper.
# The DBSCAN-distance extends dc-dist to handle border points as well.
# Working directory must be 25280058_Project when running this script.
# Saves figures/fig8_dbscan_vs_dbscan_star.png.
# python 08_dbscan_vs_dbscan_star.py

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "repo") # Adding repo to path so all authors' modules are importable

from density_tree import make_tree
from cluster_tree import dc_clustering
from DBSCAN import DBSCAN as DcDBSCAN
from sklearn.metrics import normalized_mutual_info_score as nmi

# Loading two moons dataset for a clear visual comparison

X = np.load("data/synthetic/moons_X.npy")
y = np.load("data/synthetic/moons_y.npy")

min_pts = 5
k = 2
n_neighbors = 15

print("Building density tree...")
root, dc_dists = make_tree(X, y, min_points = min_pts, n_neighbors = n_neighbors)

# Running k-center to get the epsilon that both DBSCAN variants will use

pred_labels, centers, epsilons = dc_clustering(root,
                                               num_points = len(y),
                                               k = k,
                                               min_points = min_pts)

eps = np.max(epsilons[np.where(epsilons > 0)]) + 1e-8 # Deriving epsilon from k-center solution

# Running DBSCAN* (cluster_type = "corepoints"): border points are treated as noise

print("Running DBSCAN* (corepoints only)...")
dbscan_star_model = DcDBSCAN(eps = eps, min_pts = min_pts, cluster_type = "corepoints")
dbscan_star_model.fit(X)
dbscan_star_labels = dbscan_star_model.labels_

# Running full DBSCAN (cluster_type = "all"): border points are assigned to nearest core cluster

print("Running full DBSCAN (core + border points)...")
dbscan_full_model = DcDBSCAN(eps = eps, min_pts = min_pts, cluster_type = "standard")
dbscan_full_model.fit(X)
dbscan_full_labels = dbscan_full_model.labels_

# Counting how many points each version assigns as noise (label = -1)

n_noise_star = int(np.sum(dbscan_star_labels == -1))
n_noise_full = int(np.sum(dbscan_full_labels == -1))
n_total = len(X)

print(f"\nDBSCAN* noise points: {n_noise_star} out of {n_total} ({round(n_noise_star / n_total * 100, 1)}%)")
print(f"Full DBSCAN noise points: {n_noise_full} out of {n_total} ({round(n_noise_full / n_total * 100, 1)}%)")
print(f"Border points recovered by full DBSCAN: {n_noise_star - n_noise_full}")

# Computing NMI of each method against ground truth (noise points excluded from NMI)

mask_star = dbscan_star_labels != -1
mask_full = dbscan_full_labels != -1
mask_both = mask_star & mask_full

nmi_star = nmi(y[mask_star], dbscan_star_labels[mask_star])
nmi_full = nmi(y[mask_full], dbscan_full_labels[mask_full])

print(f"\nNMI DBSCAN* vs ground truth (on non-noise points): {nmi_star:.4f}")
print(f"NMI full DBSCAN vs ground truth (on non-noise points): {nmi_full:.4f}")

# Plotting three panels: DBSCAN*, full DBSCAN, and a difference map showing recovered border points

fig, axes = plt.subplots(1, 3, figsize = (18, 5))

# Panel 1: DBSCAN* (noise points shown in grey)

noise_mask_star = dbscan_star_labels == -1
axes[0].scatter(X[~noise_mask_star, 0], X[~noise_mask_star, 1],
                c = dbscan_star_labels[~noise_mask_star], cmap = "tab10",
                s = 15, alpha = 0.8, label = "cluster point")
axes[0].scatter(X[noise_mask_star, 0], X[noise_mask_star, 1],
                c = "lightgrey", s = 15, alpha = 0.6, label = "noise")
axes[0].set_title(f"DBSCAN* (noise = {n_noise_star})", fontsize = 10)
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].legend(fontsize = 8)

# Panel 2: Full DBSCAN (fewer noise points because border points are absorbed)

noise_mask_full = dbscan_full_labels == -1
axes[1].scatter(X[~noise_mask_full, 0], X[~noise_mask_full, 1],
                c = dbscan_full_labels[~noise_mask_full], cmap = "tab10",
                s = 15, alpha = 0.8, label = "cluster point")
axes[1].scatter(X[noise_mask_full, 0], X[noise_mask_full, 1],
                c = "lightgrey", s = 15, alpha = 0.6, label = "noise")
axes[1].set_title(f"Full DBSCAN (noise = {n_noise_full})", fontsize = 10)
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].legend(fontsize = 8)

# Panel 3: Highlighting the border points that full DBSCAN recovered from DBSCAN*'s noise

recovered_border = noise_mask_star & ~noise_mask_full # Points that DBSCAN* called noise but full DBSCAN assigned to a cluster
axes[2].scatter(X[~noise_mask_full & ~recovered_border, 0],
                X[~noise_mask_full & ~recovered_border, 1],
                c = "steelblue", s = 15, alpha = 0.5, label = "core cluster points")
axes[2].scatter(X[recovered_border, 0], X[recovered_border, 1],
                c = "orange", s = 30, alpha = 0.9, label = "recovered border points")
axes[2].scatter(X[noise_mask_full, 0], X[noise_mask_full, 1],
                c = "lightgrey", s = 15, alpha = 0.4, label = "still noise")
axes[2].set_title(f"Border points recovered: {int(np.sum(recovered_border))}", fontsize = 10)
axes[2].set_xticks([])
axes[2].set_yticks([])
axes[2].legend(fontsize = 8)

plt.suptitle("Fig 8: DBSCAN* vs full DBSCAN (border point extension)", fontsize = 11)
plt.tight_layout()
plt.savefig("figures/fig8_dbscan_vs_dbscan_star.png", dpi = 150, bbox_inches = "tight")
plt.show()

print("Saved figures/fig8_dbscan_vs_dbscan_star.png")
