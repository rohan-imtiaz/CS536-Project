# CS 432 Project — 09_ultrametric_tree.py (Visualizing the dc-dist ultrametric tree)

# INSTRUCTIONS:
# Run after 01_generate_data.py.
# Builds the dc-dist ultrametric tree and visualizes it as a dendrogram.
# The dendrogram shows how points merge into clusters as epsilon increases.
# Cutting the dendrogram at a horizontal level gives DBSCAN* clusters for that epsilon.
# Working directory must be 25280058_Project when running this script.
# Saves figures/fig9_ultrametric_tree.png.
# python 09_ultrametric_tree.py

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, "repo") # Adding repo to path so all authors' modules are importable

from distance_metric import get_dc_dist_matrix

# Using a small subset of moons so the dendrogram is readable

X = np.load("data/synthetic/moons_X.npy")[:60]
y = np.load("data/synthetic/moons_y.npy")[:60]

n_points = len(X)

print(f"Computing dc-dist matrices for {n_points} points...")

fig, axes = plt.subplots(1, 3, figsize = (18, 6))

# Plotting one dendrogram per mu value to show how mu changes the tree structure

for plot_index, mu in enumerate([3, 5, 10]):

    dc_matrix = get_dc_dist_matrix(X, n_neighbors = 15, min_points = mu)

    n_unique = len(np.unique(np.round(dc_matrix, 8)))
    print(f"mu = {mu}: unique dc-dist values = {n_unique} (out of {n_points * (n_points - 1) // 2} total pairs)")

    condensed_matrix = squareform(dc_matrix) # Converting to condensed form for scipy linkage

    # Using single linkage because dc-dist is a minimax distance, so single linkage
    # exactly recovers the ultrametric tree structure from the pairwise distances

    linkage_matrix = linkage(condensed_matrix, method = "single")

    # Coloring leaves by their true cluster membership

    leaf_colors = {}
    color_map = {0: "steelblue", 1: "darkorange", -1: "lightgrey"}

    for point_index in range(n_points):
        label = int(y[point_index])
        leaf_colors[point_index] = color_map.get(label, "purple")

    dendrogram(linkage_matrix,
               ax = axes[plot_index],
               color_threshold = 0, # Disabling automatic coloring so we can control it
               above_threshold_color = "grey",
               no_labels = True)

    axes[plot_index].set_title(f"dc-dist tree (mu = {mu})\n{n_unique} unique values out of {n_points * (n_points - 1) // 2} pairs",
                               fontsize = 9)
    axes[plot_index].set_xlabel("Data points (leaves)", fontsize = 8)
    axes[plot_index].set_ylabel("dc-dist (merge height)", fontsize = 8)

    # Drawing a horizontal dashed line at the merge threshold that separates the two moons
    # This line represents the epsilon at which DBSCAN* finds the correct two clusters

    merge_heights = linkage_matrix[:, 2]
    sorted_heights = np.sort(merge_heights)

    if len(sorted_heights) >= 2:
        cluster_gap_threshold = (sorted_heights[-1] + sorted_heights[-2]) / 2 # Midpoint between last two merges
        axes[plot_index].axhline(y = cluster_gap_threshold, color = "red", linestyle = "--",
                                  linewidth = 1.2, alpha = 0.7, label = f"epsilon = {cluster_gap_threshold:.3f}")
        axes[plot_index].legend(fontsize = 7)

plt.suptitle("Fig 9: dc-dist ultrametric tree (dendrogram) for two moons, first 60 points", fontsize = 11)
plt.tight_layout()
plt.savefig("figures/fig9_ultrametric_tree.png", dpi = 150, bbox_inches = "tight")
plt.show()

print("Saved figures/fig9_ultrametric_tree.png")
