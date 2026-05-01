# CS 432 Project — 04_separability.py (Reproducing Figure 4: intra vs inter-cluster distance distributions)

# INSTRUCTIONS:
# Run after 01_generate_data.py.
# Shows that dc-dist creates a clear "valley" separating intra-cluster
# from inter-cluster distances, unlike Euclidean, cosine, and Manhattan.
# Working directory must be 25280058_Project when running this script.
# Saves figures/fig4_separability.png.
# python 04_separability.py

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

sys.path.insert(0, "repo") # Adding repo to path so all authors' modules are importable

from distance_metric import get_dc_dist_matrix

# Splitting pairwise distances into intra-cluster and inter-cluster sets
# Noise points (label = -1) are excluded from both sets

def split_intra_inter(dist_matrix, labels):

    intra_distances = []
    inter_distances = []
    n = len(labels)

    for i in range(n):
        for j in range(i + 1, n):

            if labels[i] < 0 or labels[j] < 0: # Skipping noise points
                continue

            value = dist_matrix[i, j]

            if labels[i] == labels[j]:
                intra_distances.append(value)
            else:
                inter_distances.append(value)

    return np.array(intra_distances), np.array(inter_distances)

# Plotting one panel of the distribution figure
# Intra-cluster distances shown in blue, inter-cluster in yellow

def plot_one_panel(ax, intra, inter, title):

    max_val = max(intra.max(), inter.max()) + 1e-9
    bin_edges = np.linspace(0, max_val, 21)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 / max_val * 100
    total = len(intra) + len(inter)

    intra_hist, _ = np.histogram(intra, bins = bin_edges)
    inter_hist, _ = np.histogram(inter, bins = bin_edges)

    ax.plot(bin_centers, intra_hist / total, color = "#378ADD", label = "intra-cluster")
    ax.plot(bin_centers, inter_hist / total, color = "#EF9F27", label = "inter-cluster")
    ax.set_title(title, fontsize = 9)
    ax.set_xlabel("% of max distance", fontsize = 8)
    ax.set_ylabel("relative frequency", fontsize = 8)
    ax.legend(fontsize = 7)

# Loading d1: uniform density-based clusters with no noise

X = np.load("data/synthetic/d1_X.npy")
y = np.load("data/synthetic/d1_y.npy")

fig, axes = plt.subplots(2, 3, figsize = (15, 8))

# Computing and plotting standard distance measures

euclidean_matrix = cdist(X, X, metric = "euclidean")
intra, inter = split_intra_inter(euclidean_matrix, y)
plot_one_panel(axes[0, 0], intra, inter, "Euclidean distance")

cosine_matrix = cdist(X, X, metric = "cosine")
intra, inter = split_intra_inter(cosine_matrix, y)
plot_one_panel(axes[1, 0], intra, inter, "Cosine distance")

manhattan_matrix = cdist(X, X, metric = "cityblock")
intra, inter = split_intra_inter(manhattan_matrix, y)
plot_one_panel(axes[1, 1], intra, inter, "Manhattan distance")

# Computing and plotting dc-dist for mu in [3, 5, 10]

target_axes = [axes[0, 1], axes[0, 2], axes[1, 2]] # Mapping mu values to their subplot positions

for panel_index, mu in enumerate([3, 5, 10]):

    dc_matrix = get_dc_dist_matrix(X, n_neighbors = 15, min_points = mu)
    intra, inter = split_intra_inter(dc_matrix, y)
    plot_one_panel(target_axes[panel_index], intra, inter, f"dc-dist (mu = {mu})")

    print(f"mu = {mu}: dc-dist computed, valley visible = {intra.mean() < inter.mean()}")

plt.suptitle("Fig 4: Distance separability: intra vs inter-cluster", fontsize = 11)
plt.tight_layout()
plt.savefig("figures/fig4_separability.png", dpi = 150, bbox_inches = "tight")
plt.show()

print("Saved figures/fig4_separability.png")
