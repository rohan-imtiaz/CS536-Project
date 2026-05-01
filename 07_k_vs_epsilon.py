# CS 432 Project — 07_k_vs_epsilon.py (Reproducing Figure 7: effect of k on epsilon for DBSCAN)

# INSTRUCTIONS:
# Run after 01_generate_data.py.
# Shows that setting k equal to the true number of clusters gives a consistently
# good epsilon value for DBSCAN, eliminating the need for epsilon grid search.
# The red dashed line marks the true k for each dataset.
# Working directory must be 25280058_Project when running this script.
# Saves figures/fig7_k_vs_epsilon.png.
# python 07_k_vs_epsilon.py

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "repo") # Adding repo to path so all authors' modules are importable

from density_tree import make_tree
from cluster_tree import dc_clustering

# Dataset configurations: name -> (X path, y path, true k)

dataset_configs = {"4-moons": ("data/synthetic/fourmoons_X.npy",
                               "data/synthetic/fourmoons_y.npy",
                               4),
                   "circles":  ("data/synthetic/circles_X.npy",
                                "data/synthetic/circles_y.npy",
                                2)}

mu_values = [3, 5, 7]
k_values = list(range(2, 11))

fig, axes = plt.subplots(1, 2, figsize = (12, 5))

for ax, (dataset_name, (x_path, y_path, true_k)) in zip(axes, dataset_configs.items()):

    X = np.load(x_path)
    y = np.load(y_path)

    print(f"Processing dataset: {dataset_name}, true k = {true_k}")

    for mu in mu_values:

        print(f"  mu = {mu}...")
        root, dc_dists = make_tree(X, y, min_points = mu, n_neighbors = 15) # Building the ultrametric tree once per mu

        eps_for_k = []

        for k in k_values:

            pred_labels, centers, epsilons = dc_clustering(root,
                                                           num_points = len(y),
                                                           k = k,
                                                           min_points = mu)

            valid_epsilons = epsilons[epsilons > 0] # Taking the max epsilon across all clusters as the DBSCAN threshold
            eps_val = np.max(valid_epsilons) if len(valid_epsilons) > 0 else 0.0
            eps_for_k.append(eps_val)

        ax.plot(k_values, eps_for_k, marker = "o", markersize = 4, label = f"mu = {mu}")

    ax.axvline(x = true_k, color = "red", linestyle = "--",
               alpha = 0.6, linewidth = 1.5, label = f"true k = {true_k}") # Red line marks the known ground truth number of clusters
    ax.set_xlabel("k")
    ax.set_ylabel("epsilon value for DBSCAN")
    ax.set_title(f"Fig 7: {dataset_name}")
    ax.legend(fontsize = 8)

plt.suptitle("Fig 7: Choosing k gives a good epsilon for DBSCAN", fontsize = 11)
plt.tight_layout()
plt.savefig("figures/fig7_k_vs_epsilon.png", dpi = 150, bbox_inches = "tight")
plt.show()

print("Saved figures/fig7_k_vs_epsilon.png")
