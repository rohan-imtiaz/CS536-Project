# CS 432 Project — 05_single_link.py (Reproducing Figure 5: robustness to single-link effect)

# INSTRUCTIONS:
# Run after 01_generate_data.py.
# Tests whether the dc-dist resists the single-link effect as variance (noise) increases.
# Values below zero on the plot mean the clusters can no longer be separated.
# Higher mu should stay positive for longer as variance grows.
# Working directory must be 25280058_Project when running this script.
# Saves figures/fig5_single_link.png.
# python 05_single_link.py

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "repo") # Adding repo to path so all authors' modules are importable

from distance_metric import get_dc_dist_matrix

# Experimental parameters (matching Section 6.3 of the paper)

mu_values = [1, 3, 5, 7, 9]
variances = np.linspace(1.0, 2.0, 10)
n_per_cluster = 100
n_seeds = 10

results = np.zeros((len(mu_values), len(variances))) # Storing mean separation score per (mu, variance) pair

print("Running single-link robustness experiment...")

for var_index, var in enumerate(variances):

    for seed in range(n_seeds):

        rng = np.random.RandomState(seed)
        cluster_a = rng.multivariate_normal([-3, 0], np.eye(2) * var, n_per_cluster)
        cluster_b = rng.multivariate_normal([ 3, 0], np.eye(2) * var, n_per_cluster)
        X = np.vstack([cluster_a, cluster_b])
        true_labels = np.array([0] * n_per_cluster + [1] * n_per_cluster)

        for mu_index, mu in enumerate(mu_values):

            dc_matrix = get_dc_dist_matrix(X, n_neighbors = 15, min_points = mu)

            intra_distances = []
            inter_distances = []
            n = len(X)

            for i in range(n):
                for j in range(i + 1, n):
                    val = dc_matrix[i, j]
                    if true_labels[i] == true_labels[j]:
                        intra_distances.append(val)
                    else:
                        inter_distances.append(val)

            min_inter = np.min(inter_distances)
            mean_intra = np.mean(intra_distances)

            # Positive = clusters are separable, negative = single-link effect has occurred

            results[mu_index, var_index] = results[mu_index, var_index] + (min_inter - mean_intra)

    print(f"  Variance {var:.2f} done.")

results = results / n_seeds # Averaging over seeds

# Plotting separation score vs variance for each value of mu

plt.figure(figsize = (8, 5))

for mu_index, mu in enumerate(mu_values):
    plt.plot(variances, results[mu_index], marker = "o", markersize = 4, label = f"mu = {mu}")

plt.axhline(y = 0, color = "black", linestyle = "--", linewidth = 0.8, alpha = 0.5) # Zero line: below this means single-link failure
plt.xlabel("Diagonal variance")
plt.ylabel("min(inter) - mean(intra)")
plt.title("Fig 5: dc-dist robustness to single-link effect")
plt.legend()
plt.tight_layout()
plt.savefig("figures/fig5_single_link.png", dpi = 150, bbox_inches = "tight")
plt.show()

print("Saved figures/fig5_single_link.png")
