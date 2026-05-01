# CS 432 Project — 01_generate_data.py (Synthetic and benchmark dataset generation)

# INSTRUCTIONS:
# Run this file first before any other script in the pipeline.
# Working directory must be 25280058_Project when running this script.
# It generates all synthetic datasets and saves them to data/synthetic/.
# python 01_generate_data.py

import os
import numpy as np
from sklearn.datasets import make_moons, make_circles, fetch_olivetti_faces

# Creating output directories if they do not already exist

os.makedirs("data/synthetic", exist_ok = True)
os.makedirs("figures", exist_ok = True)

print("Output directories confirmed.")

# Seed Spreader style density-based cluster generator (replicates paper Section 6.1)
# Each cluster is produced by a random walk through d-dimensional space

def make_density_clusters(n_per_cluster, n_clusters, n_features,
                           noise_count = 0, vary_density = False, seed = 42):

    local_rng = np.random.RandomState(seed)
    all_points = []
    all_labels = []

    for cluster_id in range(n_clusters):

        center = local_rng.randn(n_features) * 15
        walk_position = center.copy()
        step_size = local_rng.uniform(0.3, 1.0) if vary_density else 0.5 # Varying step size controls cluster density

        for step_index in range(n_per_cluster):
            walk_position = walk_position + local_rng.randn(n_features) * step_size
            all_points.append(walk_position.copy())
            all_labels.append(cluster_id)

    if noise_count > 0: # Adding uniform noise points labelled as -1

        all_pts_array = np.array(all_points)
        noise_points = local_rng.uniform(all_pts_array.min(),
                                         all_pts_array.max(),
                                         size = (noise_count, n_features))
        all_points.extend(noise_points.tolist())
        all_labels.extend([-1] * noise_count)

    return np.array(all_points), np.array(all_labels)

# Generating d1: uniform density, no noise (used in Fig 4 and Fig 6)

d1_X, d1_y = make_density_clusters(100, 5, 2, seed = 42)
print(f"d1 generated: X = {d1_X.shape}, classes = {len(np.unique(d1_y[d1_y >= 0]))}")

# Generating d2: varying density, no noise

d2_X, d2_y = make_density_clusters(100, 5, 2, vary_density = True, seed = 42)
print(f"d2 generated: X = {d2_X.shape}, classes = {len(np.unique(d2_y[d2_y >= 0]))}")

# Generating d3: unbalanced clusters with noise

d3_X, d3_y = make_density_clusters(100, 5, 2, noise_count = 50, seed = 42)
print(f"d3 generated: X = {d3_X.shape}, classes = {len(np.unique(d3_y[d3_y >= 0]))}")

# Generating two moons (used in Fig 3 equivalence and Fig 7)

moons_X, moons_y = make_moons(n_samples = 400, noise = 0.07, random_state = 42)
print(f"moons generated: X = {moons_X.shape}")

# Generating concentric circles (used in Fig 7)

circles_X, circles_y = make_circles(n_samples = 400, noise = 0.05,
                                     factor = 0.5, random_state = 42)
print(f"circles generated: X = {circles_X.shape}")

# Generating four moons: two shifted copies of make_moons (used in Fig 7)

moons_a_X, moons_a_y = make_moons(n_samples = 200, noise = 0.05, random_state = 42)
moons_b_X, moons_b_y = make_moons(n_samples = 200, noise = 0.05, random_state = 99)
moons_b_X = moons_b_X + np.array([4.0, 0.0]) # Shifting second pair rightward to separate them
moons_b_y = moons_b_y + 2 # Offsetting labels so all four classes are distinct

fourmoons_X = np.vstack([moons_a_X, moons_b_X])
fourmoons_y = np.concatenate([moons_a_y, moons_b_y])
print(f"fourmoons generated: X = {fourmoons_X.shape}, classes = {len(np.unique(fourmoons_y))}")

# Loading Olivetti faces directly from sklearn (40 subjects, 400 images, 4096 features)

olivetti_dataset = fetch_olivetti_faces(shuffle = True, random_state = 42)
olivetti_X = olivetti_dataset.data
olivetti_y = olivetti_dataset.target
print(f"olivetti loaded: X = {olivetti_X.shape}, classes = {len(np.unique(olivetti_y))}")

# Saving all datasets to disk as .npy files for consistent loading across all scripts

datasets = {"d1":        (d1_X,        d1_y),
            "d2":        (d2_X,        d2_y),
            "d3":        (d3_X,        d3_y),
            "moons":     (moons_X,     moons_y),
            "circles":   (circles_X,   circles_y),
            "fourmoons": (fourmoons_X,  fourmoons_y),
            "olivetti":  (olivetti_X,   olivetti_y)}

for dataset_name, (X, y) in datasets.items():
    np.save(f"data/synthetic/{dataset_name}_X.npy", X)
    np.save(f"data/synthetic/{dataset_name}_y.npy", y)

print("All datasets saved to data/synthetic/.")
print("Data generation complete.")
