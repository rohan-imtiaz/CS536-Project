# CS 432 Project — 02_verify_dcdist.py (Verifying dc-dist ultrametric properties)

# INSTRUCTIONS:
# Run after 01_generate_data.py.
# Confirms the dc-dist matrix is symmetric, has zero diagonal,
# satisfies the strong triangle inequality (ultrametric), and has O(n) unique values.
# Working directory must be 25280058_Project when running this script.
# python 02_verify_dcdist.py

import sys
import numpy as np

sys.path.insert(0, "repo") # Adding repo to path so all authors' modules are importable

from distance_metric import get_dc_dist_matrix
from density_tree import make_tree

# Loading a small subset of two moons for fast verification

X = np.load("data/synthetic/moons_X.npy")[:100]
y = np.load("data/synthetic/moons_y.npy")[:100]

print("=== dc-dist property verification ===")
print(f"Dataset: moons (first 100 points), shape = {X.shape}\n")

# Checking ultrametric and uniqueness properties across three values of mu

for mu in [3, 5, 10]:

    dc_matrix = get_dc_dist_matrix(X, n_neighbors = 15, min_points = mu) # Computing the dc-dist matrix directly

    n = len(X)
    n_total_pairs = n * (n - 1) // 2
    n_unique_values = len(np.unique(np.round(dc_matrix, 8)))

    # Sampling 5000 random triples to check the strong triangle inequality
    # d(p, r) <= max(d(p, q), d(q, r)) must hold for all triples

    violation_count = 0
    rng = np.random.RandomState(0)

    for _ in range(5000):

        p, q, r = rng.choice(n, 3, replace = False)
        left_side = dc_matrix[p, r]
        right_side = max(dc_matrix[p, q], dc_matrix[q, r])

        if left_side > right_side + 1e-9:
            violation_count = violation_count + 1

    max_asymmetry = np.max(np.abs(dc_matrix - dc_matrix.T)) # Checking symmetry: d(p, q) == d(q, p)
    diagonal_all_zero = np.all(np.diag(dc_matrix) == 0) # Checking diagonal: d(p, p) == 0

    print(f"mu = {mu}:")
    print(f"  Total pairs = {n_total_pairs}, unique dc-dist values = {n_unique_values} (expect ~O(n) = {n})")
    print(f"  Ultrametric violations from 5000 random triples: {violation_count}")
    print(f"  Max asymmetry: {max_asymmetry:.2e}")
    print(f"  Diagonal all zero: {diagonal_all_zero}\n")

# Confirming that make_tree and get_dc_dist_matrix produce identical matrices
# make_tree calls get_dc_dist_matrix internally, so both paths must match exactly

root, dc_dists_from_tree = make_tree(X, y, min_points = 5, n_neighbors = 15)
dc_dists_direct = get_dc_dist_matrix(X, n_neighbors = 15, min_points = 5)

print(f"make_tree vs direct call match: {np.allclose(dc_dists_from_tree, dc_dists_direct)}")
print("Verification complete.")
