Density-Connectivity Distance (dc-dist) Project Reproduction:

This repository contains a full reimplementation and experimental evaluation of the paper: “Connecting the Dots: Density-Connectivity Distance Unifies DBSCAN, k-Center and Spectral Clustering” [1] The project reproduces key theoretical properties of the density-connectivity distance (dc-dist) and validates the equivalence between DBSCAN*, k-center, and ultrametric spectral clustering (USC). It also extends the original experimental evaluation with additional benchmarking and visualization scripts.

SUMMARY:

The repository is organized into three main parts. The repo/ folder contains the original authors’ implementation and must remain unchanged. The your_code/ folder contains all custom reimplementation scripts numbered from 01 to 10, which are used to generate synthetic data, verify theoretical properties, and reproduce all main experimental results. The data/ folder contains both synthetic datasets (generated automatically) and external datasets such as COIL-100 and Pendigits, which must be downloaded separately. All figures and plots generated from experiments are saved in the figures/ directory.

To run the project, the execution should follow the order of scripts in your_code/, starting from 01_generate_data.py and ending at 10_olivetti_benchmark.py. These scripts collectively reproduce all core experiments including dc-dist verification, clustering equivalence, separability analysis, robustness testing, ARI benchmarking, and real-world evaluation. Some experiments require external datasets such as COIL-100.

Additionally, selected experiments from the original authors’ repository must be executed inside the repo/ directory. These include noise_robustness.py, distances_plot.py, k_vs_epsilon.py, and compare_clustering.py. The outputs of these scripts are saved as figures in the figures/ folder, including repo_fig5 (noise robustness), repo_fig2 (distance separability), repo_fig3 (k vs epsilon), and repo_fig4 (clustering comparison).
