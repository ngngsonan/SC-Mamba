"""
Module to generate Causal Synthetic Dataset for SC-Mamba pre_training.
Built on top of Mamba4Cast synthetic generation but enforces a Ground-Truth Adjacency Matrix.
"""

import numpy as np
import pandas as pd
from datetime import date
from typing import List
import tqdm
import pickle
import os

from synthetic_generation.tf_generate_series import generate_single_sample
from synthetic_generation.constants import CONTEXT_LENGTH

def generate_causal_time_series(N_assets=10, size=CONTEXT_LENGTH, freq: str = 'daily', 
                                sparsity=0.2, lag=1, options: dict = {}):
    """
    Generate coupled time series data where N_assets interact based on a 
    Ground Truth Adjacency Matrix (A_true).
    
    Args:
        N_assets: Number of interacting series (Nodes in the graph)
        size: Length of the time series
        sparsity: Probability of an edge existing in A_true
        lag: How many time steps it takes for a causal effect to manifest
    """
    # 1. Generate N independent base series using Mamba4Cast engine
    print(f"Generating {N_assets} base independent series...")
    base_series = []
    for _ in tqdm.tqdm(range(N_assets)):
        # Generate independent sample
        sample = generate_single_sample(size=size, freq=freq, transition=True, options=options)
        base_series.append(sample)
        
    # Extract the actual Y values to a matrix [N_assets, Time]
    # Mamba4Cast return format from generate_single_sample is a dictionary:
    # {"id": ..., "ts": ..., "y": ..., "noise": ...}
    Y_base = np.stack([s["y"] * s["noise"] for s in base_series])
    
    # 2. Define the Ground-Truth Causal Adjacency Matrix (A_true)
    # A_true[i, j] = 1 means Series J causally impacts Series I
    # We enforce no self-loops and keep it upper triangular to form a Directed Acyclic Graph (DAG) for stability
    A_true = np.zeros((N_assets, N_assets))
    for i in range(N_assets):
        for j in range(i + 1, N_assets):
            if np.random.rand() < sparsity:
                A_true[i, j] = np.random.uniform(0.3, 0.8) * np.random.choice([-1, 1]) # Edge weight

    print(f"Ground Truth Adjacency Matrix Sparsity: {np.count_nonzero(A_true) / (N_assets * N_assets - N_assets):.2%}")
    
    # 3. Apply Causal Coupling
    # Y_coupled[t] = Y_base[t] + A_true @ Y_coupled[t-lag]
    Y_coupled = np.copy(Y_base)
    
    print("Applying causal recursive coupling...")
    for t in range(lag, size):
        # The value of all assets at time t is influenced by their neighbors at t-lag
        causal_effect = A_true @ Y_coupled[:, t - lag]
        Y_coupled[:, t] += causal_effect
        
    # 4. Repackage into Mamba4Cast dictionary format
    coupled_dataset = []
    for i in range(N_assets):
        new_sample = {
            "id": base_series[i]["id"] + f"_Node{i}",
            "ts": base_series[i]["ts"],
            "y": Y_coupled[i, :], # The new coupled deterministic + noise signal
            "noise": base_series[i]["noise"], # Originally generated noise profile
            "node_idx": i
        }
        coupled_dataset.append(new_sample)
        
    return coupled_dataset, A_true

def main():
    # Configuration
    N_assets = 100
    size = 1000
    freq = 'daily'
    save_dir = "../data/synthetic_causal/"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"--- Generating SC-Mamba Causal Graph Synthetic Data ---")
    dataset, A_true = generate_causal_time_series(N_assets=N_assets, size=size, freq=freq)
    
    # Save the Dataset (Matches Mamba4Cast List[Dict] format)
    dataset_path = os.path.join(save_dir, f"causal_{freq}_{N_assets}nodes.pkl")
    with open(dataset_path, "wb") as f:
        pickle.dump(dataset, f)
        
    # Save the Ground Truth Matrix for Evaluation (Explainability Insights)
    gt_path = os.path.join(save_dir, f"A_true_{freq}_{N_assets}nodes.npy")
    np.save(gt_path, A_true)
    
    print(f"Saved coupled dataset to {dataset_path}")
    print(f"Saved Ground-Truth Adjacency Matrix to {gt_path}")

if __name__ == "__main__":
    main()
