"""
generate_causal_series.py
─────────────────────────
Coupled-SDE synthetic data generator for SC-Mamba Phase 3 ground-truth causality
verification.

Mathematical formulation
────────────────────────
Given a sparse Ground-Truth Adjacency Matrix A_true ∈ {0,1}^{N×N} (upper-triangular,
user-defined), each asset series Y_i[t] is generated as:

    Y_i[t] = Y_base_i[t]  +  Σ_{j: A_true[j,i]=1}  coupling_weight[j,i] * Y_j[t - lag]  +  ε_i[t]

where:
 - Y_base_i[t] is an independent AR/GP series (from existing generate_series.py).
 - lag ∈ {1, …, max_lag}: discrete causal delay from cause j → effect i.
 - coupling_weight[j,i] ~ Uniform(0.2, 0.8) so the causal signal is non-trivial.
 - ε_i[t] ~ N(0, noise_std²): residual i.i.d. innovation noise.

This construction embeds a *known causal structure* into synthetic data, enabling
hard quantitative evaluation of SC-Mamba's learned spectral filter:
    Precision / Recall / F1 of A_learned vs A_true
    Hamming distance: |{(i,j) : sign(A_learned[i,j]) ≠ A_true[i,j]}|

Usage
─────
from synthetic_generation.generate_causal_series import generate_causal_batch

batch_ts, A_true = generate_causal_batch(
    N=100, seq_len=256, max_lag=3, sparsity=0.05, seed=42
)
# batch_ts: np.ndarray [N, seq_len]  (row i = asset i)
# A_true:   np.ndarray [N, N]        (A_true[j, i]=1 means j → i)
"""

import numpy as np
from typing import Optional, Tuple

from synthetic_generation.generate_series import generate as generate_independent_series


def build_random_adjacency(
    N: int,
    sparsity: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Construct a random sparse *upper-triangular* causal adjacency matrix.

    Upper-triangular ensures a DAG (no feedback cycles by construction),
    which is the standard assumption in causal time-series literature
    (Granger causality, PCMCI).

    Parameters
    ----------
    N        : Number of assets / time series.
    sparsity : Expected fraction of non-zero edges.  E.g. 0.05 → ~5% of entries
               in the upper triangle are 1, giving an average in-degree of
               (N-1)*sparsity ≈ 5 connections per node for N=100.
    rng      : Optional numpy random Generator for reproducibility.

    Returns
    -------
    A_true : np.ndarray, shape [N, N], dtype float32.
             A_true[j, i] = 1.0  ⟺  series j causally influences series i.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Draw each upper-triangular entry independently from Bernoulli(sparsity)
    A = (rng.random((N, N)) < sparsity).astype(np.float32)
    # Enforce DAG structure: zero out diagonal and lower triangle
    A = np.triu(A, k=1)
    return A


def generate_causal_batch(
    N: int = 100,
    seq_len: int = 256,
    max_lag: int = 3,
    sparsity: float = 0.05,
    noise_std: float = 0.05,
    coupling_low: float = 0.2,
    coupling_high: float = 0.6,
    A_true: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    freq: str = "daily",
    options: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a batch of N causally-coupled time series with a known ground-truth
    causal graph A_true.

    Parameters
    ----------
    N            : Number of assets / series.
    seq_len      : Total sequence length (history + prediction horizon).
    max_lag      : Maximum causal lag (steps). Each edge (j→i) is assigned a
                   randomly drawn lag in [1, max_lag].
    sparsity     : Fraction of edges in the upper-triangular DAG (used only if
                   A_true is not provided).
    noise_std    : Std-dev of i.i.d. Gaussian innovation ε_i[t] added to each
                   series *after* causal injection, controlling the SNR.
    coupling_low : Lower bound of Uniform coupling weight draw.
    coupling_high: Upper bound of Uniform coupling weight draw.
    A_true       : Optional pre-defined adjacency matrix [N, N].  If None, a
                   random sparse DAG is constructed via `build_random_adjacency`.
    seed         : Random seed for full reproducibility.
    freq         : Frequency of the base independent series ('daily', 'weekly', …).
    options      : dict passed to `generate_independent_series`.

    Returns
    -------
    Y_coupled : np.ndarray, shape [N, seq_len], dtype float32.
                Row i is the causally-coupled time series for asset i.
    A_true    : np.ndarray, shape [N, N], dtype float32.
                The ground-truth adjacency (same object if provided).

    Notes on Design
    ───────────────
    - The independent base series Y_base are generated first (each of length
      seq_len + max_lag to allow lag lookback), then coupling is injected in a
      single forward pass over t=max_lag … seq_len+max_lag-1.
    - Coupling weights W[j,i] and lags L[j,i] are drawn randomly but stored so
      the caller can retrieve them for controlled ablations.
    - Edge (j→i) contributes `W[j,i] * Y_base_j[t - L[j,i]]` to Y_i[t].
      Using *base* series (not coupled) as source for the causal signal avoids
      confounded feedback loops in the injection step.
    """
    rng = np.random.default_rng(seed)

    if options is None:
        options = {
            "trend_exp": False,
            "scale_noise": [0.05, 0.1],
        }

    # ── Step 1: Generate N independent base series of length (seq_len + max_lag)
    extended_len = seq_len + max_lag
    Y_base = np.zeros((N, extended_len), dtype=np.float32)
    for i in range(N):
        try:
            _, series_df = generate_independent_series(
                n=extended_len, freq=freq, options=options, transition=False
            )
            vals = series_df["series_values"].values.astype(np.float32)
            # Normalize each base series to zero-mean, unit-std to keep magnitudes
            # comparable across assets (prevents dominant-series coupling artifacts).
            std = vals.std()
            if std > 1e-6:
                vals = (vals - vals.mean()) / std
            Y_base[i] = vals
        except Exception:
            # Fallback: standard Gaussian white noise if series generation fails
            Y_base[i] = rng.standard_normal(extended_len).astype(np.float32)

    # ── Step 2: Build ground-truth adjacency if not provided
    if A_true is None:
        A_true = build_random_adjacency(N, sparsity=sparsity, rng=rng)

    # ── Step 3: Draw coupling weights and lags for all active edges
    # W[j,i] ∈ [coupling_low, coupling_high] for edges where A_true[j,i]=1
    W = rng.uniform(coupling_low, coupling_high, size=(N, N)).astype(np.float32)
    W = W * A_true  # zero out non-edges

    # L[j,i] ∈ {1, ..., max_lag}: causal delay from j to i
    L = rng.integers(1, max_lag + 1, size=(N, N))

    # ── Step 4: Inject causal coupling — forward pass over time
    # Y[i, t] = Y_base[i, t] + Σ_{j} W[j,i] * Y_base[j, t - L[j,i]] + ε
    # Using Y_base (not Y_coupled) as the causal source to ensure the designed
    # causal graph is recoverable without confounding cycles.
    Y_coupled = Y_base.copy()  # starts identical to base
    noise = (rng.standard_normal((N, extended_len)) * noise_std).astype(np.float32)

    for t in range(max_lag, extended_len):
        for i in range(N):
            for j in range(N):
                if A_true[j, i] > 0:
                    lag = L[j, i]
                    Y_coupled[i, t] += W[j, i] * Y_base[j, t - lag]
        Y_coupled[:, t] += noise[:, t]

    # Return only the valid portion (discard the max_lag warm-up)
    return Y_coupled[:, max_lag:], A_true


def generate_causal_batch_fast(
    N: int = 100,
    seq_len: int = 256,
    max_lag: int = 3,
    sparsity: float = 0.05,
    noise_std: float = 0.05,
    coupling_low: float = 0.2,
    coupling_high: float = 0.6,
    A_true: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorised (NumPy-only) variant of `generate_causal_batch`.

    Replaces the Python triple-loop with a single vectorised operation over
    the lag dimension — O(N² · seq_len) but with array-level parallelism.

    Use this for large N (>200) or repeated synthetic pre-training calls.

    Trade-off: Uses Gaussian random walks as base series (instead of the full
    FP/GP generator) to avoid per-series Python overhead.
    """
    rng = np.random.default_rng(seed)
    extended_len = seq_len + max_lag

    # Gaussian random walk as lightweight base
    innovations = rng.standard_normal((N, extended_len)).astype(np.float32)
    Y_base = np.cumsum(innovations, axis=1)
    # Standardise
    std = Y_base.std(axis=1, keepdims=True).clip(min=1e-6)
    Y_base = Y_base / std

    if A_true is None:
        A_true = build_random_adjacency(N, sparsity=sparsity, rng=rng)

    W = (rng.uniform(coupling_low, coupling_high, size=(N, N)) * A_true).astype(np.float32)
    L = rng.integers(1, max_lag + 1, size=(N, N))  # shape [N, N]

    Y_coupled = Y_base.copy()
    noise = (rng.standard_normal((N, extended_len)) * noise_std).astype(np.float32)

    # Vectorised lag injection
    # For each unique lag value l, find all (j,i) pairs with L[j,i]==l and inject
    for l in range(1, max_lag + 1):
        edge_mask = (A_true > 0) & (L == l)  # [N, N] boolean
        if not edge_mask.any():
            continue
        js, is_ = np.where(edge_mask)
        w_vals = W[js, is_]  # shape [num_edges]
        # Y_base[js, max_lag : seq_len+max_lag] shifted by -l
        src = Y_base[js, max_lag - l: seq_len + max_lag - l]  # [num_edges, seq_len]
        # Accumulate into Y_coupled for each destination asset i
        np.add.at(Y_coupled, (is_, slice(max_lag, seq_len + max_lag)),
                  w_vals[:, None] * src)

    Y_coupled[:, max_lag:] += noise[:, max_lag:]
    return Y_coupled[:, max_lag:], A_true
