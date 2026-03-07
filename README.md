# SC-Mamba: Spectral Causal Mamba GNN

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Rank-A Ready](https://img.shields.io/badge/Target-IndabaX/ICLR-brightgreen.svg)

SC-Mamba represents a novel intersection between **Continuous Prior Causal Graph Extraction** and **Channel-Independent (CI) State Space Models**. Built initially upon the efficient temporal scaling logic of Mamba4Cast, SC-Mamba enforces a mathematical constraint (Phase 1) natively deriving Graphical Information Bottlenecks over financial time series (Phase 2).

## 🔥 Key Contributions

1. **Latent Space Extraction over Channel-Independence:** Instead of flattening variables implicitly, SC-Mamba explicitly maintains a separated sequence matrix extracting temporal states $\mathbf{Z} \in \mathbb{R}^{B \times N \times P_L \times D_{model}}$.
2. **Spectral Variational Inference (Continuous Prior):** Through 1D Fast Fourier Transforms (`torch.fft.rfft`), we map dependencies to the frequency domain natively bypassing standard discrete $A$ adjacency bottleneck limits. 
3. **Implicit Graph Identification:** Employs a learnable high-frequency mask/threshold ($\tau$). The resultant distribution computes a direct KL divergence against the posterior to enforce a structurally sparse, causal filter map (the "Spectral ELBO").
4. **Stochastic Output Scaling:** Outputs Multivariate Normal distributions $(\mu, \sigma^2)$ predicting not just point estimates (MASE) but modeling market shocks scaling via Negative Log-Likelihood (NLL).

---

## 📂 Repository Structure

The architecture is tightly controlled and refactored for rapid zero-shot benchmarking in `PyTorch`. 

```text
SC-Mamba/
├── core/                                # PyTorch Mathematical Core Models
│   ├── models.py                        # Central execution wrapping SCMamba_Forecaster & SpectralVariationalLayer
│   ├── train.py                         # Training pipeline strictly optimizing Spectral ELBO (NLL + KL Loss)
│   ├── eval_real_dataset.py             # Analytical inference calculating CRPS, NLL, and Point metrics (MASE/SMAPE)
│   ├── blocks.py                        # S4 Convolutional / Mamba encoding logic
│   └── utils.py 
├── data/                                # Modular Datasets Hub
│   ├── data_provider/                   # GluonTS Sequence conversion & loading configurations
│   └── scripts/
│       ├── generate_causal_series.py    # Generates Ground-Truth Causal Time-Series matrix (A_true)
│       └── store_real_datasets.py       # API Hook utilizing `yfinance` to bypass package limits & cache NASDAQ-100 sequences (pkl).
├── insights/                            # Mathematical Verification Tools
│   ├── Insights_Explainability.ipynb    # Visualizes the learned \tau threshold mapping it natively onto a Spatial Matrix Graph
│   └── Insights_Certainty.ipynb         # Plotted Analysis overlaying predicted risk limits (\pm2\sigma) over historical flash crashes.
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Installation

Ensure you have a modern GPU configured for `mamba_ssm` acceleration. Install dependencies via pip:

```bash
pip install torch mamba-ssm gluonts yfinance pydantic wandb
```

### 2. Loading the Data Pipelines

SC-Mamba tests extreme correlation via synthetic ground truths and actual market conditions. You must cache the `.pkl` files before training:

```bash
# 1. To pull the Top 20 NASDAQ stocks natively
python data/scripts/store_real_datasets.py --datasets nasdaq

# 2. To generate Synthetic Graphs strictly confirming to a static Adjacency Matrix
python data/scripts/generate_causal_series.py
```

*Note: Caching converts sequential DataFrames directly into GluonTS `ListDataset` formats matching `data_factory.py`.*

### 3. Model Training & Validation

Execute the probabilistic training loop using the custom configuration files. The model natively outputs metrics tracked over standard `MSE` and Probabilistic `NLL` losses.

```bash
cd core/
python train.py --dataset nasdaq --batch_size 16 --num_assets 20 --num_epochs 100
```
*Metrics are tracked natively inside the Weights & Biases UI.*

### 4. Evaluating Analytical Explanations

To extract the exact "why" parameters for your technical paper reviewers, reference the Jupyter Notebooks stored under `/insights/`. 
- Pass an identity vector to the SCMamba `SpectralVariationalLayer` to recover and plot the learned probability interaction matrix natively inside `Insights_Explainability`.

---

## 📜 Authors & Acknowledgments
Designed as part of the core Research Implementation track targeting Rank A Conferences (IndabaX / ICLR). Base sequential layers inspired by Mamba4Cast template mapping.
