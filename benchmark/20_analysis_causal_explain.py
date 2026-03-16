import torch, os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from core.models import SCMamba_Forecaster
from data.data_provider.multivariate_loader import MultivariateRealDataset

# --- Configuration ---
CKPT_MULTI = '/content/drive/MyDrive/Colab Notebooks/SCMamba/sc_mamba_checkpoints/SCMamba_v2_multi_exchange_rate_best_mase.pth'
CKPT_UNI = '/content/drive/MyDrive/Colab Notebooks/SCMamba/sc_mamba_checkpoints/SCMamba_v2_uni_exchange_rate_best_mase.pth'
CURRENCY_LABELS = ['AUD','GBP','CAD','CHF','CNY','JPY','NZD','SGD']

# Fallback for Local Mac testing
if not os.path.exists(os.path.dirname(CKPT_MULTI)):
    CKPT_MULTI = './checkpoints/SCMamba_v2_multi_exchange_rate_best_mase.pth'
    CKPT_UNI = './checkpoints/SCMamba_v2_uni_exchange_rate_best_mase.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def robust_load_scmamba(ckpt_path, device, force_n_assets=None):
    """Loads model with dynamic patching for older checkpoints."""
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    ssm_config = ckpt.get('ssm_config', {})

    if 'num_encoder_layers' not in ssm_config:
        layers = set([int(k.split('.')[k.split('.').index('mamba_encoder_layers')+1]) for k in state_dict.keys() if 'mamba_encoder_layers.' in k])
        if layers: ssm_config['num_encoder_layers'] = max(layers) + 1

    N_total = force_n_assets if force_n_assets else ckpt.get('N_assets', len(CURRENCY_LABELS))
    model = SCMamba_Forecaster(N_assets=N_total, ssm_config=ssm_config).to(device)
    model.load_state_dict(state_dict, strict=False) # Allow minor state mismatches for old N=1 ckpts
    return model.eval()

def extract_adjacency(model):
    """Extracts the learned Adjacency Matrix A from the SpectralVariationalGraph"""
    if model.N_assets == 1:
        # Univariate models have no cross-talk. Mathematically it is an Identity Matrix.
        return torch.eye(len(CURRENCY_LABELS))
        
    tau_val = model.spectral_layer.tau.detach().item()
    D_model = model.spectral_layer.d_model

    impulse = torch.eye(model.N_assets, D_model).unsqueeze(1).expand(model.N_assets, 16, D_model).to(device)

    with torch.no_grad():
        filtered, _, _ = model.spectral_layer(impulse, model.N_assets)

    cross = filtered.mean(dim=(1, 2)).cpu()
    A = cross.unsqueeze(0) * cross.unsqueeze(1)
    A = (A - A.min()) / (A.max() - A.min() + 1e-8)
    A = (A + A.T) / 2
    return A, tau_val

# ==========================================
# PART 1: CAUSAL EXPLAINABILITY (HEATMAPS)
# ==========================================
try:
    print(f"Loading Models for Causal Analysis...")
    model_multi = robust_load_scmamba(CKPT_MULTI, device)
    model_uni = robust_load_scmamba(CKPT_UNI, device, force_n_assets=1)
    
    A_multi, tau_multi = extract_adjacency(model_multi)
    # Replicate N=1 behavior to an 8x8 matrix for visual comparison (it's essentially a block diagonal or identity in a true N=1 loop)
    A_uni = torch.eye(model_multi.N_assets) 
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(A_uni.numpy(), cmap='Blues', vmin=0, vmax=1, annot=True, fmt='.2f',
                xticklabels=CURRENCY_LABELS[:model_multi.N_assets], yticklabels=CURRENCY_LABELS[:model_multi.N_assets], ax=axes[0])
    axes[0].set_title("Univariate / Siloed (N=1)\nNo Cross-Asset Causality")
    
    sns.heatmap(A_multi.numpy(), cmap='magma', vmin=0, vmax=1, annot=True, fmt='.2f',
                xticklabels=CURRENCY_LABELS[:model_multi.N_assets], yticklabels=CURRENCY_LABELS[:model_multi.N_assets], ax=axes[1])
    axes[1].set_title(f"SC-Mamba Multivariate Graph (N=8)\nLearned Causal Connections ($\\tau$={tau_multi:.2f})")
    
    plt.tight_layout()
    plt.savefig('20_causal_heatmap_comparison.png', dpi=300)
    print("✅ Causal Heatmap saved to '20_causal_heatmap_comparison.png'")
    
except Exception as e:
    print(f"❌ Causal Analysis Error: {e}")

# ==========================================
# PART 2: PREDICTIVE CERTAINTY (EPISTEMIC VARIANCE)
# ==========================================
try:
    print(f"\nAnalyzing Epistemic Variance (Predictive Certainty)...")
    # Load 1 batch of real data for demonstration
    dataset = MultivariateRealDataset(root_path='./dataset/exchange_rate', data_path='exchange_rate.csv', size=[256, 0, 96], scale=True)
    x_batch, y_batch, x_mark, y_mark = dataset[len(dataset)-1]
    
    x_tensor = torch.tensor(x_batch).unsqueeze(0).to(device) # [1, 256, 8]
    
    # Target Asset: 0 (AUD)
    target_idx = 0
    target_name = CURRENCY_LABELS[target_idx]
    
    with torch.no_grad():
        # Multivariate Prediction
        mu_multi, sigma_multi = model_multi.predict(x_tensor)
        mu_m = mu_multi[0, :, target_idx].cpu().numpy()
        sig_m = sigma_multi[0, :, target_idx].cpu().numpy()
        
        # Univariate Prediction (Requires looping or isolating the channel)
        # We pass only the target channel, but model expects features
        x_uni = x_tensor[:, :, target_idx:target_idx+1]
        mu_uni_pred, sigma_uni_pred = model_uni.predict(x_uni)
        mu_u = mu_uni_pred[0, :, 0].cpu().numpy()
        sig_u = sigma_uni_pred[0, :, 0].cpu().numpy()
        
    y_true = y_batch[:, target_idx]
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Ground Truth', color='black', linewidth=2)
    
    # Plot N=1
    plt.plot(mu_u, label='N=1 (Univariate) Mean', color='blue', linestyle='--')
    plt.fill_between(range(96), mu_u - 2*np.sqrt(sig_u), mu_u + 2*np.sqrt(sig_u), color='blue', alpha=0.1, label='N=1 Certainty ($\pm 2\\sigma$)')
    
    # Plot N>1
    plt.plot(mu_m, label='N=8 (SC-Mamba) Mean', color='red')
    plt.fill_between(range(96), mu_m - 2*np.sqrt(sig_m), mu_m + 2*np.sqrt(sig_m), color='red', alpha=0.3, label='N=8 Certainty ($\pm 2\\sigma$)')
    
    plt.title(f"Predictive Certainty Comparison on {target_name} (Exchange Rate)")
    plt.xlabel("Forecast Horizon (Steps)")
    plt.ylabel("Scaled Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig('20_predictive_certainty_comparison.png', dpi=300)
    print("✅ Predictive Certainty graph saved to '20_predictive_certainty_comparison.png'")

except Exception as e:
    print(f"❌ Certainty Analysis Error: {e}")
