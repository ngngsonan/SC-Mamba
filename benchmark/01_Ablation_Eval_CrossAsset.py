"""
01_Ablation_Eval_CrossAsset.py
==============================
Ablation evaluation script: num_assets=1 (univariate) vs num_assets=8 (cross-asset graph).

Paste each cell into a Colab notebook to reproduce ablation results.
Run after training with both configurations.

Cells:
  [A] Setup & imports
  [B] Evaluation function (wraps validate_on_real_dataset)
  [C] Run ablation: N=1 vs N=8
  [D] Results table + diagnosis
"""

# =============================================================================
# ── CELL [A]: Setup & Imports ─────────────────────────────────────────────────
# =============================================================================
"""
# CELL A — Run once per session
import sys, os
sys.path.insert(0, '/content/SC-Mamba')
os.environ['TRITON_F32_DEFAULT'] = 'ieee'  # Triton compiler workaround

import torch
import numpy as np
import pandas as pd
import yaml
from pprint import pprint

from core.models import SCMamba_Forecaster
from core.real_data_val_pipeline import validate_on_real_dataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_DIR = '/content/drive/MyDrive/Colab Notebooks/SCMamba/sc_mamba_checkpoints'
SCALER   = 'min_max'   # must match training config

def load_model(ckpt_path: str, num_assets: int, ssm_config: dict) -> SCMamba_Forecaster:
    \"\"\"Load SCMamba_Forecaster from checkpoint.\"\"\"
    model = SCMamba_Forecaster(N_assets=num_assets, ssm_config=ssm_config).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    # Support both raw state_dict and wrapped {'model_state_dict': ...}
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f'  ✅ Loaded: {os.path.basename(ckpt_path)} | N_assets={num_assets}')
    return model

# SSM config — must match your training config exactly
SSM_CONFIG = {
    'mamba2'             : True,
    'num_encoder_layers' : 2,
    'd_state'            : 128,
    'headdim'            : 128,
    'block_expansion'    : 2,
    'token_embed_len'    : 1024,
    'chunk_size'         : 256,
    'linear_seq'         : 15,
    'norm'               : True,
    'norm_type'          : 'layernorm',
    'residual'           : False,
    'global_residual'    : False,
    'bidirectional'      : False,
    'in_proj_norm'       : False,
    'enc_conv'           : True,
    'enc_conv_kernel'    : 5,
    'init_dil_conv'      : True,
    'init_conv_kernel'   : 5,
    'init_conv_max_dilation': 3,
    'initial_gelu_flag'  : True,
}
print('Setup complete. DEVICE =', DEVICE)
"""

# =============================================================================
# ── CELL [B]: Evaluation Wrapper ──────────────────────────────────────────────
# =============================================================================
"""
# CELL B — Evaluation function

def evaluate_model(
    model,
    dataset: str,
    scaler: str = 'min_max',
    sub_day: bool = False,
    label: str = 'model',
) -> dict:
    \"\"\"
    Run validate_on_real_dataset and return a dict of metrics.
    Prints a formatted row suitable for comparison table.
    \"\"\"
    print(f'\\n🔍 Evaluating [{label}] on {dataset} ...')
    mase_, mae_, rmse_, smape_, nll_, crps_ = validate_on_real_dataset(
        dataset, model, DEVICE, scaler, subday=sub_day
    )
    result = {
        'label'  : label,
        'dataset': dataset,
        'MASE'   : round(mase_,  4),
        'MAE'    : round(mae_,   4),
        'RMSE'   : round(rmse_,  4),
        'SMAPE'  : round(smape_, 4),
        'NLL'    : round(nll_,   4),   # ↓ better  (probabilistic, unique to SC-Mamba)
        'CRPS'   : round(crps_,  4),   # ↓ better
    }
    print(f'  MASE={mase_:.4f} | MAE={mae_:.4f} | RMSE={rmse_:.4f} | '
          f'SMAPE={smape_:.4f} | NLL={nll_:.4f} | CRPS={crps_:.4f}')
    return result
"""

# =============================================================================
# ── CELL [C]: Run Ablation ────────────────────────────────────────────────────
# =============================================================================
"""
# CELL C — Load both checkpoints and evaluate
#
# Checkpoint naming convention (set by train.py › generate_model_save_name):
#   N=1  → SCMamba_<version_univariate>_best.pth   (trained with num_assets=1)
#   N=8  → SCMamba_v_multivariate_exchange_rate_best.pth (trained with num_assets=8)
#
# Adjust the filenames below to match what you actually trained.

ABLATION_CONFIGS = [
    {
        'label'      : 'N=1 (Univariate)',           # edit version tag as needed
        'ckpt'       : f'{CKPT_DIR}/SCMamba_v1_best.pth',
        'num_assets' : 1,
        'dataset'    : 'exchange_rate',
        'sub_day'    : False,
    },
    {
        'label'      : 'N=8 (Cross-Asset Graph)',
        'ckpt'       : f'{CKPT_DIR}/SCMamba_v_multivariate_exchange_rate_best.pth',
        'num_assets' : 8,
        'dataset'    : 'exchange_rate',
        'sub_day'    : False,
    },
]

results = []
for cfg in ABLATION_CONFIGS:
    print(f'\\n━━━━ {cfg[\"label\"]} ━━━━')
    model = load_model(cfg['ckpt'], cfg['num_assets'], SSM_CONFIG)
    row   = evaluate_model(
        model,
        dataset = cfg['dataset'],
        scaler  = SCALER,
        sub_day = cfg['sub_day'],
        label   = cfg['label'],
    )
    results.append(row)
    del model          # free VRAM between runs
    torch.cuda.empty_cache()
"""

# =============================================================================
# ── CELL [D]: Results Table + Diagnosis ───────────────────────────────────────
# =============================================================================
"""
# CELL D — Pretty results table + interpretation

df = pd.DataFrame(results).set_index('label')
print('\\n' + '='*70)
print('  ABLATION RESULTS: exchange_rate — N=1 vs N=8')
print('='*70)
print(df[['MASE','MAE','RMSE','SMAPE','NLL','CRPS']].to_string())
print('='*70)
print('  ↓ lower is better for all metrics')
print()

# ── Relative improvement (N=8 vs N=1) ────────────────────────────────────────
if len(df) == 2:
    base  = df.iloc[0]   # N=1
    multi = df.iloc[1]   # N=8
    print('  Relative delta (N=8 - N=1) / |N=1|:')
    for m in ['MASE','MAE','RMSE','SMAPE','NLL','CRPS']:
        delta_pct = (multi[m] - base[m]) / (abs(base[m]) + 1e-10) * 100
        arrow = '🟢' if delta_pct < 0 else '🔴'
        print(f'    {arrow}  {m:6s}: {delta_pct:+.1f}%')
    print()

# ── Diagnosis from training log ───────────────────────────────────────────────
print('  📊 TRAINING DIAGNOSTICS (epoch 10):')
diag = {
    'tau (graph temp)' : '2.0005 → NOT learning (stuck at init)',
    'Mask Sparsity'    : '0.00%  → graph dense (no pruning)',
    'sigma2 range'     : '[0.05, 0.08] → near clamp floor (overconfident)',
    'GradNorm'         : '~15–50 → high, consider gradient clipping',
    'NLL trend'        : 'E1→E10: 0.45 → −0.30 → model learning ✅',
    'MSE trend'        : 'E1→E10: 0.0035 → 0.0001 → converging ✅',
}
for k, v in diag.items():
    print(f'    {k:20s}: {v}')
print()
print('  ⚠️  Key concern: tau not learning → Causal Sparsity NOT active.')
print('     -> Consider: lower beta_kl, higher beta_anneal_epochs,')
print('        or explicit tau learning rate (separate param group).')
"""
