# @title 11_eval_ckp_crossAsset
"""
11_eval_ckp_crossAsset.py
==============================
Ablation evaluation script: num_assets=1 (univariate) vs num_assets=8 (cross-asset graph).

Run directly via terminal:
python benchmark/01_Ckp_Eval_CrossAsset.py
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
import yaml
from pprint import pprint

# Setup paths and environment
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.environ['TRITON_F32_DEFAULT'] = 'ieee'

from core.models import SCMamba_Forecaster
from core.real_data_val_pipeline import validate_on_real_dataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCALER   = 'min_max'
CKPT_DIR = '/content/drive/MyDrive/Colab Notebooks/SCMamba/sc_mamba_checkpoints' #os.path.join(PROJECT_ROOT, 'checkpoints')

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


def load_model(ckpt_path: str, num_assets: int, ssm_config: dict) -> SCMamba_Forecaster:
    """Load SCMamba_Forecaster from checkpoint."""
    model = SCMamba_Forecaster(N_assets=num_assets, ssm_config=ssm_config).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    # Support both raw state_dict and wrapped {'model_state_dict': ...}
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f'  ✅ Loaded: {os.path.basename(ckpt_path)} | N_assets={num_assets}')
    return model


def evaluate_model(
    model,
    dataset: str,
    scaler: str = 'min_max',
    sub_day: bool = False,
    label: str = 'model',
) -> dict:
    """
    Run validate_on_real_dataset and return a dict of metrics.
    Prints a formatted row suitable for comparison table.
    """
    print(f'\n🔍 Evaluating [{label}] on {dataset} ...')
    mase_, mae_, rmse_, smape_, nll_, crps_, mcrps_ = validate_on_real_dataset(
        dataset, model, DEVICE, scaler, subday=sub_day
    )
    result = {
        'label'  : label,
        'dataset': dataset,
        'MASE'   : round(mase_,  4),
        'MAE'    : round(mae_,   4),
        'RMSE'   : round(rmse_,  4),
        'SMAPE'  : round(smape_, 4),
        'NLL'    : round(nll_,   4),
        'mCRPS'  : round(mcrps_, 4),
    }
    mase_val = result.get("MASE")
    mcrps_val = result.get("mCRPS")

    mase_str = f"{mase_val:.4f}" if isinstance(mase_val, (int, float)) else "?"
    mcrps_str = f"{mcrps_val:.4f}" if isinstance(mcrps_val, (int, float)) else "?"

    print(f'  MASE={mase_str} | MAE={result.get("MAE","?")} | RMSE={result.get("RMSE","?")} | '
          f'SMAPE={result.get("SMAPE","?")} | NLL={result.get("NLL","?")} | mCRPS={mcrps_str}')
    return result


def main():
    # Checkpoint names based on your local folder
    ckpt_dir = CKPT_DIR

    ABLATION_CONFIGS = [
        {
            'label'      : 'N=1 (Univariate)',
            'ckpt'       : os.path.join(ckpt_dir, 'SCMamba_v_17data_N_uni_best_mase.pth'),
            'num_assets' : 1,
            'dataset'    : 'exchange_rate',
            'sub_day'    : False,
        },
        {
            'label'      : 'N=8 (Cross-Asset Graph)',
            'ckpt'       : os.path.join(ckpt_dir, 'SCMamba_v2_multi_exchange_rate_best_mase.pth'),
            'num_assets' : 8,
            'dataset'    : 'exchange_rate',
            'sub_day'    : False,
        },
        {
            'label'      : 'N=8 (Cross-Asset Graph) best NLL',
            'ckpt'       : os.path.join(ckpt_dir, 'SCMamba_v2_multi_exchange_rate_best.pth'),
            'num_assets' : 8,
            'dataset'    : 'exchange_rate',
            'sub_day'    : False,
        },
    ]

    results = []
    for cfg in ABLATION_CONFIGS:
        print(f'\n━━━━ {cfg["label"]} ━━━━')
        if not os.path.exists(cfg['ckpt']):
            print(f"⚠️ Warning: Checkpoint not found -> {cfg['ckpt']}")
            print("Skipping this configuration.\n")
            continue

        model = load_model(cfg['ckpt'], cfg['num_assets'], SSM_CONFIG)
        row   = evaluate_model(
            model,
            dataset = cfg['dataset'],
            scaler  = SCALER,
            sub_day = cfg['sub_day'],
            label   = cfg['label'],
        )
        results.append(row)

        # Free memory between runs
        del model
        torch.cuda.empty_cache()

    if not results:
        print("No valid results found to display.")
        return

    # ── Results Table + Interpretation ────────────────────────────────────────
    df = pd.DataFrame(results).set_index('label')
    print('\n' + '='*70)
    print('  ABLATION RESULTS: exchange_rate — N=1 vs N=8')
    print('='*70)
    # Show mCRPS instead of CRPS
    metric_cols = [c for c in ['MASE','MAE','RMSE','SMAPE','NLL','mCRPS'] if c in df.columns]
    print(df[metric_cols].to_string())
    print('='*70)
    print('  ↓ lower is better for all metrics')
    print()

    # Relative improvement (N=8 vs N=1)
    if len(df) >= 2:
        base  = df.iloc[0]   # N=1
        multi = df.iloc[1]   # N=8
        print('  Relative delta (N=8 - N=1) / |N=1|:')
        for m in metric_cols:
            delta_pct = (multi[m] - base[m]) / (abs(base[m]) + 1e-10) * 100
            arrow = '🟢' if delta_pct < 0 else '🔴'
            print(f'    {arrow}  {m:6s}: {delta_pct:+.1f}%')
        print()

# if __name__ == '__main__':
main()
