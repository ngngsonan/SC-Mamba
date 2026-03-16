# @title 13_test_zeroshot_multi
"""
13_test_zeroshot_multi.py — v6 (Canonical Refactored)
=====================================================================
Zero-shot Foundation Model benchmark across GluonTS datasets.
Supports N=8 multivariate checkpoints with random sub-sampling.
=====================================================================
"""
import os, sys, yaml, warnings
import numpy as np, pandas as pd, torch
from pathlib import Path

# --- DIRECTORY SETUP (LOCAL/MAC ADAPTIVE) ---
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# sys.path.insert(0, PROJECT_ROOT)
PROJECT_ROOT

from scipy.stats import t as t_dist, norm as scipy_norm
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.time_feature.seasonality import get_seasonality
from utilsforecast.losses import mase, mae, smape, rmse
from data.data_provider.multivariate_loader import MultivariateRealDataset
from core.eval_real_dataset import scale_data, nll_eval, crps_gaussian
from core.models import SCMamba_Forecaster

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONTEXT_LEN = 256       
N_ASSETS    = 8         
SCALER      = 'min_max' 
SEEDS       = [7270, 860, 5390, 5191, 5734]

# Checkpoint path (Using the multi-asset best MASE checkpoint)
CKPT_DIR = '/content/drive/MyDrive/Colab Notebooks/SCMamba/sc_mamba_checkpoints' 
CHECKPOINT_PATH = os.path.join(CKPT_DIR, 'SCMamba_v2_multi_exchange_rate_best_mase.pth')

# All 17 Target datasets
TARGET_DATASETS = {
    "car_parts_without_missing": 12,
    "cif_2016":                  12,
    "covid_deaths":              30,
    "ercot":                     24,
    "exchange_rate":             30,
    "fred_md":                   12,
    "hospital":                  12,
    "m1_monthly":                18,
    "m1_quarterly":              8,
    "m3_monthly":                18,
    "m3_quarterly":              8,
    "nn5_daily_without_missing": 56,
    "nn5_weekly":                8,
    "tourism_monthly":           24,
    "tourism_quarterly":         8,
    "traffic":                   24,
    "weather":                   30,
}

# ─────────────────────────────────────────────────────────────────────────────
# Robust model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model_from_checkpoint(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    ssm_config = ckpt.get('ssm_config', {})

    # Inferred layers if missing
    if 'num_encoder_layers' not in ssm_config:
        layer_indices = sorted(set(int(k.split('.')[2]) for k in state_dict.keys() if 'mamba_encoder_layers.' in k))
        ssm_config['num_encoder_layers'] = max(layer_indices) + 1 if layer_indices else 2

    N_assets = ckpt.get('N_assets', N_ASSETS)
    model = SCMamba_Forecaster(N_assets=N_assets, ssm_config=ssm_config).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"  ✅ Loaded: {os.path.basename(ckpt_path)} | N_assets={model.N_assets}")
    return model

# ─────────────────────────────────────────────────────────────────────────────
# Helper: canonical inference with subsetting
# ─────────────────────────────────────────────────────────────────────────────
def canonical_evaluate(
    model, pkl_path, pred_len, context_len, col_indices, scaler, device, sub_day=False,
):
    N = len(col_indices)
    test_ds = MultivariateRealDataset(pkl_path, pred_len=pred_len, context_len=context_len, split='test', col_indices=col_indices, sub_day=sub_day)
    train_ds = MultivariateRealDataset(pkl_path, pred_len=pred_len, context_len=context_len, split='train', col_indices=col_indices, sub_day=sub_day)

    if len(test_ds) == 0: return None

    ds_name = os.path.basename(pkl_path).replace('_nopad_512.pkl', '')
    try:
        gts_ds = get_dataset(ds_name, regenerate=False)
        seasonality = get_seasonality(gts_ds.metadata.freq)
        if gts_ds.metadata.freq == 'D': seasonality = 7
    except: seasonality = 1

    batch_train_dfs, batch_pred_dfs = [], []

    with torch.no_grad():
        for win_idx in range(len(test_ds)):
            sample = test_ds[win_idx]
            x, y = sample['x'].to(device), sample['y'].to(device)
            ts_x, ts_y = sample['ts_x'].to(device), sample['ts_y'].to(device)

            T_ctx, T_pred = x.shape[0], y.shape[0]
            data = {
                'history': x.permute(1, 0),
                'ts': ts_x.unsqueeze(0).expand(N, -1, -1),
                'target_dates': ts_y.unsqueeze(0).expand(N, -1, -1),
                'task': torch.zeros(N, T_pred, dtype=torch.int32, device=device),
            }

            output = model(data, prediction_length=T_pred)
            scaled_mu, scaled_sigma2 = scale_data(output, scaler)
            mu_np, sig_np, y_np = scaled_mu.cpu().numpy(), scaled_sigma2.cpu().numpy(), y.cpu().numpy()

            for asset_i in range(N):
                asset_id = f"zs_asset_{asset_i}"
                train_sample = train_ds[len(train_ds)-1]
                train_hist_i = train_sample['x'][:, asset_i].numpy()
                
                batch_train_dfs.append(pd.DataFrame({'id': [asset_id]*len(train_hist_i), 'target': train_hist_i}))
                
                sigma_i = np.sqrt(np.clip(sig_np[asset_i], 1e-6, None))
                crps_vals = crps_gaussian(mu_np[asset_i], sigma_i, y_np[:, asset_i])
                
                batch_pred_dfs.append(pd.DataFrame({
                    'id': [asset_id]*T_pred, 'pred': mu_np[asset_i], 'target': y_np[:, asset_i],
                    'variance': sig_np[asset_i], 'nll': nll_eval(torch.tensor(mu_np[asset_i]), torch.tensor(sig_np[asset_i]), torch.tensor(y_np[:, asset_i])).numpy(),
                    'crps': crps_vals
                }))

    train_df = pd.concat(batch_train_dfs)
    pred_df  = pd.concat(batch_pred_dfs)

    # MASE, CRPS, mCRPS
    mase_mean = float(mase(pred_df, ['pred'], seasonality, train_df, 'id', 'target')['pred'].replace([np.inf, -np.inf], np.nan).mean())
    raw_crps = float(pred_df['crps'].mean())
    mean_abs_target = float(train_df['target'].abs().mean()) if train_df['target'].abs().mean() > 0 else 1.0
    mcrps = raw_crps / mean_abs_target

    return {
        'mase': mase_mean, 'mae': float(mae(pred_df, ['pred'], 'id', 'target')['pred'].mean()),
        'rmse': float(rmse(pred_df, ['pred'], 'id', 'target')['pred'].mean()),
        'smape': float(smape(pred_df, ['pred'], 'id', 'target')['pred'].mean()),
        'nll': float(pred_df['nll'].mean()), 'crps': raw_crps, 'mcrps': mcrps
    }

def benchmark_dataset(ds_name, pred_len, model, device, seeds):
    pkl_path = os.path.join(PROJECT_ROOT, 'data', 'real_val_datasets', f'{ds_name}_nopad_512.pkl')
    if not os.path.exists(pkl_path):
        print(f"  ⏭️ Skip: {ds_name} (PKL missing)")
        return None

    seed_results = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        try:
            # Get N_total
            tmp_ds = MultivariateRealDataset(pkl_path, pred_len=pred_len, context_len=CONTEXT_LEN, split='test')
            col_idx = sorted(rng.choice(tmp_ds.N_assets, size=N_ASSETS, replace=False)) if tmp_ds.N_assets >= N_ASSETS else list(range(tmp_ds.N_assets))
            
            res = canonical_evaluate(model, pkl_path, pred_len, CONTEXT_LEN, col_idx, SCALER, device)
            if res: seed_results.append(res)
        except Exception as e:
            print(f"    Seed {seed} error: {e}")
    
    if not seed_results: return None
    
    mase_m = np.mean([r['mase'] for r in seed_results])
    mcrps_m = np.mean([r['mcrps'] for r in seed_results])
    print(f"  📡 {ds_name:25s} | MASE={mase_m:.4f} | mCRPS={mcrps_m:.4f}")
    
    return {
        'dataset': ds_name, 'mase': mase_m, 'mcrps': mcrps_m,
        'mae': np.mean([r['mae'] for r in seed_results]),
        'nll': np.mean([r['nll'] for r in seed_results]),
        'count': len(seed_results)
    }

def main():
    print(f"\n{'='*70}\n🚀 SC-Mamba Multi-Asset Zero-Shot Benchmark\n{'='*70}")
    
    try:
        model = load_model_from_checkpoint(CHECKPOINT_PATH, DEVICE)
    except Exception as e:
        print(f"❌ Load error: {e}")
        return

    results = []
    for ds, pl in TARGET_DATASETS.items():
        res = benchmark_dataset(ds, pl, model, DEVICE, SEEDS)
        if res: results.append(res)

    if not results:
        print("No results generated.")
        return

    df = pd.DataFrame(results)
    print(f"\n{'='*85}")
    print(f"{'Dataset':<30} {'MASE':>10} {'mCRPS':>10} {'MAE':>10} {'NLL':>10}")
    print(f"{'─'*85}")
    for _, r in df.iterrows():
        print(f"{r['dataset']:<30} {r['mase']:>10.4f} {r['mcrps']:>10.4f} {r['mae']:>10.4f} {r['nll']:>10.4f}")
    print(f"{'─'*85}")
    print(f"💎 Avg MASE: {df['mase'].mean():.4f} | Avg mCRPS: {df['mcrps'].mean():.4f}")
    print(f"{'='*85}\n✅ Zero-shot Multi complete.")

# if __name__ == "__main__":
main()

