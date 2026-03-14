"""
02_test_zeroshot_multi.py — v6 (Canonical)
=====================================================================
Zero-shot Foundation Model benchmark across GluonTS datasets.

DESIGN PRINCIPLE:
    Reuses 100% of the canonical evaluation pipeline from training:
      - MultivariateRealDataset   (PKL, sliding windows, time features)
      - scale_data()              (eval_real_dataset.py — proven correct)
      - utilsforecast.mase()      (seasonality-aware denominator)

    The ONLY addition is column sub-sampling: for each seed, we randomly
    select N=8 columns from the target dataset, so the spectral graph
    layer operates on cross-asset relationships it has never seen.

    This eliminates all custom code that caused bugs in v3–v5:
      - No scale_data_fixed  (unsqueeze bug → [N,N,P])
      - No custom lag-1 MASE (denominator mismatch)
      - No manual GluonTS alignment (PKL handles it)

PREREQUISITE:
    PKL files must exist in data/real_val_datasets/.
    Run `python data/scripts/store_real_datasets.py` to generate them.
=====================================================================
"""
# @title  02_test_zeroshot_multi.py v6
import os, sys, pickle, warnings
import numpy as np, pandas as pd, torch

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.environ.get(
    'PROJECT_ROOT',
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)
sys.path.insert(0, PROJECT_ROOT)

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
CONTEXT_LEN = 256       # Must match source training config
N_ASSETS    = 8         # Must match model.N_assets
SCALER      = 'min_max' # Must match training scaler
N_SEEDS     = 5
SEEDS       = [7270, 860, 5390, 5191, 5734]

# Target datasets with their native pred_len
TARGET_DATASETS = {
    "nn5_daily_without_missing": 56,
    "weather":                   30,
    "hospital":                  12,
    "fred_md":                   12,
    "car_parts_without_missing": 12,
    "traffic":                   24,
    "ercot":                     24,
    "exchange_rate":             30,  # sanity: same as training domain
    "tourism_monthly":           24,
    "tourism_quarterly":         8,
    "cif_2016":                  12,
    "m1_monthly":                18,
    "m1_quarterly":              8,
    "nn5_weekly":                8,
    "covid_deaths":              30,
}

# Mamba4Cast in-domain baselines (from leaderboard, seasonality MASE)
BASELINES = {
    "nn5_daily_without_missing": 1.1440,
    "weather":                   1.3876,
    "hospital":                  0.8060,
    "fred_md":                   4.0096,
    "car_parts_without_missing": 0.7925,
    "traffic":                   2.6900,
    "ercot":                     4.8530,
    "exchange_rate":             2.7712,
    "tourism_monthly":           3.8000,
    "tourism_quarterly":         1.5000,
    "cif_2016":                  1.2000,
    "covid_deaths":              7.9000,
    "m1_monthly":                1.2000,
    "m1_quarterly":              1.5000,
    "nn5_weekly":                1.0000,
}


# ─────────────────────────────────────────────────────────────────────────────
# Robust model loading — reads architecture config from checkpoint
# ─────────────────────────────────────────────────────────────────────────────
def load_model_from_checkpoint(ckpt_path, device, N_assets_override=None):
    """
    Load SCMamba_Forecaster from a checkpoint, reconstructing the exact
    architecture the model was trained with.

    The checkpoint (saved by train.py) stores:
      - 'model_state_dict': the weights
      - 'ssm_config': the backbone config dict (num_encoder_layers, etc.)

    If ssm_config is missing (legacy checkpoints), infer num_encoder_layers
    from the state_dict keys.

    Parameters
    ----------
    ckpt_path       : path to .pth checkpoint file
    device          : torch device
    N_assets_override : if set, override N_assets (default: infer from ckpt)

    Returns
    -------
    model : SCMamba_Forecaster on device, in eval mode
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)

    # ── Recover ssm_config from checkpoint ────────────────────────────────────
    ssm_config = ckpt.get('ssm_config', {})

    # If ssm_config doesn't have num_encoder_layers, infer from state_dict
    if 'num_encoder_layers' not in ssm_config:
        layer_indices = set()
        for key in state_dict.keys():
            if 'mamba_encoder_layers.' in key:
                # e.g. "backbone.mamba_encoder_layers.2.mamba_layer.dt_bias"
                parts = key.split('.')
                idx = parts[parts.index('mamba_encoder_layers') + 1]
                layer_indices.add(int(idx))
        if layer_indices:
            inferred = max(layer_indices) + 1
            ssm_config['num_encoder_layers'] = inferred
            print(f"  [loader] Inferred num_encoder_layers={inferred} from state_dict keys")

    # ── Recover N_assets ──────────────────────────────────────────────────────
    if N_assets_override is not None:
        N_assets = N_assets_override
    else:
        # Try checkpoint metadata first, then fall back to config default
        N_assets = ckpt.get('N_assets', N_ASSETS)

    # ── Construct model with correct architecture ─────────────────────────────
    model = SCMamba_Forecaster(N_assets=N_assets, ssm_config=ssm_config).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(f"  ✅ Loaded: {os.path.basename(ckpt_path)}")
    print(f"     N_assets={model.N_assets} | "
          f"num_encoder_layers={ssm_config.get('num_encoder_layers', '?')} | "
          f"tau={model.spectral_layer.tau.item():.4f}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Helper: canonical inference on a sub-sampled MultivariateRealDataset
# Mirrors multivariate_predict_aligned() from eval_real_dataset.py exactly
# ─────────────────────────────────────────────────────────────────────────────
def canonical_evaluate(
    model, pkl_path, pred_len, context_len, col_indices, scaler, device, sub_day=False,
):
    """
    Run canonical multivariate evaluation on a column subset.

    Returns
    -------
    mase_val, mae_val, rmse_val, smape_val, nll_val, crps_val : floats
    """
    N = len(col_indices)

    # Load sub-sampled datasets (test + train for MASE denominator)
    test_ds = MultivariateRealDataset(
        pkl_path, pred_len=pred_len, context_len=context_len,
        split='test', col_indices=col_indices, sub_day=sub_day,
    )
    train_ds = MultivariateRealDataset(
        pkl_path, pred_len=pred_len, context_len=context_len,
        split='train', col_indices=col_indices, sub_day=sub_day,
    )

    if len(test_ds) == 0:
        return None

    # Get seasonality from GluonTS metadata (same as training pipeline)
    ds_name = os.path.basename(pkl_path).replace('_nopad_512.pkl', '')
    try:
        gts_ds = get_dataset(ds_name, regenerate=False)
        seasonality = get_seasonality(gts_ds.metadata.freq)
        if gts_ds.metadata.freq == 'D':
            seasonality = 7
    except Exception:
        seasonality = 1

    model.eval()
    batch_train_dfs = []
    batch_pred_dfs  = []

    with torch.no_grad():
        for win_idx in range(len(test_ds)):
            sample = test_ds[win_idx]
            x    = sample['x'].to(device)     # (T_ctx, N)
            y    = sample['y'].to(device)     # (T_pred, N)
            ts_x = sample['ts_x'].to(device)  # (T_ctx, ts_dim)
            ts_y = sample['ts_y'].to(device)  # (T_pred, ts_dim)

            T_ctx  = x.shape[0]
            T_pred = y.shape[0]

            # ── Same transpose as canonical multivariate_predict_aligned ──────
            history  = x.permute(1, 0)                          # (N, T_ctx)
            ts_x_rep = ts_x.unsqueeze(0).expand(N, -1, -1)     # (N, T_ctx, ts_dim)
            ts_y_rep = ts_y.unsqueeze(0).expand(N, -1, -1)     # (N, T_pred, ts_dim)

            data = {
                'history'      : history,
                'ts'           : ts_x_rep,
                'target_dates' : ts_y_rep,
                'task'         : torch.zeros(N, T_pred, dtype=torch.int32, device=device),
            }

            output = model(data, prediction_length=T_pred)

            # ── Same scale_data (ORIGINAL, proven correct) ────────────────────
            scaled_mu, scaled_sigma2 = scale_data(output, scaler)
            mu_np  = scaled_mu.detach().cpu().numpy()      # (N, T_pred)
            sig_np = scaled_sigma2.detach().cpu().numpy()   # (N, T_pred)
            y_np   = y.cpu().numpy()                        # (T_pred, N)

            # Shape assertions — will catch any tensor bug immediately
            assert mu_np.ndim == 2 and mu_np.shape == (N, T_pred), \
                f"mu shape mismatch: expected ({N}, {T_pred}), got {mu_np.shape}"

            for asset_i in range(N):
                asset_id = f"zs_asset_{asset_i}"

                # MASE denominator: full training history for this asset
                if len(train_ds) > 0:
                    train_sample = train_ds[len(train_ds) - 1]
                    train_hist_i = train_sample['x'][:, asset_i].numpy()
                else:
                    train_hist_i = x[:, asset_i].cpu().numpy()

                batch_train_dfs.append(pd.DataFrame({
                    'id':     [asset_id] * len(train_hist_i),
                    'target': train_hist_i,
                }))

                nll_vals = nll_eval(
                    torch.tensor(mu_np[asset_i]),
                    torch.tensor(sig_np[asset_i]),
                    torch.tensor(y_np[:, asset_i]),
                ).numpy()

                sigma_i = np.sqrt(np.clip(sig_np[asset_i], 1e-6, None))
                crps_vals = crps_gaussian(mu_np[asset_i], sigma_i, y_np[:, asset_i])

                batch_pred_dfs.append(pd.DataFrame({
                    'id':       [asset_id] * T_pred,
                    'pred':     mu_np[asset_i],
                    'target':   y_np[:, asset_i],
                    'variance': sig_np[asset_i],
                    'nll':      nll_vals,
                    'crps':     crps_vals,
                }))

    # ── Same MASE calculation as canonical pipeline ───────────────────────────
    train_df = pd.concat(batch_train_dfs)
    pred_df  = pd.concat(batch_pred_dfs)

    mase_loss  = mase(pred_df, ['pred'], seasonality, train_df, 'id', 'target')
    mae_loss   = mae(pred_df, ['pred'], 'id', 'target')
    rmse_loss  = rmse(pred_df, ['pred'], 'id', 'target')
    smape_loss = smape(pred_df, ['pred'], 'id', 'target')

    mase_vals = mase_loss['pred'].replace([float('inf'), float('-inf')], float('nan'))
    mase_mean = float(mase_vals.mean(skipna=True)) if mase_vals.notna().any() else float('nan')

    return {
        'mase':  mase_mean,
        'mae':   float(mae_loss['pred'].mean()),
        'rmse':  float(rmse_loss['pred'].mean()),
        'smape': float(smape_loss['pred'].mean()),
        'nll':   float(pred_df['nll'].mean()),
        'crps':  float(pred_df['crps'].mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset benchmark (multi-seed)
# ─────────────────────────────────────────────────────────────────────────────
def benchmark_dataset(ds_name, pred_len, model, device, seeds):
    """Run canonical evaluation with K random column subsets."""
    pkl_dir  = os.path.join(PROJECT_ROOT, 'data', 'real_val_datasets')
    pkl_path = os.path.join(pkl_dir, f'{ds_name}_nopad_512.pkl')

    print(f"\n{'─'*60}")
    print(f"  📡  {ds_name}  (pred_len={pred_len})")

    # ── Check PKL exists ──────────────────────────────────────────────────────
    if not os.path.exists(pkl_path):
        print(f"  ⏭️  Skip: PKL not found at {pkl_path}")
        print(f"      Run: python data/scripts/store_real_datasets.py")
        return None

    # ── Probe total columns ───────────────────────────────────────────────────
    try:
        probe_ds = MultivariateRealDataset(
            pkl_path, pred_len=pred_len, context_len=CONTEXT_LEN,
            split='test', sub_day=False,
        )
        N_total = probe_ds.N_assets
        n_windows = len(probe_ds)
        print(f"  N_total={N_total} | test_windows={n_windows}")
    except Exception as e:
        print(f"  ❌ Failed to load: {e}")
        return None

    if N_total < N_ASSETS:
        print(f"  ⏭️  Skip: N_total={N_total} < N_ASSETS={N_ASSETS}")
        return None

    if n_windows == 0:
        print(f"  ⏭️  Skip: 0 test windows (series too short for ctx={CONTEXT_LEN}+pred={pred_len})")
        return None

    # ── Multi-seed evaluation ─────────────────────────────────────────────────
    seed_results = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        col_idx = sorted(rng.choice(N_total, size=N_ASSETS, replace=False))

        try:
            result = canonical_evaluate(
                model, pkl_path, pred_len, CONTEXT_LEN,
                col_indices=col_idx, scaler=SCALER, device=device,
            )
            if result is None:
                print(f"    Seed {seed:5d} | cols={col_idx} | ⏭️ no test windows")
                continue

            print(f"    Seed {seed:5d} | cols={col_idx} | "
                  f"MASE={result['mase']:.4f} | CRPS={result['crps']:.6f}")
            seed_results.append(result)
        except Exception as e:
            print(f"    Seed {seed:5d} | ❌ {e}")
            continue

    if not seed_results:
        print(f"  ❌ All seeds failed")
        return None

    # ── Aggregate across seeds ────────────────────────────────────────────────
    mase_arr = np.array([r['mase'] for r in seed_results])
    crps_arr = np.array([r['crps'] for r in seed_results])

    K = len(mase_arr)
    mase_mean = mase_arr.mean()
    mase_std  = mase_arr.std(ddof=1) if K > 1 else 0.0
    mase_ci   = t_dist.ppf(0.975, K-1) * mase_std / np.sqrt(K) if K > 1 else 0.0
    crps_mean = crps_arr.mean()

    baseline = BASELINES.get(ds_name, None)
    if baseline:
        delta_pct = (1 - mase_mean / baseline) * 100
        delta_str = f"{'✅' if delta_pct > 0 else '  '} {delta_pct:+.1f}%"
    else:
        delta_str = "N/A"

    print(f"  MASE={mase_mean:.4f}±{mase_ci:.4f}  CRPS={crps_mean:.6f}    {delta_str}")

    return {
        'dataset':   ds_name,
        'mase_mean': mase_mean,
        'mase_ci':   mase_ci,
        'crps_mean': crps_mean,
        'baseline':  baseline,
        'delta_str': delta_str,
        'K':         K,
        'mae':       np.mean([r['mae'] for r in seed_results]),
        'rmse':      np.mean([r['rmse'] for r in seed_results]),
        'smape':     np.mean([r['smape'] for r in seed_results]),
        'nll':       np.mean([r['nll'] for r in seed_results]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main — expects `model` and `device` to be defined in the Colab notebook
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  SC-Mamba Zero-Shot Foundation Model Benchmark (v6 Canonical)")
print(f"  Context lock: ctx={CONTEXT_LEN} | N={N_ASSETS}")
print(f"  Seeds: {SEEDS}")
print(f"{'='*60}")

# Contract check
assert hasattr(model, 'N_assets'), "model must be SCMamba_Forecaster"
assert model.N_assets == N_ASSETS, \
    f"Checkpoint N_assets={model.N_assets} != script N_ASSETS={N_ASSETS}"
print(f"  ✅ model.N_assets={model.N_assets} matches N_ASSETS={N_ASSETS}")

all_results = []
for ds_name, pred_len in TARGET_DATASETS.items():
    result = benchmark_dataset(ds_name, pred_len, model, device, SEEDS)
    if result is not None:
        all_results.append(result)

# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print(f"{'Dataset':<35} {'MASE (mean±CI)':>18} {'CRPS':>12} {'MAE':>10} {'vs Baseline':>14}")
print(f"{'─'*90}")

n_beat = 0
for r in all_results:
    mase_str = f"{r['mase_mean']:.4f}±{r['mase_ci']:.4f}"
    line = (f"{r['dataset']:<35} {mase_str:>18} {r['crps_mean']:>12.4f} "
            f"{r['mae']:>10.4f} {r['delta_str']:>14}")
    print(line)
    if r['baseline'] and r['mase_mean'] < r['baseline']:
        n_beat += 1

n_total = len(TARGET_DATASETS)
n_eval  = len(all_results)
n_skip  = n_total - n_eval
print(f"{'─'*90}")
print(f"BEAT BASELINE : {n_beat}/{n_eval}  |  COVERAGE: {n_eval}/{n_total}  |  SKIPPED: {n_skip}")
print(f"{'='*90}")
print(f"\n✅ 02_test_zeroshot_multi.py v6 complete.")
