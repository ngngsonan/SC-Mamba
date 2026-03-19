# @title 14_eval_all.py
"""
14_eval_all.py — Self-Contained Universal Zero-Shot Benchmark
=============================================================
Evaluates ALL checkpoints (Uni N=1, Multi N>1) across ALL 17 GluonTS datasets.

For N>1 models, uses **Asset-dimension Chunking**: the full dataset (M assets)
is split into ceil(M/N) chunks of size N, with circular-padding on the last
chunk. This produces a fair Apple-to-Apple comparison with Univariate SOTA
(Chronos, MOIRAI, TimesFM, Mamba4Cast).

Self-contained: does NOT use subprocess. All inference happens in-process
with detailed per-chunk logging for debugging.
=============================================================
"""

import os, sys, yaml, warnings, pickle, math, time, traceback
import numpy as np, pandas as pd, torch
from pathlib import Path

# ── Path Setup ────────────────────────────────────────────────────────────────
# SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
# PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
PROJECT_ROOT = '/content/SC-Mamba'
CKPT_DIR = '/content/drive/MyDrive/Colab Notebooks/SCMamba/sc_mamba_checkpoints'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scipy.stats import norm as scipy_norm
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.time_feature.seasonality import get_seasonality
from utilsforecast.losses import mase, mae, smape, rmse
from core.eval_real_dataset import scale_data, nll_eval, crps_gaussian
from core.models import SCMamba_Forecaster

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONTEXT_LEN = 256
SCALER      = 'min_max'

# All 17 Target datasets: {name: prediction_length}
TARGET_DATASETS = {
    "nn5_daily_without_missing": 56, "nn5_weekly": 8, "covid_deaths": 30,
    "weather": 30, "hospital": 12, "fred_md": 12,
    "car_parts_without_missing": 12, "traffic": 24,
    "m3_monthly": 18, "ercot": 24, "m1_monthly": 18,
    "m1_quarterly": 8, "cif_2016": 12, "exchange_rate": 30,
    "m3_quarterly": 8, "tourism_monthly": 24, "tourism_quarterly": 8,
}


# Format: (label, checkpoint_path, config_yaml_path_or_None)
MODEL_TO_TEST = [
    (
        'N=1 MASE',
        os.path.join(CKPT_DIR, 'SCMamba_v3_17data_N_uni_best.pth'),
        os.path.join(PROJECT_ROOT, 'core', 'config.based_setup.yaml'),
    ),
    (
        'N=1 NLL',
        os.path.join(CKPT_DIR, 'SCMamba_v3_17data_N_uni_best_mase.pth'),
        os.path.join(PROJECT_ROOT, 'core', 'config.based_setup.yaml'),
    ),
    (
        'N=8 (Chunked)',
        os.path.join(CKPT_DIR, 'SCMamba_v3_multi_exchange_rate_best.pth'),
        os.path.join(PROJECT_ROOT, 'core', 'config.v3_multi_exchange_rate.yaml'),
    ),
]
REAL_VAL_DIR = os.path.join(PROJECT_ROOT, 'data', 'real_val_datasets')


# ─────────────────────────────────────────────────────────────────────────────
# Ensure all dataset PKLs exist
# ─────────────────────────────────────────────────────────────────────────────
def ensure_datasets():
    """Generate missing dataset PKLs from GluonTS."""
    os.makedirs(REAL_VAL_DIR, exist_ok=True)
    missing = [ds for ds in TARGET_DATASETS
               if not os.path.exists(os.path.join(REAL_VAL_DIR, f'{ds}_nopad_512.pkl'))]
    if missing:
        print(f"\n🔄 Generating {len(missing)} missing dataset(s) from GluonTS...")
        import subprocess
        subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, 'data', 'scripts', 'store_real_datasets.py')])
    return missing


# ─────────────────────────────────────────────────────────────────────────────
# Robust Model Loading (same logic as script 13)
# ─────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path, config_yaml_path, device):
    """Load model from checkpoint with config YAML for architecture resolution."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    ssm_config = ckpt.get('ssm_config', {})

    # Infer num_encoder_layers from state_dict keys if missing
    if 'num_encoder_layers' not in ssm_config:
        layer_indices = sorted(set(
            int(k.split('.')[2]) for k in state_dict.keys()
            if 'mamba_encoder_layers.' in k
        ))
        ssm_config['num_encoder_layers'] = max(layer_indices) + 1 if layer_indices else 2

    # Resolve N_assets from config YAML (priority) → checkpoint → default
    n_assets = 1
    sub_day = False
    if config_yaml_path and os.path.exists(config_yaml_path):
        with open(config_yaml_path) as f:
            cfg = yaml.safe_load(f)
        n_assets = cfg.get('num_assets', 1)
        sub_day = cfg.get('sub_day', False)
    else:
        # Fallback: infer from checkpoint name or embedded metadata
        if 'uni' in os.path.basename(ckpt_path).lower():
            n_assets = 1
        else:
            n_assets = ckpt.get('N_assets', ckpt.get('num_assets', 1))

    # Override ssm_config from YAML if available
    if config_yaml_path and os.path.exists(config_yaml_path):
        yaml_ssm = cfg.get('ssm_config', {})
        for k, v in yaml_ssm.items():
            ssm_config[k] = v

    model = SCMamba_Forecaster(N_assets=n_assets, ssm_config=ssm_config).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"  ✅ Loaded: {os.path.basename(ckpt_path)}")
    print(f"     N_assets={model.N_assets}, sub_day={sub_day}")
    print(f"     mamba2={ssm_config.get('mamba2','?')}, d_state={ssm_config.get('d_state','?')}, "
          f"layers={ssm_config.get('num_encoder_layers','?')}")
    return model, sub_day


# ─────────────────────────────────────────────────────────────────────────────
# Inline RobustZeroShotDataset (from script 13, production-proven)
# ─────────────────────────────────────────────────────────────────────────────
class RobustZeroShotDataset(torch.utils.data.Dataset):
    """
    Handles sparse/asynchronous multivariate PKLs gracefully.
    - Dense time-cropping (drop rows where ALL selected series are NaN).
    - ffill/bfill for spectral soundness (FFT requires continuous signal).
    """
    def __init__(self, pkl_path, pred_len, context_len, split, col_indices, sub_day=False):
        self.pred_len = pred_len
        self.context_len = context_len
        self.N_assets = len(col_indices)
        self.sub_day = sub_day

        with open(pkl_path, 'rb') as f:
            df_raw = pickle.load(f)

        df_flat = df_raw.reset_index()
        df_piv = df_flat.pivot_table(
            index='date', columns='Series', values='target', aggfunc='first'
        ).sort_index()

        available = df_piv.shape[1]
        valid_idx = [i for i in col_indices if i < available]
        if len(valid_idx) < self.N_assets:
            raise ValueError(
                f"Requested {self.N_assets} assets, only {len(valid_idx)} valid "
                f"(available={available})"
            )

        df_sub = df_piv.iloc[:, valid_idx]

        # Dense Time-Cropping: drop rows where all selected series are NaN
        orig_len = len(df_sub)
        df_sub = df_sub.dropna(how='all')
        cropped_len = len(df_sub)
        nan_count = df_sub.isna().sum().sum()
        total_cells = df_sub.size

        df_sub = df_sub.ffill().bfill().fillna(0.0)

        # Build timestamp features
        ts_index = pd.to_datetime(df_sub.index)
        if sub_day:
            ts_feats = np.stack([
                ts_index.year, ts_index.month, ts_index.day,
                ts_index.day_of_week + 1, ts_index.day_of_year,
                ts_index.hour, ts_index.minute
            ], axis=-1)
        else:
            ts_feats = np.stack([
                ts_index.year, ts_index.month, ts_index.day,
                ts_index.day_of_week + 1, ts_index.day_of_year
            ], axis=-1)

        self.ts_feats = ts_feats.astype(np.float32)
        self.values = df_sub.values.astype(np.float32)

        T_total = len(df_sub)
        n_test = pred_len
        min_train_required = context_len + pred_len

        if T_total < min_train_required + pred_len:
            # Dataset too short — zero windows
            self.n_windows = 0
            self._start = 0
            self._end = T_total
            self._split = split
            self._val_target_start = 0
            self._test_target_start = 0
            return

        max_val_allowed = max(pred_len, T_total - n_test - min_train_required)
        n_val = min(max(pred_len, 30), max_val_allowed)

        ideal_train_end = T_total - n_test - n_val
        train_end = max(ideal_train_end, min_train_required)
        if train_end > T_total:
            train_end = T_total

        if split == 'train':
            self._start, self._end = 0, train_end
            self.n_windows = max(0, train_end - context_len - pred_len + 1)
        elif split == 'val':
            val_target_start = T_total - n_test - n_val
            self._start = max(0, val_target_start - context_len)
            self._end = T_total - n_test
            self.n_windows = max(0, n_val - pred_len + 1)
        else:  # test
            test_target_start = T_total - n_test
            self._start = max(0, test_target_start - context_len)
            self._end = T_total
            self.n_windows = max(0, n_test - pred_len + 1)

        self._split = split
        self._val_target_start = T_total - n_test - n_val
        self._test_target_start = T_total - n_test

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        if self._split == 'train':
            abs_start = self._start + idx
            ctx_end = abs_start + self.context_len
        else:
            target_start = (self._val_target_start if self._split == 'val'
                            else self._test_target_start) + idx
            abs_start = max(0, target_start - self.context_len)
            ctx_end = target_start

        tgt_end = ctx_end + self.pred_len
        ctx_len_actual = ctx_end - abs_start

        if ctx_len_actual < self.context_len:
            pad = self.context_len - ctx_len_actual
            x = np.concatenate([
                np.zeros((pad, self.N_assets), dtype=np.float32),
                self.values[abs_start:ctx_end]
            ], axis=0)
            ts_x = np.concatenate([
                np.zeros((pad, self.ts_feats.shape[1]), dtype=np.float32),
                self.ts_feats[abs_start:ctx_end]
            ], axis=0)
        else:
            x = self.values[abs_start:ctx_end]
            ts_x = self.ts_feats[abs_start:ctx_end]

        y = self.values[ctx_end:tgt_end]
        ts_y = self.ts_feats[ctx_end:tgt_end]

        return {
            'x': torch.from_numpy(x), 'y': torch.from_numpy(y),
            'ts_x': torch.from_numpy(ts_x), 'ts_y': torch.from_numpy(ts_y),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Core: Chunked Evaluation (the heart of Asset-dimension Chunking)
# ─────────────────────────────────────────────────────────────────────────────
def get_total_assets(pkl_path):
    """Count total number of unique series in a PKL file."""
    with open(pkl_path, 'rb') as f:
        df = pickle.load(f)
    return len(df.index.get_level_values('Series').unique())


def evaluate_single_chunk(model, pkl_path, pred_len, context_len,
                          col_indices, scaler, device, sub_day, chunk_label=""):
    """
    Run inference on ONE chunk of assets. Returns dict of metrics or None.
    Works for both Uni (N=1) and Multi (N>1) models.
    """
    N = len(col_indices)
    model_n = getattr(model, 'N_assets', 1)

    try:
        test_ds = RobustZeroShotDataset(
            pkl_path, pred_len=pred_len, context_len=context_len,
            split='test', col_indices=col_indices, sub_day=sub_day,
        )
        train_ds = RobustZeroShotDataset(
            pkl_path, pred_len=pred_len, context_len=context_len,
            split='train', col_indices=col_indices, sub_day=sub_day,
        )
    except Exception as e:
        print(f"      ❌ {chunk_label} Loader error: {e}")
        return None

    if len(test_ds) == 0:
        print(f"      ⚠️ {chunk_label} 0 test windows after cropping. Skipping.")
        return None

    batch_train_dfs, batch_pred_dfs = [], []

    with torch.no_grad():
        for win_idx in range(len(test_ds)):
            sample = test_ds[win_idx]
            x = sample['x'].to(device)       # (ctx, N)
            y = sample['y'].to(device)        # (pred, N)
            ts_x = sample['ts_x'].to(device)
            ts_y = sample['ts_y'].to(device)
            T_pred = y.shape[0]

            if model_n == 1:
                # Univariate: loop each asset independently
                mu_list, sig_list = [], []
                for ai in range(N):
                    data = {
                        'history': x[:, ai:ai+1].permute(1, 0),
                        'ts': ts_x.unsqueeze(0),
                        'target_dates': ts_y.unsqueeze(0),
                        'task': torch.zeros(1, T_pred, dtype=torch.int32, device=device),
                    }
                    out = model(data, prediction_length=T_pred)
                    a_mu, a_sig = scale_data(out, scaler)
                    mu_list.append(a_mu)
                    sig_list.append(a_sig)
                scaled_mu = torch.cat(mu_list, dim=0)
                scaled_sigma2 = torch.cat(sig_list, dim=0)
            else:
                # Multivariate: forward all N assets at once
                data = {
                    'history': x.permute(1, 0),           # (N, ctx)
                    'ts': ts_x.unsqueeze(0).expand(N, -1, -1),
                    'target_dates': ts_y.unsqueeze(0).expand(N, -1, -1),
                    'task': torch.zeros(N, T_pred, dtype=torch.int32, device=device),
                }
                output = model(data, prediction_length=T_pred)
                scaled_mu, scaled_sigma2 = scale_data(output, scaler)

            mu_np = scaled_mu.cpu().numpy()      # (N, T_pred)
            sig_np = scaled_sigma2.cpu().numpy()  # (N, T_pred)
            y_np = y.cpu().numpy()                # (T_pred, N)

            for ai in range(N):
                aid = f"asset_{col_indices[ai]}"

                train_hist = train_ds.values[:, ai]
                batch_train_dfs.append(pd.DataFrame({
                    'id': [aid] * len(train_hist), 'target': train_hist,
                }))

                sigma_i = np.sqrt(np.clip(sig_np[ai], 1e-6, None))
                crps_vals = crps_gaussian(mu_np[ai], sigma_i, y_np[:, ai])
                nll_vals = nll_eval(
                    torch.tensor(mu_np[ai]),
                    torch.tensor(sig_np[ai]),
                    torch.tensor(y_np[:, ai]),
                ).numpy()

                batch_pred_dfs.append(pd.DataFrame({
                    'id': [aid] * T_pred,
                    'pred': mu_np[ai],
                    'target': y_np[:, ai],
                    'variance': sig_np[ai],
                    'nll': nll_vals,
                    'crps': crps_vals,
                }))

    return batch_train_dfs, batch_pred_dfs


def compute_metrics(batch_train_dfs, batch_pred_dfs, ds_name):
    """Compute MASE and mCRPS from prediction DataFrames."""
    if not batch_pred_dfs:
        return {'mase': np.nan, 'mcrps': np.nan}

    train_df = pd.concat(batch_train_dfs)
    pred_df = pd.concat(batch_pred_dfs)

    try:
        gts_ds = get_dataset(ds_name, regenerate=False)
        seasonality = get_seasonality(gts_ds.metadata.freq)
        if gts_ds.metadata.freq == 'D':
            seasonality = 7
    except Exception:
        seasonality = 1

    # MASE (robust against inf from zero-variance training windows)
    try:
        mase_res = mase(pred_df, ['pred'], seasonality, train_df, 'id', 'target')
        mase_series = mase_res['pred'].replace([np.inf, -np.inf], np.nan)
        mase_mean = float(mase_series.mean(skipna=True))
        if np.isnan(mase_mean):
            mase_mean = float('nan')
    except Exception as e:
        print(f"      [MASE Error] {e}")
        mase_mean = float('nan')

    # mCRPS (Mean-Scaled CRPS)
    raw_crps = float(pred_df['crps'].replace([np.inf, -np.inf], np.nan).mean(skipna=True))
    mean_abs_target = float(train_df['target'].abs().mean(skipna=True))
    if mean_abs_target < 1e-8:
        mean_abs_target = 1.0
    mcrps = raw_crps / mean_abs_target

    return {'mase': mase_mean, 'mcrps': mcrps}


# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluation Loop
# ─────────────────────────────────────────────────────────────────────────────
# Evaluation strategy constants
# ─────────────────────────────────────────────────────────────────────────────
# For N=1 models: random-subset sampling, same protocol as script 13.
# Using 8 random assets × 5 fixed seeds → averaged result.
# This is statistically valid, fast (no timeout), and avoids async-series issues.
UNI_SAMPLE_N = 8        # How many assets to sample per seed for N=1 eval
EVAL_SEEDS   = [7270, 860, 5390, 5191, 5734]  # Fixed seeds (same as script 13)


def evaluate_model_on_dataset(model, ds_name, pred_len, device, sub_day):
    """
    Evaluate a model on a single dataset.

    Dual strategy:
    ─ N=1 (Univariate): Random-subset sampling.
        Pick UNI_SAMPLE_N=8 assets per seed × EVAL_SEEDS (5 seeds).
        Each seed is evaluated with evaluate_single_chunk (N=1 → Uni inner loop).
        Average MASE/mCRPS across seeds.
        ✓ Fast (no PKL reload storm).
        ✓ Handles asynchronous datasets (only 1 or few assets → no alignment issue).
        ✓ Same protocol as script 13, reviewers accept it.

    ─ N>1 (Multivariate): Full Asset-dimension Chunking.
        Split all M assets into K=ceil(M/N) chunks, circular-pad last chunk.
        Evaluate ALL M assets → Full-dataset metrics.
        This is the novel contribution allowing fair comparison with Uni baselines.
    """
    model_n = getattr(model, 'N_assets', 1)
    pkl_path = os.path.join(REAL_VAL_DIR, f'{ds_name}_nopad_512.pkl')

    if not os.path.exists(pkl_path):
        print(f"    ⏭️ PKL missing: {pkl_path}")
        return None

    try:
        total_M = get_total_assets(pkl_path)
    except Exception as e:
        print(f"    ⏭️ PKL metadata error: {e}")
        return None

    print(f"    📊 {ds_name}: {total_M} assets, model_N={model_n}, pred_len={pred_len}")

    if model_n == 1:
        # ── N=1: Random-subset sampling (same as script 13) ─────────────────
        # Do NOT group multiple assets — each seed gets its own col_indices slice
        # passed as a group so the inner loop treats them as independent assets.
        seed_metrics = []

        for seed in EVAL_SEEDS:
            rng = np.random.default_rng(seed)
            sample_n = min(UNI_SAMPLE_N, total_M)  # safety for tiny datasets
            col_indices = sorted(rng.choice(total_M, size=sample_n, replace=False).tolist())

            result = evaluate_single_chunk(
                model, pkl_path, pred_len, CONTEXT_LEN,
                col_indices, SCALER, device, sub_day,
                chunk_label=f"[Seed {seed}]",
            )
            if result is not None:
                m = compute_metrics(result[0], result[1], ds_name)
                if pd.notna(m['mase']) and pd.notna(m['mcrps']):
                    seed_metrics.append(m)
                    print(f"      Seed {seed}: MASE={m['mase']:.4f}, mCRPS={m['mcrps']:.4f}")
                else:
                    print(f"      Seed {seed}: metric NaN, skipped")
            else:
                print(f"      Seed {seed}: no valid windows, skipped")

        if not seed_metrics:
            print(f"    ❌ No valid seed results for {ds_name}")
            return None

        metrics = {
            'mase':  float(np.nanmean([m['mase']  for m in seed_metrics])),
            'mcrps': float(np.nanmean([m['mcrps'] for m in seed_metrics])),
        }
        print(f"    ✅ {ds_name}: MASE={metrics['mase']:.4f}, mCRPS={metrics['mcrps']:.4f} "
              f"(avg {len(seed_metrics)}/{len(EVAL_SEEDS)} seeds, sample_n={sample_n})")
        return metrics

    else:
        # ── N>1: Asset-dimension Chunking — evaluate FULL dataset ──────────
        N = model_n
        K = math.ceil(total_M / N)
        print(f"    🔀 Chunked eval: {total_M} assets → {K} chunks of {N}")

        all_train_dfs, all_pred_dfs = [], []
        n_chunks_ok = 0

        for k in range(K):
            start = k * N
            end = min((k + 1) * N, total_M)
            valid_len = end - start

            # Circular-pad last chunk if needed
            col_indices = list(range(start, end))
            if valid_len < N:
                pad_needed = N - valid_len
                col_indices += [col_indices[i % valid_len] for i in range(pad_needed)]
                if k == K - 1:  # only print pad info for last chunk
                    print(f"      Last chunk {k+1}/{K}: assets [{start}..{end-1}] + {pad_needed} pad")
            elif k % 10 == 0 or k == K - 1:
                # Print every 10th chunk to avoid log spam
                print(f"      Chunk {k+1}/{K}: assets [{start}..{end-1}]")

            result = evaluate_single_chunk(
                model, pkl_path, pred_len, CONTEXT_LEN,
                col_indices, SCALER, device, sub_day,
                chunk_label=f"[Chunk {k+1}/{K}]",
            )

            if result is not None:
                train_dfs, pred_dfs = result
                all_train_dfs.extend(train_dfs)
                all_pred_dfs.extend(pred_dfs)
                n_chunks_ok += 1

        if not all_pred_dfs:
            print(f"    ❌ No valid predictions for {ds_name}")
            return None

        metrics = compute_metrics(all_train_dfs, all_pred_dfs, ds_name)
        print(f"    ✅ {ds_name}: MASE={metrics['mase']:.4f}, mCRPS={metrics['mcrps']:.4f} "
              f"({n_chunks_ok}/{K} chunks successful, {len(all_pred_dfs)} pred-blocks)")
        return metrics


def main():
    print(f"\n{'='*110}")
    print(f"🚀 SC-Mamba Universal Zero-Shot Benchmark (17 Datasets)")
    print(f"   Asset-dimension Chunking for N>1 models")
    print(f"   Device: {DEVICE}")
    print(f"{'='*110}\n")

    ensure_datasets()

    all_results = []
    model_labels = []

    for label, ckpt_path, config_path in MODEL_TO_TEST:
        print(f"\n{'─'*80}")
        print(f"▶️  {label}")
        print(f"   Checkpoint: {ckpt_path}")
        print(f"   Config:     {config_path}")
        print(f"{'─'*80}")

        if not os.path.exists(ckpt_path):
            print(f"   ❌ Checkpoint not found! Skipping.")
            continue

        try:
            model, sub_day = load_model(ckpt_path, config_path, DEVICE)
        except Exception as e:
            print(f"   ❌ Load failed: {e}")
            traceback.print_exc()
            continue

        model_labels.append(label)
        model_n = getattr(model, 'N_assets', 1)

        for ds_name, pred_len in TARGET_DATASETS.items():
            t0 = time.time()
            try:
                metrics = evaluate_model_on_dataset(model, ds_name, pred_len, DEVICE, sub_day)
            except Exception as e:
                print(f"    ❌ {ds_name} EXCEPTION: {e}")
                traceback.print_exc()
                metrics = None

            elapsed = time.time() - t0

            if metrics:
                all_results.append({
                    'Model': label, 'Dataset': ds_name,
                    'MASE': metrics['mase'], 'mCRPS': metrics['mcrps'],
                })
                print(f"    ⏱️  {elapsed:.1f}s")
            else:
                all_results.append({
                    'Model': label, 'Dataset': ds_name,
                    'MASE': np.nan, 'mCRPS': np.nan,
                })
                print(f"    ⏱️  {elapsed:.1f}s (FAILED)")

        # Free GPU memory between models
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if not all_results:
        print("\n❌ No results. Check checkpoints and dataset paths.")
        return

    # ── Build Comparison Table ────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    n_models = len(model_labels)
    col_w = 22
    ds_w = 30
    sep_w = ds_w + 3 + (col_w + 3) * n_models * 2

    print(f"\n\n{'='*sep_w}")
    print(f"📊 FULL BENCHMARK RESULTS")
    print(f"{'='*sep_w}")

    # Header row 1
    mase_hdr = "MASE".center((col_w + 3) * n_models - 3)
    mcrps_hdr = "mCRPS".center((col_w + 3) * n_models - 3)
    print(f"{'Dataset':<{ds_w}} | {mase_hdr} | {mcrps_hdr}")

    # Header row 2 (model names)
    model_cols = " | ".join(f"{m[:col_w]:<{col_w}}" for m in model_labels)
    print(f"{'':<{ds_w}} | {model_cols} | {model_cols}")
    print(f"{'─'*sep_w}")

    pivot_mase = df.pivot(index='Dataset', columns='Model', values='MASE')
    pivot_mcrps = df.pivot(index='Dataset', columns='Model', values='mCRPS')

    for ds in TARGET_DATASETS:
        if ds not in pivot_mase.index:
            continue
        row = f"{ds:<{ds_w}} | "
        for m in model_labels:
            v = pivot_mase.at[ds, m] if m in pivot_mase.columns else np.nan
            row += f"{v:>{col_w}.4f} | " if pd.notna(v) else f"{'—':>{col_w}} | "
        for m in model_labels:
            v = pivot_mcrps.at[ds, m] if m in pivot_mcrps.columns else np.nan
            row += f"{v:>{col_w}.4f} | " if pd.notna(v) else f"{'—':>{col_w}} | "
        print(row)

    print(f"{'─'*sep_w}")

    # Global summary
    summary = df.groupby('Model')[['MASE', 'mCRPS']].agg(
        lambda x: np.nanmean(x) if x.notna().any() else np.nan
    ).rename(columns={'MASE': 'Avg MASE', 'mCRPS': 'Avg mCRPS'})
    summary = summary.reindex([lbl for lbl in model_labels if lbl in summary.index])

    print(f"\n📊 GLOBAL SUMMARY (Mean across {len(TARGET_DATASETS)} datasets):\n")
    print(summary.to_string(float_format='%.4f', na_rep='—'))
    print(f"\n{'='*sep_w}")
    print(f"✅ Benchmark 14 complete.\n")


# if __name__ == '__main__':
main()
