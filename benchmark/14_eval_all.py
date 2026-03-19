# @title 14_eval_all.py
"""
14_eval_all.py — Gold-Standard Multivariate Zero-Shot Benchmark (Asset-dimension Chunking)
========================================================================================
Evaluates Multivariate (N>1) SC-Mamba models across ALL 17 GluonTS datasets.
(Note: Univariate N=1 evaluation is handled separately in `12_eval ckp_N_uni.py`
 using the original time-axis and full dataset cardinality).

For N>1 models, this script implements Apple-to-Apple "Asset-dimension Chunking":
  - K = ⌈M/N⌉ sequential chunks via RobustZeroShotDataset.
  - Test sets with length < context_len correctly pad their historical context (0-windows bug fixed).
========================================================================================
"""

import os, sys, yaml, warnings, pickle, math, time, traceback, subprocess
import numpy as np, pandas as pd, torch
from pathlib import Path

# ── Path Setup (Colab-adaptive) ──────────────────────────────────────────────
# NOTE: User may override these for Colab environment
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
colab_ckpt = '/content/drive/MyDrive/Colab Notebooks/SCMamba/sc_mamba_checkpoints'
CKPT_DIR = colab_ckpt if os.path.exists(colab_ckpt) else os.path.join(PROJECT_ROOT, 'checkpoints')

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scipy.stats import norm as scipy_norm
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.time_feature.seasonality import get_seasonality
from utilsforecast.losses import mase, mae, smape, rmse
from core.eval_real_dataset import scale_data, nll_eval, crps_gaussian, REAL_DATASETS
from core.models import SCMamba_Forecaster

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONTEXT_LEN = 256
SCALER      = 'min_max'

# All 17 Target datasets: {name: prediction_length}
TARGET_DATASETS = dict(REAL_DATASETS)  # Import from eval_real_dataset.py

# Format: (label, checkpoint_path, config_yaml_path)
# Only testing Multivariate N>1 models here.
MODEL_TO_TEST = [
    (
        'N=8 (Chunked)',
        os.path.join(CKPT_DIR, 'SCMamba_v3_multi_exchange_rate_best.pth'),
        os.path.join(PROJECT_ROOT, 'core', 'config.v3_multi_exchange_rate.yaml'),
    ),
]

REAL_VAL_DIR = os.path.join(PROJECT_ROOT, 'data', 'real_val_datasets')

# Suffix stripping: must match eval_real_dataset.py::main_evaluator line 826
_STRIP_SUFFIXES = ('_best_mase', '_best', '_Final')


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def derive_model_name(ckpt_path):
    """Derive cache directory name from checkpoint filename."""
    name = os.path.basename(ckpt_path).replace('.pth', '')
    for suffix in _STRIP_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def resolve_n_assets(config_path):
    """Read num_assets from config YAML. Returns 1 if missing or file not found."""
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get('num_assets', 1)
    return 1


def ensure_datasets():
    """Generate missing dataset PKLs from GluonTS."""
    os.makedirs(REAL_VAL_DIR, exist_ok=True)
    missing = [ds for ds in TARGET_DATASETS
               if not os.path.exists(os.path.join(REAL_VAL_DIR, f'{ds}_nopad_512.pkl'))]
    if missing:
        import subprocess
        print(f"\n🔄 Generating {len(missing)} missing dataset(s) from GluonTS...")
        subprocess.run([sys.executable,
                        os.path.join(PROJECT_ROOT, 'data', 'scripts', 'store_real_datasets.py')])


# ═══════════════════════════════════════════════════════════════════════════════
# PATH B: N>1 Evaluation via Inline Asset-dimension Chunking
# ═══════════════════════════════════════════════════════════════════════════════

class RobustZeroShotDataset(torch.utils.data.Dataset):
    """
    Handles sparse/asynchronous multivariate PKLs gracefully.
    Dense time-cropping + ffill/bfill for FFT spectral soundness.
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
        df_sub = df_sub.dropna(how='all')
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

        # CRITICAL FIX for asynchronous/short datasets:
        # Guarantee at least 1 testing window even if T_total < min_train_required + pred_len
        ideal_train_end = T_total - n_test - min(max(pred_len, 30), max(pred_len, T_total - n_test - min_train_required))
        train_end = max(ideal_train_end, min_train_required)
        
        # Extreme fallback: if dataset is even smaller than 1 window, use all for train
        if train_end > T_total:
            train_end = T_total

        if split == 'train':
            self._start = 0
            self._end = train_end
            self.n_windows = max(0, train_end - context_len - pred_len + 1)
        
        elif split == 'val':
            n_val = min(max(pred_len, 30), max(pred_len, T_total - n_test - min_train_required))
            val_target_start = T_total - n_test - n_val
            val_target_end = T_total - n_test
            self._start = max(0, val_target_start - context_len)
            self._end = val_target_end
            self.n_windows = max(0, n_val - pred_len + 1)
        
        else:  # test
            test_target_start = T_total - n_test
            self._start = max(0, test_target_start - context_len)
            self._end = T_total
            self.n_windows = max(0, n_test - pred_len + 1)

        self._split = split
        self._val_target_start = T_total - n_test - min(max(pred_len, 30), max(pred_len, T_total - n_test - min_train_required))
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

        # Handle prediction length padding if test set is extremely short (T_total < pred_len)
        y_actual_len = tgt_end - ctx_end
        if y_actual_len < self.pred_len:
            pad_y = self.pred_len - y_actual_len
            y = np.concatenate([
                self.values[ctx_end:tgt_end],
                np.zeros((pad_y, self.N_assets), dtype=np.float32)
            ], axis=0)
            ts_y = np.concatenate([
                self.ts_feats[ctx_end:tgt_end],
                np.zeros((pad_y, self.ts_feats.shape[1]), dtype=np.float32)
            ], axis=0)
        else:
            y = self.values[ctx_end:tgt_end]
            ts_y = self.ts_feats[ctx_end:tgt_end]

        return {
            'x': torch.from_numpy(x), 'y': torch.from_numpy(y),
            'ts_x': torch.from_numpy(ts_x), 'ts_y': torch.from_numpy(ts_y),
        }


def get_total_assets(pkl_path):
    """Count total unique series in a PKL file."""
    with open(pkl_path, 'rb') as f:
        df = pickle.load(f)
    return len(df.index.get_level_values('Series').unique())


def load_model(ckpt_path, config_path, device):
    """Load SC-Mamba model from checkpoint + config YAML."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    ssm_config = ckpt.get('ssm_config', {})

    if 'num_encoder_layers' not in ssm_config:
        layer_indices = sorted(set(
            int(k.split('.')[2]) for k in state_dict.keys()
            if 'mamba_encoder_layers.' in k
        ))
        ssm_config['num_encoder_layers'] = max(layer_indices) + 1 if layer_indices else 2

    n_assets = resolve_n_assets(config_path)
    sub_day = False
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        sub_day = cfg.get('sub_day', False)
        yaml_ssm = cfg.get('ssm_config', {})
        for k, v in yaml_ssm.items():
            ssm_config[k] = v

    model = SCMamba_Forecaster(N_assets=n_assets, ssm_config=ssm_config).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"    ✅ Loaded: {os.path.basename(ckpt_path)}")
    print(f"       N_assets={model.N_assets}, sub_day={sub_day}, "
          f"mamba2={ssm_config.get('mamba2','?')}, d_state={ssm_config.get('d_state','?')}")
    return model, sub_day


def evaluate_chunk(model, pkl_path, pred_len, context_len,
                   col_indices, scaler, device, sub_day, chunk_label=""):
    """Run inference on ONE chunk of assets. Returns (train_dfs, pred_dfs) or None."""
    N = len(col_indices)
    try:
        test_ds = RobustZeroShotDataset(pkl_path, pred_len, context_len, 'test', col_indices, sub_day)
        train_ds = RobustZeroShotDataset(pkl_path, pred_len, context_len, 'train', col_indices, sub_day)
    except Exception as e:
        print(f"      ❌ {chunk_label} Loader: {e}")
        return None

    if len(test_ds) == 0:
        print(f"      ⚠️ {chunk_label} 0 test windows. Skip.")
        return None

    batch_train_dfs, batch_pred_dfs = [], []

    with torch.no_grad():
        for win_idx in range(len(test_ds)):
            sample = test_ds[win_idx]
            x = sample['x'].to(device)
            y = sample['y'].to(device)
            ts_x = sample['ts_x'].to(device)
            ts_y = sample['ts_y'].to(device)
            T_pred = y.shape[0]

            data = {
                'history': x.permute(1, 0),
                'ts': ts_x.unsqueeze(0).expand(N, -1, -1),
                'target_dates': ts_y.unsqueeze(0).expand(N, -1, -1),
                'task': torch.zeros(N, T_pred, dtype=torch.int32, device=device),
            }
            output = model(data, prediction_length=T_pred)
            scaled_mu, scaled_sigma2 = scale_data(output, scaler)

            mu_np = scaled_mu.cpu().numpy()
            sig_np = scaled_sigma2.cpu().numpy()
            y_np = y.cpu().numpy()

            for ai in range(N):
                aid = f"asset_{col_indices[ai]}"
                # Even if len(train_ds) == 0, train_ds.values contains all historical data
                train_hist = train_ds.values[:, ai]
                if len(train_hist) == 0:
                     train_hist = np.zeros(1) # fallback array to avoid pd.concat failures
                batch_train_dfs.append(pd.DataFrame({
                    'id': [aid] * len(train_hist), 'target': train_hist,
                }))
                sigma_i = np.sqrt(np.clip(sig_np[ai], 1e-6, None))
                crps_vals = crps_gaussian(mu_np[ai], sigma_i, y_np[:, ai])
                nll_vals = nll_eval(
                    torch.tensor(mu_np[ai]), torch.tensor(sig_np[ai]),
                    torch.tensor(y_np[:, ai])
                ).numpy()
                batch_pred_dfs.append(pd.DataFrame({
                    'id': [aid] * T_pred, 'pred': mu_np[ai],
                    'target': y_np[:, ai], 'variance': sig_np[ai],
                    'nll': nll_vals, 'crps': crps_vals,
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

    try:
        mase_res = mase(pred_df, ['pred'], seasonality, train_df, 'id', 'target')
        mase_series = mase_res['pred'].replace([np.inf, -np.inf], np.nan)
        mase_mean = float(mase_series.mean(skipna=True))
        if np.isnan(mase_mean):
            mase_mean = float('nan')
    except Exception as e:
        print(f"      [MASE Error] {e}")
        mase_mean = float('nan')

    raw_crps = float(pred_df['crps'].replace([np.inf, -np.inf], np.nan).mean(skipna=True))
    mean_abs_target = float(train_df['target'].abs().mean(skipna=True))
    if mean_abs_target < 1e-8:
        mean_abs_target = 1.0
    mcrps = raw_crps / mean_abs_target

    return {'mase': mase_mean, 'mcrps': mcrps}


def eval_chunked_dataset(model, ds_name, pred_len, device, sub_day):
    """
    Asset-dimension Chunking: evaluate ALL M assets via K=⌈M/N⌉ chunks.
    """
    N = getattr(model, 'N_assets', 1)
    pkl_path = os.path.join(REAL_VAL_DIR, f'{ds_name}_nopad_512.pkl')
    if not os.path.exists(pkl_path):
        print(f"    ⏭️ PKL missing: {pkl_path}")
        return None

    try:
        total_M = get_total_assets(pkl_path)
    except Exception as e:
        print(f"    ⏭️ PKL error: {e}")
        return None

    K = math.ceil(total_M / N)
    print(f"    📊 {ds_name}: {total_M} assets → {K} chunks of {N}, pred_len={pred_len}")

    all_train_dfs, all_pred_dfs = [], []
    n_ok = 0

    for k in range(K):
        start = k * N
        end = min((k + 1) * N, total_M)
        valid_len = end - start

        col_indices = list(range(start, end))
        if valid_len < N:
            # Circular-pad last chunk
            col_indices += [col_indices[i % valid_len] for i in range(N - valid_len)]
            print(f"      Chunk {k+1}/{K}: assets [{start}..{end-1}] + {N - valid_len} pad")
        elif k % max(1, K // 5) == 0 or k == K - 1:
            print(f"      Chunk {k+1}/{K}: assets [{start}..{end-1}]")

        result = evaluate_chunk(
            model, pkl_path, pred_len, CONTEXT_LEN,
            col_indices, SCALER, device, sub_day,
            chunk_label=f"[Chunk {k+1}/{K}]",
        )
        if result is not None:
            all_train_dfs.extend(result[0])
            all_pred_dfs.extend(result[1])
            n_ok += 1

    if not all_pred_dfs:
        print(f"    ❌ No valid predictions for {ds_name}")
        return None

    metrics = compute_metrics(all_train_dfs, all_pred_dfs, ds_name)
    print(f"    ✅ {ds_name}: MASE={metrics['mase']:.4f}, mCRPS={metrics['mcrps']:.4f} "
          f"({n_ok}/{K} chunks OK)")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"\n{'='*110}")
    print(f"🚀 SC-Mamba Gold-Standard Zero-Shot Benchmark (Multivariate Only)")
    print(f"   (17 Datasets Evaluated via Asset-dimension Chunking)")
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

        if not os.path.exists(ckpt_path):
            print(f"   ❌ Checkpoint not found! Skipping.")
            continue

        n_assets = resolve_n_assets(config_path)
        model_name = derive_model_name(ckpt_path)
        print(f"   N_assets:   {n_assets}")
        print(f"   Cache key:  {model_name}")
        print(f"{'─'*80}")

        model_labels.append(label)

        print(f"  🔬 [PATH B] Multivariate Asset-dimension Chunking (N={n_assets})...")
        try:
            model, sub_day = load_model(ckpt_path, config_path, DEVICE)
        except Exception as e:
            print(f"   ❌ Load failed: {e}")
            traceback.print_exc()
            for ds in TARGET_DATASETS:
                all_results.append({
                    'Model': label, 'Dataset': ds,
                    'MASE': np.nan, 'mCRPS': np.nan,
                })
            continue

        for ds_name, pred_len in TARGET_DATASETS.items():
            t0 = time.time()
            try:
                metrics = eval_chunked_dataset(model, ds_name, pred_len, DEVICE, sub_day)
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

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_results:
        print("\n❌ No results. Check checkpoints.")
        return

    # ── Build Comparison Table ────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    n_models = len(model_labels)
    col_w = 20
    ds_w = 30
    sep_w = ds_w + 3 + (col_w + 3) * n_models * 2

    print(f"\n\n{'='*sep_w}")
    print(f"📊 FULL BENCHMARK RESULTS")
    print(f"{'='*sep_w}")

    mase_hdr = "MASE".center((col_w + 3) * n_models - 3)
    mcrps_hdr = "mCRPS".center((col_w + 3) * n_models - 3)
    print(f"{'Dataset':<{ds_w}} | {mase_hdr} | {mcrps_hdr}")

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

    summary = df.groupby('Model')[['MASE', 'mCRPS']].agg(
        lambda x: np.nanmean(x) if x.notna().any() else np.nan
    ).rename(columns={'MASE': 'Avg MASE', 'mCRPS': 'Avg mCRPS'})
    summary = summary.reindex([lbl for lbl in model_labels if lbl in summary.index])

    print(f"\n📊 GLOBAL SUMMARY (Mean across {len(TARGET_DATASETS)} datasets):\n")
    print(summary.to_string(float_format='%.4f', na_rep='—'))
    print(f"\n{'='*sep_w}")
    print(f"✅ Multivariate Benchmark complete.\n")


# if __name__ == '__main__':
main()
