# @title 13_test_zeroshot_multi
"""
13_test_zeroshot_multi.py — v6 (Canonical Refactored)
=====================================================================
Zero-shot Foundation Model benchmark across GluonTS datasets.
Supports N=8 multivariate checkpoints with random sub-sampling.
=====================================================================
"""
import os, sys, yaml, warnings, subprocess
import numpy as np, pandas as pd, torch
from pathlib import Path

# --- DIRECTORY SETUP (LOCAL/MAC ADAPTIVE) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
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
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONTEXT_LEN = 256       
N_ASSETS    = 8         
UNI_ASSETS  = 1
SCALER      = 'min_max' 
SEEDS       = [7270, 860, 5390, 5191, 5734]

# Checkpoints path configurations
colab_ckpt = '/content/drive/MyDrive/Colab Notebooks/SCMamba/sc_mamba_checkpoints'
CKPT_DIR = colab_ckpt if os.path.exists(colab_ckpt) else os.path.join(PROJECT_ROOT, 'checkpoints')

MODEL_TO_TEST = [
    ('N=1 (Uni) v2 best_mase', os.path.join(CKPT_DIR, 'SCMamba_v2_17data_N_uni_best_mase.pth')),
    ('N=1 (Uni) v2 best_NLL', os.path.join(CKPT_DIR, 'SCMamba_v2_17data_N_uni_best.pth')),
    ('N=8 (Multi) v2 best_mase', os.path.join(CKPT_DIR, 'SCMamba_v2_multi_exchange_rate_best_mase.pth'))
]

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
# Ensure all Datasets are generated (especially for Colab environments)
# ─────────────────────────────────────────────────────────────────────────────
import subprocess
REAL_VAL_DIR = os.path.join(PROJECT_ROOT, 'data', 'real_val_datasets')
os.makedirs(REAL_VAL_DIR, exist_ok=True)
missing_ds = [ds for ds in TARGET_DATASETS if not os.path.exists(os.path.join(REAL_VAL_DIR, f'{ds}_nopad_512.pkl'))]

if missing_ds:
    print(f"\n🔄 Generating {len(missing_ds)} missing dataset(s) from GluonTS...")
    subprocess.run(['python', os.path.join(PROJECT_ROOT, 'data', 'scripts', 'store_real_datasets.py')])


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

    N_assets = UNI_ASSETS if 'uni' in os.path.basename(ckpt_path).lower() else ckpt.get('N_assets', N_ASSETS)
    model = SCMamba_Forecaster(N_assets=N_assets, ssm_config=ssm_config).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"  ✅ Loaded: {os.path.basename(ckpt_path)} | N_assets={model.N_assets}")
    return model

# ─────────────────────────────────────────────────────────────────────────────
# RobustZeroShotDataset for Asynchronous Coverage
# ─────────────────────────────────────────────────────────────────────────────
import pickle

class RobustZeroShotDataset(torch.utils.data.Dataset):
    """
    Local subclass specialized for Zero-Shot.
    1. Extracts specific `col_indices` (N=8).
    2. Performs 'Dense Time-Cropping': Finds the global min/max timestamp where AT LEAST ONE of the target series is active.
       (In purely asynchronous sets, requiring all 8 to be simultaneously active might yield 0 windows).
    3. Forward/Backward fills the remaining internal NaNs to ensure continuous signals for FFT.
    """
    def __init__(self, pkl_path: str, pred_len: int, context_len: int, split: str, col_indices: list, sub_day: bool = False):
        self.pred_len = pred_len
        self.context_len = context_len
        self.split = split
        self.N_assets = len(col_indices)
        self.sub_day = sub_day

        with open(pkl_path, 'rb') as f:
            df_raw = pickle.load(f)

        df_flat = df_raw.reset_index()
        df_piv = df_flat.pivot_table(index='date', columns='Series', values='target', aggfunc='first').sort_index()

        # Select only the specific 8 indices
        available = df_piv.shape[1]
        valid_idx = [i for i in col_indices if i < available]
        if len(valid_idx) < self.N_assets:
             raise ValueError(f"Requested {self.N_assets} assets, but only found {len(valid_idx)} valid indices.")
        
        df_sub = df_piv.iloc[:, valid_idx]

        # --- DENSE TIME-CROPPING & TELEMETRY ---
        # 1. Find the union of intervals where data is present
        # Drop rows where ALL 8 series are NaN (outside coverage)
        orig_len = len(df_sub)
        df_sub = df_sub.dropna(how='all')
        cropped_len = len(df_sub)

        # 2. Imputation Stats
        nan_count = df_sub.isna().sum().sum()
        total_cells = df_sub.size
        # Update log to intuitively explain the mathematical cropping process
        if total_cells > 0:
            print(f"      [RobustDS] Global Time Axis: {orig_len} -> 8-Asset Bounded Time: {cropped_len} (Trimmed {orig_len-cropped_len} empty edges). Inner NaNs (imputed): {nan_count}/{total_cells} ({nan_count/total_cells:.1%})")
        
        # Now, ffill/bfill for Spectral Soundness (FFT requires continuous signal)
        df_sub = df_sub.ffill().bfill().fillna(0.0)

        # Build Data Tensors
        ts_index = pd.to_datetime(df_sub.index)
        if sub_day:
            ts_feats = np.stack([ts_index.year.values, ts_index.month.values, ts_index.day.values, ts_index.day_of_week.values + 1, ts_index.day_of_year.values, ts_index.hour.values, ts_index.minute.values], axis=-1)
        else:
            ts_feats = np.stack([ts_index.year.values, ts_index.month.values, ts_index.day.values, ts_index.day_of_week.values + 1, ts_index.day_of_year.values], axis=-1)

        self.ts_feats = ts_feats.astype(np.float32)
        self.values = df_sub.values.astype(np.float32)
        # Store unpadded length to prevent metric corruption from zero-padding
        self.unpadded_len = cropped_len

        T_total = len(df_sub)
        n_test = pred_len
        min_train_required = context_len + pred_len
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
            val_target_end   = T_total - n_test
            self._start = max(0, val_target_start - context_len)
            self._end = val_target_end
            self.n_windows = max(0, n_val - pred_len + 1)
        else: # test
            test_target_start = T_total - n_test
            self._start = max(0, test_target_start - context_len)
            self._end = T_total
            self.n_windows = max(0, n_test - pred_len + 1)

        self._split = split
        self._val_target_start = T_total - n_test - n_val
        self._test_target_start = T_total - n_test

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> dict:
        if self._split == 'train':
            abs_start = self._start + idx
            ctx_end = abs_start + self.context_len
        else:
            target_start = (self._val_target_start if self._split == 'val' else self._test_target_start) + idx
            abs_start = max(0, target_start - self.context_len)
            ctx_end = target_start

        tgt_end = ctx_end + self.pred_len
        ctx_len_actual = ctx_end - abs_start

        if ctx_len_actual < self.context_len:
            pad = self.context_len - ctx_len_actual
            x = np.concatenate([np.zeros((pad, self.N_assets), dtype=np.float32), self.values[abs_start : ctx_end]], axis=0)
            ts_x = np.concatenate([np.zeros((pad, self.ts_feats.shape[1]), dtype=np.float32), self.ts_feats[abs_start : ctx_end]], axis=0)
        else:
            x = self.values[abs_start : ctx_end]
            ts_x = self.ts_feats[abs_start : ctx_end]

        y = self.values[ctx_end : tgt_end]
        ts_y = self.ts_feats[ctx_end : tgt_end]

        return {
            'x': torch.from_numpy(x), 'y': torch.from_numpy(y),
            'ts_x': torch.from_numpy(ts_x), 'ts_y': torch.from_numpy(ts_y),
            'window_idx': abs_start,
        }

# ─────────────────────────────────────────────────────────────────────────────
# Helper: canonical inference with subsetting
# ─────────────────────────────────────────────────────────────────────────────
def canonical_evaluate(
    model, pkl_path, pred_len, context_len, col_indices, scaler, device, sub_day=False,
):
    N = len(col_indices)
    
    # Use the Robust dataset that gracefully aligns asynchronous series via dense time-cropping
    try:
        test_ds = RobustZeroShotDataset(pkl_path, pred_len=pred_len, context_len=context_len, split='test', col_indices=col_indices, sub_day=sub_day)
        train_ds = RobustZeroShotDataset(pkl_path, pred_len=pred_len, context_len=context_len, split='train', col_indices=col_indices, sub_day=sub_day)
    except Exception as e:
        print(f"      [Robust Loader Error]: {e}")
        return None

    if len(test_ds) == 0: 
        print(f"      [Not enough overlap windows] Found 0 windows after cropping.")
        return None

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
            if getattr(model, 'N_assets', 1) == 1:
                # Univariate model: Evaluate each asset independently
                mu_list, sig_list = [], []
                for asset_i in range(N):
                    asset_data = {
                        'history': x[:, asset_i:asset_i+1].permute(1, 0),
                        'ts': ts_x.unsqueeze(0),
                        'target_dates': ts_y.unsqueeze(0),
                        'task': torch.zeros(1, T_pred, dtype=torch.int32, device=device),
                    }
                    asset_out = model(asset_data, prediction_length=T_pred)
                    a_mu, a_sig = scale_data(asset_out, scaler)
                    mu_list.append(a_mu); sig_list.append(a_sig)
                scaled_mu = torch.cat(mu_list, dim=0)
                scaled_sigma2 = torch.cat(sig_list, dim=0)
            else:
                # Multivariate model: Evaluate together
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
                
                # IMPROVEMENT: Use the full unpadded history for MASE denominator calculation
                # instead of just the last window (which might be mostly zero-padding)
                train_hist_i = train_ds.values[:, asset_i]
                
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

    # MASE, CRPS, mCRPS (Safeguarded against NaN from 0 variance/flatlines)
    # utilsforecast's mase can divide by 0 if the naive training error is exactly 0.0 (e.g. constant ffill). 
    # We add a tiny epsilon to avoid inf/nan.
    try:
        # Add epsilon to denominator to be absolutely sure we never divide by zero
        # utilsforecast doesn't provide epsilon natively, so we ensure train_df target is not all-zero
        mase_res = mase(pred_df, ['pred'], seasonality, train_df, 'id', 'target')
        # If MASE is inf/nan, fallback to MAE / (mean_abs_train + epsilon)
        mase_series = mase_res['pred'].replace([np.inf, -np.inf], np.nan)
        
        if mase_series.isna().any():
            for idx, val in mase_series.items():
                if np.isnan(val):
                    # Manual robust calculation for this ID
                    tdf_id = train_df[train_df['id'] == mase_res.loc[idx, 'id']]
                    pdf_id = pred_df[pred_df['id'] == mase_res.loc[idx, 'id']]
                    mae_id = np.mean(np.abs(pdf_id['target'].values - pdf_id['pred'].values))
                    y_train_id = tdf_id['target'].values
                    if len(y_train_id) > seasonality:
                        denom = np.mean(np.abs(y_train_id[seasonality:] - y_train_id[:-seasonality]))
                    else:
                        denom = 0.0 # Force fallback
                        
                    if np.isnan(denom) or denom < 1e-8:
                        denom = np.mean(np.abs(y_train_id)) + 1e-8
                    mase_series.at[idx] = mae_id / denom

        mase_mean = float(mase_series.mean(skipna=True))
        # fallback if everything is nan
        if np.isnan(mase_mean): mase_mean = float('nan')
    except Exception as e:
        print(f"      [MASE Error]: {e}")
        mase_mean = float('nan')
    
    raw_crps = float(pred_df['crps'].replace([np.inf, -np.inf], np.nan).mean(skipna=True))
    mean_abs_target = float(train_df['target'].abs().mean(skipna=True)) if train_df['target'].abs().mean(skipna=True) > 0 else 1.0
    mcrps = raw_crps / mean_abs_target

    return {
        'mase': mase_mean, 'mae': float(mae(pred_df, ['pred'], 'id', 'target')['pred'].mean()),
        'rmse': float(rmse(pred_df, ['pred'], 'id', 'target')['pred'].mean()),
        'smape': float(smape(pred_df, ['pred'], 'id', 'target')['pred'].mean()),
        'nll': float(pred_df['nll'].mean()), 'crps': raw_crps, 'mcrps': mcrps
    }

def get_total_assets(pkl_path):
    with open(pkl_path, 'rb') as f:
        df = pickle.load(f)
    return len(df.index.get_level_values('Series').unique())

def benchmark_dataset(ds_name, pred_len, model, device, seeds):
    # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    pkl_path = os.path.join(PROJECT_ROOT, 'data', 'real_val_datasets', f'{ds_name}_nopad_512.pkl')
    if not os.path.exists(pkl_path):
        print(f"  ⏭️ Skip: {ds_name} (PKL missing)")
        return None

    seed_results = []
    try:
        total_assets = get_total_assets(pkl_path)
    except Exception as e:
         print(f"  ⏭️ Skip: {ds_name} (Metadata read error: {e})")
         return None

    for seed in seeds:
        rng = np.random.default_rng(seed)
        
        # Intersection Retry Logic: Try to find a valid subset of 8 assets
        max_retries = 10
        valid_res = None
        
        for attempt in range(max_retries):
            try:
                col_idx = sorted(rng.choice(total_assets, size=N_ASSETS, replace=False)) if total_assets >= N_ASSETS else list(range(total_assets))
                valid_res = canonical_evaluate(model, pkl_path, pred_len, CONTEXT_LEN, col_idx, SCALER, device)
                if valid_res is not None:
                    break # Success!
            except Exception as e:
                pass # Try another slice
                
        if valid_res:
            seed_results.append(valid_res)
        else:
            print(f"    Seed {seed}: Failed to find valid overlapping sequence after {max_retries} retries.")
    
    if not seed_results: return None
    
    # Mean across seeds
    mase_m = np.nanmean([r['mase'] for r in seed_results])
    mcrps_m = np.nanmean([r['mcrps'] for r in seed_results])
    print(f"  📡 {ds_name:25s} | MASE={mase_m:.4f} | mCRPS={mcrps_m:.4f} (Avg across {len(seed_results)} seeds)")
    
    return {
        'dataset': ds_name, 'mase': mase_m, 'mcrps': mcrps_m,
        'mae': np.nanmean([r['mae'] for r in seed_results]),
        'nll': np.nanmean([r['nll'] for r in seed_results]),
        'count': len(seed_results)
    }

def run_full_uni_evaluation(model_name, ckpt_path):
    eval_dir = os.path.join(PROJECT_ROOT, 'data', 'real_data_evals', model_name, 'multipoint')
    os.makedirs(eval_dir, exist_ok=True)
    
    missing = []
    for ds in TARGET_DATASETS.keys():
        if not os.path.exists(os.path.join(eval_dir, f'{ds}_512.yml')):
            missing.append(ds)
            
    if missing:
        print(f"    [Full-Uni Eval] Phân tích toàn tập dữ liệu (Missing {len(missing)} caches) bằng eval_real_dataset.py...")
        # Lấy config cho Uni model (tạm thời lấy template config v_config06_uni_17data nếu có)
        config_path = os.path.join(PROJECT_ROOT, 'core', 'config.v_config06_uni_17data.yaml')
        if not os.path.exists(config_path):
             config_path = os.path.join(PROJECT_ROOT, 'core', 'config.yaml')
             
        cmd = [
            'python', os.path.join(PROJECT_ROOT, 'core', 'eval_real_dataset.py'),
            '-c', ckpt_path,
            '-o', model_name,
            '-cfg', config_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"    ⚠️ Warning: eval_real_dataset.py lỗi:\n{res.stderr}")
            return None
    return eval_dir

def read_full_uni_metrics(eval_dir, ds_name):
    yml_path = os.path.join(eval_dir, f'{ds_name}_512.yml')
    if os.path.exists(yml_path):
        with open(yml_path) as f:
            raw = yaml.safe_load(f)
            if raw:
                metrics = next(iter(raw.values()), {})
                m_mase = metrics.get('mase', float('nan'))
                m_crps = metrics.get('mcrps')
                if m_crps is None:
                    m_crps = metrics.get('crps_scaled', float('nan'))
                return {
                    'mase_full': float(m_mase),
                    'mcrps_full': float(m_crps),
                }
    return {'mase_full': float('nan'), 'mcrps_full': float('nan')}

def main():
    print(f"\n{'='*110}\n🚀 SC-Mamba Zero-Shot Comparison: N=1 (Full & Subset 8) vs N=8 (Subset 8)\n{'='*110}")
    
    all_summaries = []

    for label, path in MODEL_TO_TEST:
        print(f"\n▶️ Testing {label}...")
        try:
            model = load_model_from_checkpoint(path, DEVICE)
        except Exception as e:
            print(f"❌ Load error for {label}: {e}")
            continue

        # Kiểm tra xem đây có phải là Univariate model không (N_assets = 1)
        is_uni = getattr(model, 'N_assets', 1) == 1
        full_uni_eval_dir = None
        if is_uni:
            model_name = os.path.basename(path).replace('.pth', '')
            full_uni_eval_dir = run_full_uni_evaluation(model_name, path)

        results = []
        for ds, pl in TARGET_DATASETS.items():
            res = benchmark_dataset(ds, pl, model, DEVICE, SEEDS)
            if res: 
                res['model'] = label
                
                # Nạp thêm Full metrics nếu là Uni model
                if is_uni and full_uni_eval_dir:
                    full_mets = read_full_uni_metrics(full_uni_eval_dir, ds)
                    res.update(full_mets)
                else:
                    res.update({'mase_full': np.nan, 'mcrps_full': np.nan})
                    
                results.append(res)

        if results:
            df_m = pd.DataFrame(results)
            all_summaries.append(df_m)
            print(f"✅ {label} complete. Avg MASE: {df_m['mase'].mean():.4f}")

    if not all_summaries:
        print("No results generated.")
        return

    # Merge and Print Final Comparison
    final_df = pd.concat(all_summaries)

    pd.options.display.float_format = '{:,.4f}'.format
    
    print(f"\n{'='*125}")
    print(f"{'Dataset':<26} {'Model':<26} | {'MASE (Subset 8)':>15} {'mCRPS (Subset 8)':>16} | {'MASE (Full All)':>15} {'mCRPS (Full All)':>16}")
    print(f"{'─'*125}")
    
    pivot_mase_sub = final_df.pivot(index='dataset', columns='model', values='mase')
    pivot_mcrps_sub = final_df.pivot(index='dataset', columns='model', values='mcrps')

    for ds in TARGET_DATASETS.keys():
        if ds in pivot_mase_sub.index:
            for lbl, _ in MODEL_TO_TEST:
                # Trích xuất dữ liệu của model & dataset
                row_data = final_df[(final_df['dataset'] == ds) & (final_df['model'] == lbl)]
                if not row_data.empty:
                    row = row_data.iloc[0]
                    
                    m_sub = row['mase']
                    c_sub = row['mcrps']
                    m_full = row.get('mase_full', np.nan)
                    c_full = row.get('mcrps_full', np.nan)
                    
                    m_sub_str = f"{m_sub:>15.4f}" if not pd.isna(m_sub) else f"{'-':>15}"
                    c_sub_str = f"{c_sub:>16.4f}" if not pd.isna(c_sub) else f"{'-':>16}"
                    
                    m_f_str = f"{m_full:>15.4f}" if not pd.isna(m_full) else f"{'-':>15}"
                    c_f_str = f"{c_full:>16.4f}" if not pd.isna(c_full) else f"{'-':>16}"
                    
                    print(f"{ds:<26} {lbl:<26} | {m_sub_str} {c_sub_str} | {m_f_str} {c_f_str}")
            print(f"{'─'*125}")

    # Global Average Summary
    summary = final_df.groupby('model').agg({
        'mase': 'mean', 
        'mcrps': 'mean', 
        'mase_full': lambda x: np.nanmean(x) if x.notna().any() else np.nan,
        'mcrps_full': lambda x: np.nanmean(x) if x.notna().any() else np.nan
    }).rename(columns={'mase': 'MASE (Sub8)', 'mcrps': 'mCRPS (Sub8)', 'mase_full': 'MASE (Full)', 'mcrps_full': 'mCRPS (Full)'})
    
    print(f"\nGLOBAL SUMMARY (Averaged across datasets):")
    print(summary.to_string(float_format='%.4f', na_rep='-'))
    print(f"{'='*125}\n✅ Multi-checkpoint & Uni/Multi Validation benchmark complete.")


# if __name__ == "__main__":
main()

