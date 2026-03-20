import torch
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import yaml
import argparse
try:
    import submitit
    HAS_SUBMITIT = True
except ImportError:
    HAS_SUBMITIT = False
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.time_feature.seasonality import get_seasonality
from data.data_provider.data_factory import data_provider
from utilsforecast.losses import mase, mae, smape, rmse
import csv
from core.models import SCMamba_Forecaster
from tqdm import tqdm
import time
from scipy.stats import norm as scipy_norm

REAL_DATASETS = {
    "nn5_daily_without_missing": 56,
    "nn5_weekly": 8,
    "covid_deaths": 30,
    "weather": 30,
    "hospital": 12,
    "fred_md": 12,
    "car_parts_without_missing": 12,
    "traffic": 24,
    "m3_monthly": 18,
    "ercot": 24,
    "m1_monthly": 18,
    "m1_quarterly": 8,
    "cif_2016": 12,
    "exchange_rate": 30,
    "m3_quarterly": 8,
    "tourism_monthly": 24,
    "tourism_quarterly": 8,
}

REAL_DATASET_ASSETS = {
    "nn5_daily_without_missing": 111,
    "nn5_weekly": 111,
    "covid_deaths": 108,
    "weather": 21,
    "hospital": 767,
    "fred_md": 107,
    "car_parts_without_missing": 2674,
    "traffic": 862,
    "m3_monthly": 1428,
    "ercot": 39,
    "m1_monthly": 617,
    "m1_quarterly": 203,
    "cif_2016": 72,
    "exchange_rate": 8,
    "m3_quarterly": 756,
    "tourism_monthly": 366,
    "tourism_quarterly": 427,
}

MAX_LENGTH = 512

# Fallback SSM config — used ONLY if neither checkpoint nor --config provides one.
# WARNING: mamba2 flag must match the architecture used during training.
DEFAULT_SSM_CONFIG = {
    "bidirectional":False,
    "enc_conv" : True,
    "init_dil_conv" : True,
    "enc_conv_kernel" : 5,
    "init_conv_kernel" : 5,
    "init_conv_max_dilation" : 3,
    "global_residual":False,
    "in_proj_norm":False,
    "initial_gelu_flag":True,
    "linear_seq":15,
    "mamba2":False,
    "norm":True,
    "norm_type":"layernorm",
    "num_encoder_layers":2,
    "d_state":128,
    "residual":False,
    "token_embed_len":1024,
}
DEFAULT_MODEL_NAME = "SCMamba_v1"


def resolve_ssm_config(checkpoint_data, config_yaml_path=None):
    """
    Resolve ssm_config with priority:
      1. From checkpoint (if 'ssm_config' key exists — future checkpoints)
      2. From training config YAML (if --config flag is given)
      3. Fallback to DEFAULT_SSM_CONFIG
    """
    # Priority 1: checkpoint
    if 'ssm_config' in checkpoint_data:
        print("  📦 ssm_config loaded from checkpoint")
        return checkpoint_data['ssm_config']

    # Priority 2: training config YAML
    if config_yaml_path and os.path.exists(config_yaml_path):
        with open(config_yaml_path) as f:
            train_config = yaml.load(f, yaml.loader.SafeLoader)
        if 'ssm_config' in train_config:
            print(f"  📄 ssm_config loaded from {config_yaml_path}")
            return train_config['ssm_config']

    # Priority 3: default
    print("  ⚠️  Using DEFAULT_SSM_CONFIG (no config in checkpoint or --config)")
    return DEFAULT_SSM_CONFIG

def set_queue(q_, log_folder, maximum_runtime=None):
    global ex
    global q
    if q_ == 'all':
        q = 'alldlc_gpu-rtx2080'
    if q_ == 'ml':
        q = 'mldlc_gpu-rtx2080'
    if q_ == 'mlhiwi':
        q = "mlhiwidlc_gpu-rtx2080"

    if maximum_runtime is None:
        if q == 'alldlc_gpu-rtx2080' or q == 'mlhiwidlc_gpu-rtx2080':
            maximum_runtime = 24*60*1-1
        else:
            maximum_runtime = 24*60*4-1

    ex = submitit.AutoExecutor(folder=log_folder)
    ex.update_parameters(timeout_min=maximum_runtime,
                        slurm_partition=q, #  mldlc_gpu-rtx2080
                        slurm_signal_delay_s=180, # time to pass the USR2 signal to slurm before the job times out so that it can finish the run
                        tasks_per_node=1,
                        nodes=1,
                        cpus_per_task=30, #24
                        mem_per_cpu=4096,
                        slurm_gres=f'gpu:{1}'
       )

    return maximum_runtime

def scale_data(output, scaler):
    # Scale both mu and sigma2
    if scaler == 'custom_robust':
        scaled_mu = (output['mu'] * output['scale'][1].squeeze(-1)) + output['scale'][0].squeeze(-1)
        scaled_sigma2 = output['sigma2'] * (output['scale'][1].squeeze(-1) ** 2)
    elif scaler == 'min_max':
        scaling_span = (output['scale'][0].squeeze(-1) - output['scale'][1].squeeze(-1))
        scaled_mu = (output['mu'] * scaling_span) + output['scale'][1].squeeze(-1)
        scaled_sigma2 = output['sigma2'] * (scaling_span ** 2)
    elif scaler == 'identity':
        scaled_mu = output['mu']
        scaled_sigma2 = output['sigma2']
        
    return scaled_mu, scaled_sigma2
    
def nll_eval(mu, sigma2, target):
    """
    Per-element Gaussian NLL used as a probabilistic quality metric at evaluation.
    Not used for gradient computation (eval only).
    """
    sigma2 = torch.clamp(sigma2, min=1e-6)
    loss = 0.5 * torch.log(torch.tensor(2 * np.pi) * sigma2) + 0.5 * ((target - mu) ** 2) / sigma2
    return loss


def crps_gaussian(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Closed-form CRPS for a Gaussian predictive distribution.

    CRPS(N(mu, sigma^2), y) = sigma * [
        (z * (2*Phi(z) - 1)) + 2*phi(z) - 1/sqrt(pi)
    ]
    where z = (y - mu) / sigma, Phi = Gaussian CDF, phi = Gaussian PDF.

    References
    ----------
    Gneiting & Raftery (2007), "Strictly Proper Scoring Rules, Prediction, and Estimation".
    Jordan et al. (2019), properscoring library closed-form derivation.

    Parameters
    ----------
    mu     : point forecast mean,  shape [...]
    sigma  : predictive std-dev (> 0), shape [...]
    y      : ground-truth observation, shape [...]

    Returns
    -------
    crps   : per-element CRPS scores, shape [...] (lower is better)
    """
    sigma = np.clip(sigma, a_min=1e-6, a_max=None)
    z = (y - mu) / sigma
    crps = sigma * (
        z * (2.0 * scipy_norm.cdf(z) - 1.0)
        + 2.0 * scipy_norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )
    return crps


def auto_regressive_predict(model, batch_x, batch_y, batch_x_mark, batch_y_mark, eval_pred_len, real_data_args, scaler, device):
        # decoder input
        dec_inp = torch.zeros_like(
            batch_y[-eval_pred_len:]).unsqueeze(0).int().to(device)

        x = {}
        x['ts'] = batch_x_mark.to(device)
        x['history'] = batch_x.reshape(1,batch_x.size(1)).to(device)
        outputs = []
        for pred_ind in range(0, eval_pred_len):
            x['target_dates'] = batch_y_mark[:, pred_ind].unsqueeze(1).to(device)
            x['task'] = dec_inp[:, pred_ind].unsqueeze(1)
            multipoint = isinstance(model, SCMamba_Forecaster)
            if multipoint:
                output = model(x, prediction_length=1)
            else:
                output = model(x)
                
            output = scale_data(output, scaler)
            outputs.append(output)
            
            x['history'] = torch.cat([x['history'], output], dim=1)
            x['ts'] = torch.cat([x['ts'], x['target_dates']], dim=1)
    
        outputs = torch.stack(outputs, dim=1).detach().cpu().squeeze()
            
        return outputs
    

def batch_predict(model, batch_x, batch_x_mark, batch_y_mark, eval_pred_len, scaler, device):
    x = {}
    x['ts'] = batch_x_mark.repeat(eval_pred_len, 1, 1).to(device)
    x['history'] = batch_x.reshape(1,batch_x.size(1)).repeat(eval_pred_len, 1).to(device)

    x['target_dates'] = batch_y_mark.transpose(0, 1).to(device)
    x['task'] = torch.zeros(eval_pred_len,1).int().to(device)

    output = model(x)
    scaled_mu, scaled_sigma2 = scale_data(output, scaler)
    
    mu_out = scaled_mu.detach().cpu().squeeze()
    sigma2_out = scaled_sigma2.detach().cpu().squeeze()
    
    return mu_out, sigma2_out

def multipoint_predict(model, batch_x, batch_x_mark, batch_y_mark, pred_len, scaler, device):
    x = {}
    x['ts'] = batch_x_mark.to(device)
    x['history'] = batch_x.reshape(1,batch_x.size(1)).to(device)
    x['target_dates'] = batch_y_mark.to(device)
    x['task'] = torch.zeros(1,pred_len).int().to(device)
    
    output = model(x, prediction_length=pred_len)
    scaled_mu, scaled_sigma2 = scale_data(output, scaler)
    
    mu_out = scaled_mu.detach().cpu().squeeze()
    sigma2_out = scaled_sigma2.detach().cpu().squeeze()
    return mu_out, sigma2_out


def efficient_ensemble_predict(model, batch_x, batch_x_mark, batch_y_mark, pred_len, scaler, device, ensemble_config):
    x = {}
    x['ts'] = batch_x_mark.to(device).repeat(ensemble_config["num_ensembles"], 1, 1)
    x['history'] = batch_x.reshape(1,batch_x.size(1)).to(device).repeat(ensemble_config["num_ensembles"], 1)
    x['target_dates'] = batch_y_mark.to(device).repeat(ensemble_config["num_ensembles"], 1, 1)
    x['task'] = torch.zeros(1,pred_len).int().to(device).repeat(ensemble_config["num_ensembles"], 1)
    
    cl = x["history"].size(1)
    
    drops = np.linspace(0,cl*ensemble_config["max_fraction"], ensemble_config["num_ensembles"]).astype(int)
    assert isinstance(model, SCMamba_Forecaster), "Model must be an instance of SCMamba_Forecaster"
  
    outputs = []
    z = x.copy()
    for i, drop in enumerate(drops):
        drop_indices = np.random.choice(cl, size=drop, replace=False)  # Randomly choose 'drop' indices to remove
        keep_indices = np.setdiff1d(np.arange(cl), drop_indices)
        z['ts'][i, drop:, :] = x['ts'][i, keep_indices, :]
        z['ts'][i, :drop, :] = x['ts'][i, keep_indices, :].min(dim=0, keepdim=True).values
        z['history'][i, drop:] = x['history'][i, keep_indices]
        z['history'][i, :drop] = x['history'][i, keep_indices].min(dim=0, keepdim=True).values
    output = model(z, pred_len)
    output = scale_data(output, scaler)
    outputs = output.detach().cpu().squeeze()
    if ensemble_config["method"] == "median":
        output = outputs.median(dim=0)
    else:
        output = outputs.mean(dim=0)
    output = output.detach().cpu().squeeze()
    return output


def ensemble_predict(model, batch_x, batch_x_mark, batch_y_mark, pred_len, scaler, device, ensemble_config):
    x = {}
    x['ts'] = batch_x_mark.to(device)
    x['history'] = batch_x.reshape(1,batch_x.size(1)).to(device)
    x['target_dates'] = batch_y_mark.to(device)
    x['task'] = torch.zeros(1,pred_len).int().to(device)
    
    cl = x["history"].size(1)
    
    drops = np.linspace(0,cl*ensemble_config["max_fraction"], ensemble_config["num_ensembles"]).astype(int)
    assert isinstance(model, SCMamba_Forecaster), "Model must be an instance of SCMamba_Forecaster"
  
    outputs = []
    z = {}
    for drop in drops:
        drop_indices = np.random.choice(cl, size=drop, replace=False)  # Randomly choose 'drop' indices to remove
        keep_indices = np.setdiff1d(np.arange(cl), drop_indices)
        z['ts'] = x['ts'][:, keep_indices, :]
        z['history'] = x['history'][:, keep_indices]
        z['target_dates'] = x['target_dates']
        z['task'] = x['task']
        output = model(z, pred_len)
        output = scale_data(output, scaler)
        output = output.detach().cpu().squeeze()
        outputs.append(output)
    outputs = torch.stack(outputs, dim=1)
    if ensemble_config["method"] == "median":
        output = outputs.median(dim=1)
    else:
        output = outputs.mean(dim=1)
    output = output.detach().cpu().squeeze()
    return output


def multivariate_predict_aligned(
    model,
    dataset: str,
    pred_len: int,
    scaler: str,
    device,
    sub_day: bool = False,
    context_len: int = None,
) -> tuple:
    """
    Feed ALL N_assets series (time-aligned) into a single forward pass.
    Only called when ``model.N_assets > 1`` and the dataset's asset count
    matches ``model.N_assets``.

    Returns
    -------
    batch_train_dfs, batch_pred_dfs : lists of DataFrames
        Same schema as produced by the univariate path in evaluate_real_dataset,
        so the calling code can concat and compute metrics identically.
    """
    import pickle
    from data.data_provider.multivariate_loader import MultivariateRealDataset

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir  = os.path.join(base_dir, '..', 'data', 'real_val_datasets')
    pkl_path  = os.path.join(data_dir, f'{dataset}_nopad_{MAX_LENGTH}.pkl')

    N = model.N_assets
    if context_len is None:
        context_len = MAX_LENGTH

    # Test split contains exactly 1 window (last pred_len timesteps as target)
    test_ds = MultivariateRealDataset(
        pkl_path, pred_len=pred_len, context_len=context_len,
        split='test', N_assets=N, sub_day=sub_day,
    )
    # Also grab the train split to compute MASE denominator
    train_ds = MultivariateRealDataset(
        pkl_path, pred_len=pred_len, context_len=context_len,
        split='train', N_assets=N, sub_day=sub_day,
    )

    model.eval()
    batch_train_dfs = []
    batch_pred_dfs  = []

    with torch.no_grad():
        for win_idx in range(len(test_ds)):
            sample = test_ds[win_idx]   # single window
            x    = sample['x'].to(device)    # (T_ctx, N)
            y    = sample['y'].to(device)    # (T_pred, N)
            ts_x = sample['ts_x'].to(device) # (T_ctx, ts_dim)
            ts_y = sample['ts_y'].to(device) # (T_pred, ts_dim)

            T_ctx  = x.shape[0]
            T_pred = y.shape[0]

            # Flatten to (N, T_ctx) for backbone
            history  = x.permute(1, 0)                                     # (N, T_ctx)
            ts_x_rep = ts_x.unsqueeze(0).expand(N, -1, -1)                # (N, T_ctx, ts_dim)
            ts_y_rep = ts_y.unsqueeze(0).expand(N, -1, -1)                # (N, T_pred, ts_dim)

            data = {
                'history'      : history,
                'ts'           : ts_x_rep,
                'target_dates' : ts_y_rep,
                'task'         : torch.zeros(N, T_pred, dtype=torch.int32, device=device),
            }

            output = model(data, prediction_length=T_pred)
            scaled_mu, scaled_sigma2 = scale_data(output, scaler)

            # mu/sigma2 shape: (N, T_pred)
            mu_np    = scaled_mu.detach().cpu().numpy()       # (N, T_pred)
            sig_np   = scaled_sigma2.detach().cpu().numpy()   # (N, T_pred)
            y_np     = y.cpu().numpy()                        # (T_pred, N)

            for asset_i in range(N):
                asset_id = f"{dataset}_asset_{asset_i}"
                # Training context for MASE denominator (last window available in train split)
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

                batch_pred_dfs.append(pd.DataFrame({
                    'id':       [asset_id] * T_pred,
                    'pred':     mu_np[asset_i],
                    'target':   y_np[:, asset_i],
                    'variance': sig_np[asset_i],
                    'nll':      nll_vals,
                }))

    return batch_train_dfs, batch_pred_dfs


class RobustZeroShotDataset(torch.utils.data.Dataset):
    """
    Robust multivariate dataset for zero-shot N>1 evaluation.

    Handles sparse/asynchronous PKLs gracefully:
      - Selects only col_indices columns from the pivot table.
      - Drops rows that are fully NaN (not series).
      - Applies ffill → bfill → fillna(0.0) to handle remaining gaps.
      - Stores self.values (full time-series array) for MASE denominator.

    Contrast with MultivariateRealDataset (used for training), which drops
    any series with >50% NaN timestamps. That strategy is correct for
    spectral training but fatal for evaluation on sparse datasets (e.g. weather
    has 3010 sub-daily series that appear sparse when pivoted to daily grid).

    Port of RobustZeroShotDataset from benchmark/14_eval_all.py.
    """

    def __init__(self, pkl_path, pred_len, context_len, split, col_indices, sub_day=False):
        import pickle as _pickle
        self.pred_len = pred_len
        self.context_len = context_len
        self.N_assets = len(col_indices)
        self.sub_day = sub_day

        with open(pkl_path, 'rb') as f:
            df_raw = _pickle.load(f)

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
        # Drop only fully-empty rows (not series) — preserves cardinality.
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

        # Guarantee at least 1 testing window for short/asynchronous datasets.
        ideal_train_end = T_total - n_test - min(
            max(pred_len, 30),
            max(pred_len, T_total - n_test - min_train_required)
        )
        train_end = max(ideal_train_end, min_train_required)

        if train_end > T_total:
            train_end = T_total

        if split == 'train':
            self._start = 0
            self._end = train_end
            self.n_windows = max(0, train_end - context_len - pred_len + 1)
        else:  # test
            test_target_start = T_total - n_test
            self._start = max(0, test_target_start - context_len)
            self._end = T_total
            self.n_windows = max(0, n_test - pred_len + 1)

        self._split = split
        self._test_target_start = T_total - n_test

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        if self._split == 'train':
            abs_start = self._start + idx
            ctx_end = abs_start + self.context_len
        else:
            target_start = self._test_target_start + idx
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


def get_total_assets(pkl_path: str) -> int:
    """
    Count total unique series in a PKL file without NaN filtering.
    Used to determine M for chunking — must match baseline cardinality exactly.
    """
    import pickle as _pickle
    with open(pkl_path, 'rb') as f:
        df = _pickle.load(f)
    return len(df.index.get_level_values('Series').unique())


def multivariate_predict_chunked(
    model,
    dataset: str,
    pred_len: int,
    scaler: str,
    device,
    sub_day: bool = False,
    context_len: int = None,
) -> tuple:
    """
    Asset-dimension Chunking: evaluate a multivariate model (N_assets > 1)
    on ANY dataset regardless of its number of series M.

    Algorithm:
      1. Count M via get_total_assets() — NO NaN filtering, preserving cardinality
         parity with univariate baselines.
      2. Split M assets into K = ceil(M/N) chunks of size N.
      3. Last chunk gets circular-padded if M % N != 0.
      4. Each chunk is loaded via RobustZeroShotDataset (ffill/bfill, no series drop).
      5. Forward pass each chunk, discard padded column predictions.
      6. MASE denominator uses full train-split history (train_ds.values[:,ai]),
         not a single sliding window — matching utilsforecast.losses.mase semantics.
      7. Concatenate all M real predictions for metric computation.

    Invariants:
      - M_eval == M_raw (no series dropped during evaluation).
      - MASE denominator is length-equivalent to the training series.
      - Output schema (batch_train_dfs, batch_pred_dfs) is identical whether
        using univariate_per_series or multivariate_chunked routes.
    """
    import math

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'data', 'real_val_datasets')

    config_file = os.path.join(base_dir, 'real_data_args.yaml')
    with open(config_file) as file:
        real_data_args = yaml.load(file, yaml.loader.SafeLoader)

    if real_data_args.get('pad', False):
        pkl_path = os.path.join(data_dir, f'{dataset}_pad_{MAX_LENGTH}.pkl')
    else:
        pkl_path = os.path.join(data_dir, f'{dataset}_nopad_{MAX_LENGTH}.pkl')

    N = model.N_assets
    if context_len is None:
        context_len = MAX_LENGTH

    # ── Step 1: Probe M without NaN filtering ──────────────────────────────
    # CRITICAL: must use raw series count, NOT MultivariateRealDataset which
    # drops series with >50% NaN. Dropping series breaks cardinality parity
    # with baselines that evaluate on full M assets.
    try:
        actual_M = get_total_assets(pkl_path)
    except Exception as e:
        print(f"  ⚠️ [Chunked Eval] Probe failed for {dataset}: {e}")
        return [], []

    batch_train_dfs = []
    batch_pred_dfs  = []
    model.eval()

    K = math.ceil(actual_M / N)
    print(f"  [eval] Chunked Multivariate: {dataset} has {actual_M} assets → {K} chunks of {N}.")

    with torch.no_grad():
        for k in range(K):
            start_idx = k * N
            end_idx   = min((k + 1) * N, actual_M)
            valid_len = end_idx - start_idx

            # Build col_indices for this chunk (circular-pad if last chunk is short)
            col_indices = list(range(start_idx, end_idx))
            if valid_len < N:
                pad_needed = N - valid_len
                col_indices += [col_indices[i % valid_len] for i in range(pad_needed)]

            # Use RobustZeroShotDataset: ffill/bfill, no series drop, self.values for MASE.
            try:
                test_ds  = RobustZeroShotDataset(
                    pkl_path, pred_len, context_len, 'test', col_indices, sub_day
                )
                train_ds = RobustZeroShotDataset(
                    pkl_path, pred_len, context_len, 'train', col_indices, sub_day
                )
            except Exception as e:
                print(f"  ⚠️ [Chunked Eval] Chunk {k} load failed: {e}")
                continue

            if len(test_ds) == 0:
                print(f"  ⚠️ [Chunked Eval] Chunk {k}: 0 test windows. Skipping.")
                continue

            for win_idx in range(len(test_ds)):
                sample = test_ds[win_idx]
                x    = sample['x'].to(device)     # (ctx, N)
                y    = sample['y'].to(device)      # (pred, N)
                ts_x = sample['ts_x'].to(device)  # (ctx, ts_dim)
                ts_y = sample['ts_y'].to(device)  # (pred, ts_dim)

                T_pred = y.shape[0]

                # Reshape for backbone: (N, ctx), (N, ctx, ts_dim), (N, pred, ts_dim)
                history  = x.permute(1, 0)
                ts_x_rep = ts_x.unsqueeze(0).expand(N, -1, -1)
                ts_y_rep = ts_y.unsqueeze(0).expand(N, -1, -1)

                data = {
                    'history'      : history,
                    'ts'           : ts_x_rep,
                    'target_dates' : ts_y_rep,
                    'task'         : torch.zeros(N, T_pred, dtype=torch.int32, device=device),
                }

                output = model(data, prediction_length=T_pred)
                scaled_mu, scaled_sigma2 = scale_data(output, scaler)

                mu_np  = scaled_mu.detach().cpu().numpy()     # (N, T_pred)
                sig_np = scaled_sigma2.detach().cpu().numpy() # (N, T_pred)
                y_np   = y.cpu().numpy()                      # (T_pred, N)

                # Only collect REAL assets (discard circular-padded duplicates)
                for asset_in_chunk in range(valid_len):
                    asset_i_global = start_idx + asset_in_chunk
                    asset_id = f"{dataset}_asset_{asset_i_global}"

                    # MASE denominator: full training series for this asset.
                    # train_ds.values contains the complete train-split value array.
                    # This is semantically equivalent to the per-series train context
                    # used by utilsforecast.losses.mase and matches univariate baseline protocol.
                    train_hist_i = train_ds.values[:, asset_in_chunk]
                    if len(train_hist_i) == 0:
                        train_hist_i = x[:, asset_in_chunk].cpu().numpy()  # fallback

                    batch_train_dfs.append(pd.DataFrame({
                        'id':     [asset_id] * len(train_hist_i),
                        'target': train_hist_i,
                    }))

                    nll_vals = nll_eval(
                        torch.tensor(mu_np[asset_in_chunk]),
                        torch.tensor(sig_np[asset_in_chunk]),
                        torch.tensor(y_np[:, asset_in_chunk]),
                    ).numpy()

                    batch_pred_dfs.append(pd.DataFrame({
                        'id':       [asset_id] * T_pred,
                        'pred':     mu_np[asset_in_chunk],
                        'target':   y_np[:, asset_in_chunk],
                        'variance': sig_np[asset_in_chunk],
                        'nll':      nll_vals,
                    }))

    print(f"  [eval] Chunked result: {len(batch_pred_dfs)} asset-prediction blocks collected.")
    return batch_train_dfs, batch_pred_dfs


def evaluate_real_dataset(dataset: str, model, scaler, context_len, eval_pred_len, device, pred_style=None, sub_day=None):
    # Use absolute path for real_data_args.yaml (same dir as this script)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(base_dir, 'real_data_args.yaml')
    
    with open(config_file) as file:
        real_data_args = yaml.load(file, yaml.loader.SafeLoader)

    if pred_style is None:
        pred_style = real_data_args['pred_style']
        
    if real_data_args['pad']:
        real_data_args['data_path'] = dataset + f'_pad_{MAX_LENGTH}.pkl'
    else:
        real_data_args['data_path'] = dataset + f'_nopad_{MAX_LENGTH}.pkl'

    pred_len = REAL_DATASETS[dataset]
    real_data_args['data'] = dataset
    real_data_args['pred_len'] = pred_len
    
    # Extract sub_day from model_name dynamically or explicit parameter
    if sub_day is None:
        sub_day = "subday" in real_data_args.get("model_name", "")
    test_dataset, test_dataloader = data_provider(real_data_args, real_data_args['flag'], subday=sub_day)

    gts_dataset = get_dataset(real_data_args['data'], regenerate=False)
    seasonality = get_seasonality(gts_dataset.metadata.freq) if gts_dataset.metadata.freq != 'D' else 7
    print(seasonality)

    # ── Model Path Router ───────────────────────────────────────────────────
    _model_n = getattr(model, 'N_assets', 1)
    _ds_n    = REAL_DATASET_ASSETS.get(dataset, 1)

    if _model_n > 1:
        if _model_n == _ds_n:
            print(f"  [eval] Multivariate EXACT-aligned eval: N_assets={_model_n}")
            batch_train_dfs, batch_pred_dfs = multivariate_predict_aligned(
                model=model, dataset=dataset, pred_len=pred_len,
                scaler=scaler, device=device, sub_day=sub_day,
                context_len=context_len,
            )
        else:
            batch_train_dfs, batch_pred_dfs = multivariate_predict_chunked(
                model=model, dataset=dataset, pred_len=pred_len,
                scaler=scaler, device=device, sub_day=sub_day,
                context_len=context_len,
            )
    else:
        # ── Original univariate path (num_assets=1) ───────────────────────
        batch_train_dfs = []
        batch_pred_dfs = []
        model.eval()
        j = 0
        print(f"pred_style: {pred_style}")
        with torch.no_grad():
            for i, batch in tqdm(enumerate(test_dataloader)):
                ids = batch["id"]
                batch_x = batch["x"].float()
                batch_y = batch["y"][:, -eval_pred_len:, :].squeeze().float()

                batch_x_mark = batch["ts_x"].float()
                batch_y_mark = batch["ts_y"][:, -eval_pred_len:, :].float()

                if pred_style == 'multipoint':
                    mu_out, sigma2_out = multipoint_predict(model, batch_x, batch_x_mark, batch_y_mark, eval_pred_len, scaler, device)
                else:
                    mu_out, sigma2_out = multipoint_predict(model, batch_x, batch_x_mark, batch_y_mark, eval_pred_len, scaler, device)

                batch_train_dfs.append(pd.DataFrame({
                    'id': ids.repeat_interleave(batch_x.size(1)).numpy(),
                    'target': batch_x.flatten().numpy()
                }))

                nll_vals = nll_eval(mu_out, sigma2_out, batch_y.cpu()).flatten().numpy()

                batch_pred_dfs.append(pd.DataFrame({
                    'id': ids.repeat_interleave(eval_pred_len).numpy(),
                    'pred': mu_out.flatten().numpy(),
                    'target': batch_y.flatten().numpy(),
                    'variance': sigma2_out.flatten().numpy(),
                    'nll': nll_vals
                }))

    train_df = pd.concat(batch_train_dfs)
    pred_df = pd.concat(batch_pred_dfs)

    # CRITICAL FIX for sparse datasets (cif_2016):
    # multivariate_loader pads NaN with 0.0/mean to prevent model explosion.
    # However, evaluating MASE/MAE against an artificial 0.0 target produces massive false errors.
    # We must identify which targets were originally missing. In this simplified pipeline,
    # we filter out exact 0.0 targets (which is the fallback fill value in loader) to
    # prevent dividing by zero or penalizing valid forecasts against blank labels.
    if dataset == 'cif_2016':
        train_df.loc[train_df['target'] == 0.0, 'target'] = np.nan
        pred_df.loc[pred_df['target'] == 0.0, 'target'] = np.nan

    # Drop NaNs BEFORE passing to metrics so they don't corrupt the aggregation
    pred_df_clean = pred_df.dropna(subset=['target'])
    train_df_clean = train_df.dropna(subset=['target'])

    # If completely empty after drop (edge case for tiny series), fallback to original
    if len(pred_df_clean) == 0:
        pred_df_clean = pred_df
    if len(train_df_clean) == 0:
        train_df_clean = train_df

    mase_loss = mase(pred_df_clean, ['pred'], seasonality, train_df_clean, 'id', 'target')
    mae_loss = mae(pred_df_clean, ['pred'], 'id', 'target')
    rmse_loss = rmse(pred_df_clean, ['pred'], 'id', 'target')
    smape_loss = smape(pred_df_clean, ['pred'], 'id', 'target')
    mean_nll = pred_df_clean['nll'].mean()

    # MASE: replace inf/-inf with NaN (caused by zero-variance training windows in
    # sparse datasets like car_parts, covid_deaths). Use skipna to get a valid mean.
    mase_vals = mase_loss['pred'].replace([float('inf'), float('-inf')], float('nan'))
    n_inf = int(mase_loss['pred'].isin([float('inf'), float('-inf')]).sum())
    if n_inf > 0:
        print(f"  ⚠️  MASE: {n_inf}/{len(mase_vals)} series have ∞ (zero-var training window) → excluded from mean")
    mase_mean = float(mase_vals.mean(skipna=True)) if mase_vals.notna().any() else float('nan')

    # CRPS (Continuous Ranked Probability Score) — closed-form Gaussian CRPS.
    # Primary probabilistic metric distinguishing SC-Mamba from deterministic baselines.
    mu_np = pred_df_clean['pred'].values
    sigma_np = np.sqrt(np.clip(pred_df_clean['variance'].values, 1e-6, None))
    y_np = pred_df_clean['target'].values
    crps_vals = crps_gaussian(mu_np, sigma_np, y_np)
    mean_crps = float(crps_vals.mean())

    # mCRPS (Mean-Scaled CRPS): Normalize by Mean Absolute Value of target data for cross-dataset comparison.
    # Raw CRPS is in original unit scale (e.g. 4M for CIF which has large values), standard deviation scaling is less stable.
    mean_abs_target = float(train_df_clean['target'].abs().mean()) if train_df_clean['target'].abs().mean() > 0 else 1.0
    mcrps = mean_crps / mean_abs_target

    out_dict = {
        'mase': mase_mean,
        'mase_has_inf': n_inf > 0,
        'mae': float(mae_loss['pred'].mean()),
        'rmse': float(rmse_loss['pred'].mean()),
        'smape': float(smape_loss['pred'].mean()),
        'nll': float(mean_nll),
        'crps': mean_crps,
        'mcrps': mcrps,
    }

    return out_dict, train_df, pred_df


def adapt_state_dict_keys(old_state_dict):
    new_state_dict = {}

    for key in old_state_dict.keys():
        if "linear_layer" in key:
            # Extract the layer index
            layer_idx = key.split('.')[1]
            
            # Replace "linear_layer" with "stage_2_layer.0"
            new_key = key.replace(f"linear_layer", f"stage_2_layer.0")
            
            # Add the updated key to the new state dict
            new_state_dict[new_key] = old_state_dict[key]
        else:
            # Keep other keys unchanged
            new_state_dict[key] = old_state_dict[key]

    return new_state_dict

def csv_writer(csv_file, result_dict):
    fieldnames = set()
    for entry in result_dict.values():
        fieldnames.update(entry.keys())
    # Convert set to list and sort or maintain any specific order if required
    fieldnames = list(fieldnames)
    fieldnames.insert(0, 'ID')
    # Open the file in write mode
    with open(csv_file, 'w', newline='') as csvfile:
        # Create a DictWriter object with the extended fieldnames
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header
        writer.writeheader()
        
        # Write the data
        for key, value in result_dict.items():
            # Insert the outer key into the row dictionary
            row = {'ID': key}
            row.update(value)
            writer.writerow(row)

def main_evaluator(pred_style=None, checkpoint_path=None, config_yaml_path=None, out_name=None):
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'real_data_args.yaml')
    with open(config_path) as file:
        real_data_args = yaml.load(file, yaml.loader.SafeLoader)
    real_data_args["model_name"] = os.path.basename(checkpoint_path).replace('.pth', '')

    if pred_style is None:
        pred_style = real_data_args['pred_style']

    # --- Checkpoint Validation ---
    print(f"\n{'='*60}")
    print(f"  CHECKPOINT")
    print(f"{'='*60}")
    print(f"  path   : {checkpoint_path}")
    print(f"  exists : {os.path.exists(checkpoint_path)}")

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"\n[ERROR] Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    model_string = checkpoint_path
    if out_name:
        model_name = out_name
    else:
        # Derive model name from filename for result directory naming
        model_name = os.path.basename(checkpoint_path).replace('.pth', '')
        # Strip suffixes for cleaner directory names (order matters: longest first)
        for suffix in ('_best_mase', '_best', '_Final'):
            if model_name.endswith(suffix):
                model_name = model_name[:-len(suffix)]
                break
    print(f"  model  : {model_name}")
    print(f"{'='*60}\n")

    eval_dir = os.path.join(base_dir, f'../data/real_data_evals/{model_name}/{pred_style}')
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    checkpoint_data = torch.load(model_string, map_location=device, weights_only=False)
    new_state_dict = adapt_state_dict_keys(checkpoint_data['model_state_dict'])

    # Resolve architecture config from checkpoint / YAML / default
    ssm_config = resolve_ssm_config(checkpoint_data, config_yaml_path)
    print(f"  🔧 mamba2 = {ssm_config.get('mamba2', 'N/A')}")
    print(f"  🔧 d_state = {ssm_config.get('d_state', 'N/A')}")
    print(f"  🔧 num_encoder_layers = {ssm_config.get('num_encoder_layers', 'N/A')}")

    # N_assets for model construction: must match training config, NOT dataset series count.
    # SC-Mamba trains channel-independent (num_assets=1 by default).
    # The SpectralVariationalLayer reshapes Z as [B, N_assets, P_L, D], so
    # using the dataset's actual series count (e.g. 111 for nn5_daily) when the
    # model was trained with N_assets=1 would produce B=0 → crash.
    if config_yaml_path and os.path.exists(config_yaml_path):
        with open(config_yaml_path) as f:
            _tc = yaml.load(f, yaml.loader.SafeLoader)
        n_assets = _tc.get('num_assets', 1)
        sub_day = _tc.get('sub_day', False)
    else:
        n_assets = checkpoint_data.get('num_assets', 1)
        sub_day = False
    print(f"  🔧 N_assets (model) = {n_assets}")
    print(f"  🔧 sub_day = {sub_day}")

    context_lens = [512]

    for dataset_name in REAL_DATASETS.keys():

        model = SCMamba_Forecaster(
            N_assets=n_assets,
            ssm_config=ssm_config
        ).to(device)
        model.load_state_dict(new_state_dict, strict=False)
        
        res_dict = {}
        print(f'pred_style: {pred_style}')
        yml_path = os.path.join(eval_dir, f'{dataset_name}_{MAX_LENGTH}.yml')
        if not os.path.exists(yml_path):
            pred_lens = [REAL_DATASETS[dataset_name]] #range(1, REAL_DATASETS[dataset_name]+1)
            for cl in context_lens:
                for pl in pred_lens:
                    print(f'evaluating {dataset_name} for context length:{cl} and prediction length:{pl}')
                    start_time = time.time()
                    out_dict, train_df, pred_df = evaluate_real_dataset(dataset_name, model, 'min_max', cl, pl, device, pred_style=pred_style, sub_day=sub_day)
                    end_time = time.time()
                    res_dict[f'{cl}_{pl}'] = out_dict
                    print(f"Time taken by {pred_style}: {end_time - start_time}")
                    #saving train and pred dfs if its max pred_len of entire series
                    if pl == REAL_DATASETS[dataset_name]:
                        train_df.to_csv(os.path.join(eval_dir, f'{dataset_name}_train_df_{cl}_{pl}.csv'), index=False)
                        pred_df.to_csv(os.path.join(eval_dir, f'{dataset_name}_pred_df_{cl}_{pl}.csv'), index=False)
        
            with open(yml_path, 'w') as outfile:
                yaml.dump(res_dict, outfile, default_flow_style=True)
            
            # Specify the file name
            filename = os.path.join(eval_dir, f'{dataset_name}_{MAX_LENGTH}.csv')

            # Write the dictionary to the CSV file
            csv_writer(filename, res_dict)
                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate SC-Mamba checkpoint on all 17 benchmark datasets.'
    )
    parser.add_argument("-c", "--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pth file")
    parser.add_argument("-cfg", "--config", type=str, default=None,
                        help="Path to training config YAML (for ssm_config)")
    parser.add_argument("-o", "--out_name", type=str, default=None,
                        help="Explicit output directory model name (bypasses _best/_Final stripping)")
    parser.add_argument("-s", "--slurm", type=bool, default=False,
                        help="Run on SLURM cluster")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    config_yaml_path = args.config

    if args.slurm:
        if not HAS_SUBMITIT:
            print("[ERROR] --slurm requires 'submitit'. pip install submitit", file=sys.stderr)
            sys.exit(1)
        script_basedir = os.path.dirname(os.path.abspath(__file__))
        log_folder = os.path.join(script_basedir, '../logs/')
        maximum_runtime = set_queue('mlhiwi', log_folder)
        submit_func = ex.submit
        job = submit_func(main_evaluator, checkpoint_path=checkpoint_path, config_yaml_path=config_yaml_path, out_name=args.out_name)
        print(job)
    else:
        print("Running on local machine")
        for pred_style in ['multipoint']:
            main_evaluator(pred_style, checkpoint_path, config_yaml_path, out_name=args.out_name)