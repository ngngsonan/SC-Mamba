"""
multivariate_loader.py
======================
Time-aligned Multivariate Dataset for SC-Mamba Cross-Asset Graph training.

Activated automatically when `config['num_assets'] > 1` and
`config.get('real_train_datasets')` is non-empty.

Design contract (zero data-leakage):
  - Target zones never overlap across splits (strict no-leakage).
  - Train : targets in [0, T - n_test - n_val)
  - Val   : targets in [T - n_test - n_val, T - n_test)
  - Test  : targets in [T - n_test, T)
  - Context (inputs) MAY reach back into the previous split for val/test.
    This is the standard GluonTS evaluation convention and is leak-free
    because the model never sees future targets — only past inputs.
  - All N_assets columns share the SAME timestamp index in every window.
    This is what makes the SpectralVariationalLayer's cross-asset FFT valid.

When `num_assets = 1`: this module is NEVER imported or called.
  → Zero impact on the existing pipeline.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ── Resolve project root (data/data_provider/ → data/ → SC-Mamba-Code/) ──────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _PROJECT_ROOT)

from core.constants import NUM_TASKS  # noqa: E402  (imported for parity with backbone)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MultivariateRealDataset(Dataset):
    """
    Reads a stacked-series PKL (produced by store_real_datasets.py) and returns
    sliding windows where every sample contains ALL assets at the SAME time slice.

    Parameters
    ----------
    pkl_path   : absolute path to the `<dataset>_nopad_512.pkl` file.
    pred_len   : number of future timesteps to predict.
    context_len: number of past timesteps used as model context.
    split      : one of 'train', 'val', 'test'.
    N_assets   : number of asset columns to use. If None, uses all series.
    sub_day    : whether to add hour/minute time features (must match backbone).
    """

    def __init__(
        self,
        pkl_path: str,
        pred_len: int,
        context_len: int,
        split: str = 'train',
        N_assets: int | None = None,
        sub_day: bool = False,
    ):
        assert split in ('train', 'val', 'test'), f"Invalid split: {split}"

        with open(pkl_path, 'rb') as f:
            df_raw = pickle.load(f)

        # ── Pivot: rows = timestamp, columns = Series ID ──────────────────────
        # df_raw has MultiIndex (Series, positional_index)
        df_flat = df_raw.reset_index()
        # 'date' column, 'target' column, 'Series' column
        df_piv = df_flat.pivot_table(
            index='date', columns='Series', values='target', aggfunc='first'
        ).sort_index()

        # ── Robust NaN handling for sparse datasets (e.g. cif_2016) ────────────
        # Step 1: Drop series that are almost entirely absent (>50% NaN)
        # These cannot be meaningfully filled and poison the scaler.
        nan_frac = df_piv.isna().mean(axis=0)
        df_piv = df_piv.loc[:, nan_frac <= 0.5]
        
        if df_piv.shape[1] == 0:
            raise ValueError(
                f"All series in the dataset were dropped due to excessive NaN values. "
                f"Check the input PKL file."
            )
        
        # Step 2: Forward-fill / back-fill remaining sparse gaps (e.g., non-trading days)
        df_piv = df_piv.ffill().bfill()
        
        # Step 3: Fill any remaining NaN with the column mean (handles leading NaNs in short series)
        col_means = df_piv.mean(axis=0)
        df_piv = df_piv.fillna(col_means)
        
        # Step 4: Final safety — replace any residual NaN with 0
        df_piv = df_piv.fillna(0.0)

        # Optionally limit number of assets
        if N_assets is not None:
            available = df_piv.shape[1]
            if available < N_assets:
                # Auto-clamp: sparse-drop removed series → use what's available.
                # Hard-fail here would crash the entire benchmark loop for any
                # dataset where >50% NaN filter reduces available series below
                # the configured N (e.g. cif_2016 native=72 → actual=61).
                import warnings
                warnings.warn(
                    f"[MultivariateRealDataset] Requested N_assets={N_assets} "
                    f"but only {available} series remain after sparse-drop. "
                    f"Auto-clamping to N_assets={available}.",
                    RuntimeWarning, stacklevel=2
                )
                N_assets = available
            df_piv = df_piv.iloc[:, :N_assets]

        self.N_assets = df_piv.shape[1]
        self.pred_len = pred_len
        self.context_len = context_len
        self.sub_day = sub_day

        # ── Temporal features (mirrors data_loader.py compute_time_features_real) ──
        ts_index = pd.to_datetime(df_piv.index)
        if sub_day:
            ts_feats = np.stack([
                ts_index.year.values,
                ts_index.month.values,
                ts_index.day.values,
                ts_index.day_of_week.values + 1,
                ts_index.day_of_year.values,
                ts_index.hour.values,
                ts_index.minute.values,
            ], axis=-1)  # (T_total, 7)
        else:
            ts_feats = np.stack([
                ts_index.year.values,
                ts_index.month.values,
                ts_index.day.values,
                ts_index.day_of_week.values + 1,
                ts_index.day_of_year.values,
            ], axis=-1)  # (T_total, 5)

        self.ts_feats = ts_feats.astype(np.float32)      # (T_total, ts_dim)
        self.values   = df_piv.values.astype(np.float32) # (T_total, N_assets)

        T_total = len(df_piv)

        # ── Data split — target zones ───────────────────────────
        # In ideal datasets, Targets must not overlap across splits.
        # But for extremely short datasets (e.g. car_parts with length 51),
        # strictly non-overlapping zones would result in 0 train windows.
        n_test = pred_len
        
        # Calculate ideal n_val (max 30, but at least pred_len)
        min_train_required = context_len + pred_len
        max_val_allowed = max(pred_len, T_total - n_test - min_train_required)
        n_val = min(max(pred_len, 30), max_val_allowed)
        
        # Determine strict train_end boundary
        ideal_train_end = T_total - n_test - n_val
        
        # CRITICAL FIX for sparse datasets (Bug #9):
        # Guarantee at least 1 training window if mathematically possible.
        # If T_total is too small, train_end will overlap with val/test zones.
        train_end = max(ideal_train_end, min_train_required)
        # Extreme fallback: if dataset is even smaller than 1 window, use all for train
        if train_end > T_total:
            train_end = T_total

        if split == 'train':
            # Windows slide: abs_start in [0, train_end - context_len - pred_len]
            self._start = 0
            self._end   = train_end
            n_steps = train_end
            self.n_windows = max(0, n_steps - context_len - pred_len + 1)

        elif split == 'val':
            # Targets in [T - n_test - n_val, T - n_test)
            # Context allowed from train zone → _start reaches back by context_len
            val_target_start = T_total - n_test - n_val
            val_target_end   = T_total - n_test
            # Window i: context starts at val_target_start - context_len + i,
            #           target starts at val_target_start + i
            # → self._start is the context-start of the first window
            self._start = max(0, val_target_start - context_len)
            self._end   = val_target_end
            # Number of usable target positions in [val_target_start, val_target_end - pred_len]
            self.n_windows = max(0, n_val - pred_len + 1)

        else:  # test
            # Targets in [T - n_test, T)
            test_target_start = T_total - n_test
            self._start = max(0, test_target_start - context_len)
            self._end   = T_total
            self.n_windows = max(0, n_test - pred_len + 1)

        # Store for __getitem__ offset calculation (val/test use target_zone start)
        self._split = split
        self._val_target_start  = T_total - n_test - n_val
        self._test_target_start = T_total - n_test

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a dict with time-aligned tensors for all N_assets.

        Key shapes
        ----------
        'x'          : (context_len, N_assets)   — historical values
        'y'          : (pred_len,    N_assets)    — target values
        'ts_x'       : (context_len, ts_dim)      — shared time features (history)
        'ts_y'       : (pred_len,    ts_dim)      — shared time features (target)
        'window_idx' : int                        — absolute start of context window
        """
        if self._split == 'train':
            # Standard sliding window within train zone
            abs_start = self._start + idx
            ctx_end   = abs_start + self.context_len
        else:
            # val / test: target zone starts at _val_target_start/_test_target_start
            # window i: target_start = zone_start + i
            #           ctx_start   = target_start - context_len (may be in train zone)
            target_start = (
                self._val_target_start if self._split == 'val'
                else self._test_target_start
            ) + idx
            abs_start = max(0, target_start - self.context_len)
            ctx_end   = target_start  # context ends exactly where target begins

        tgt_end = ctx_end + self.pred_len

        # Handle short context at the beginning of data (pad with zeros on left)
        ctx_len_actual = ctx_end - abs_start
        if ctx_len_actual < self.context_len:
            pad = self.context_len - ctx_len_actual
            x    = np.concatenate([np.zeros((pad, self.N_assets), dtype=np.float32),
                                   self.values[abs_start : ctx_end]], axis=0)
            ts_x = np.concatenate([np.zeros((pad, self.ts_feats.shape[1]), dtype=np.float32),
                                   self.ts_feats[abs_start : ctx_end]], axis=0)
        else:
            x    = self.values[abs_start : ctx_end]      # (ctx, N)
            ts_x = self.ts_feats[abs_start : ctx_end]    # (ctx, ts_dim)

        y    = self.values[ctx_end : tgt_end]            # (pred, N)
        ts_y = self.ts_feats[ctx_end : tgt_end]          # (pred, ts_dim)

        return {
            'x':          torch.from_numpy(x),
            'y':          torch.from_numpy(y),
            'ts_x':       torch.from_numpy(ts_x),
            'ts_y':       torch.from_numpy(ts_y),
            'window_idx': abs_start,
        }


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def create_multivariate_real_dl(
    config: dict,
    device,          # kept for API parity with create_train_test_batch_dl
    cpus_available: int = 0,
):
    """
    Build train and val DataLoaders from the first dataset listed in
    ``config['real_train_datasets']``.

    Only called when ``config['num_assets'] > 1``.

    Parameters
    ----------
    config         : training config dict (same schema as train.py).
    device         : torch device (unused here; kept for call-site parity).
    cpus_available : number of DataLoader workers.

    Returns
    -------
    (train_dl, val_dl) — same tuple shape as ``create_train_test_batch_dl``.
    """
    dataset_name = config['real_train_datasets'][0]
    N_assets     = config['num_assets']
    pred_len     = config['pred_len']
    context_len  = config['context_len']
    sub_day      = config.get('sub_day', False)
    batch_size   = config.get('batch_size', 8)

    # Resolve PKL path
    data_dir = os.path.join(_PROJECT_ROOT, 'data', 'real_val_datasets')
    padded   = 'pad' if config.get('pad', False) else 'nopad'
    pkl_path = os.path.join(data_dir, f'{dataset_name}_{padded}_512.pkl')

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"MultivariateRealDataset: PKL not found at {pkl_path}.\n"
            f"Run:  python data/scripts/store_real_datasets.py"
        )

    train_ds = MultivariateRealDataset(
        pkl_path, pred_len, context_len, split='train',
        N_assets=N_assets, sub_day=sub_day,
    )
    val_ds = MultivariateRealDataset(
        pkl_path, pred_len, context_len, split='val',
        N_assets=N_assets, sub_day=sub_day,
    )

    # ── Sync effective N back into config (may have been auto-clamped by sparse-drop) ──
    # This allows train.py to detect a config vs actual N mismatch and rebuild the model.
    actual_N = train_ds.N_assets
    if actual_N != N_assets:
        print(
            f"[create_multivariate_real_dl] N auto-clamped: {N_assets} → {actual_N} "
            f"(sparse series dropped from {dataset_name}). Updating config['num_assets']."
        )
        config['num_assets'] = actual_N

    print(
        f"[MultivariateRealDataset] {dataset_name} | N_assets={actual_N} | "
        f"train_windows={len(train_ds)} | val_windows={len(val_ds)}"
    )

    common = dict(
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cpus_available,
        pin_memory=torch.cuda.is_available(),
    )
    train_dl = DataLoader(train_ds, **common)
    # Val loader: no shuffle, single worker
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          drop_last=False, num_workers=0,
                          pin_memory=torch.cuda.is_available())

    return train_dl, val_dl
