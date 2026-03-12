"""
multivariate_loader.py
======================
Time-aligned Multivariate Dataset for SC-Mamba Cross-Asset Graph training.

Activated automatically when `config['num_assets'] > 1` and
`config.get('real_train_datasets')` is non-empty.

Design contract (zero data-leakage):
  - Data split is applied BEFORE any windowing.
  - Train/Val/Test boundaries mirror the GluonTS test convention:
      - Test  : last `pred_len` timesteps  (held out, matches eval_real_dataset.py)
      - Val   : `max(pred_len, 30)` steps before the test block
      - Train : everything before Val block
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

        # Forward-fill / back-fill sparse gaps (e.g., non-trading days)
        df_piv = df_piv.ffill().bfill()

        # Optionally limit number of assets
        if N_assets is not None:
            if df_piv.shape[1] < N_assets:
                raise ValueError(
                    f"Requested N_assets={N_assets} but dataset only has "
                    f"{df_piv.shape[1]} series."
                )
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

        # ── Data split — BEFORE windowing (no leakage) ───────────────────────
        n_test = pred_len
        n_val  = max(pred_len, 30)

        if split == 'train':
            end = T_total - n_test - n_val
        elif split == 'val':
            end = T_total - n_test
        else:  # test
            end = T_total

        # Earliest start a window can begin
        if split == 'train':
            start = 0
        elif split == 'val':
            start = T_total - n_test - n_val
        else:
            start = T_total - n_test  # only 1 full window for test

        self._start = start
        self._end   = end

        # Number of valid windows inside the split
        # A window needs context_len + pred_len timesteps
        n_steps = end - start
        self.n_windows = max(0, n_steps - context_len - pred_len + 1)

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
        'window_idx' : int                        — absolute start of window
        """
        abs_start = self._start + idx
        ctx_end   = abs_start + self.context_len
        tgt_end   = ctx_end + self.pred_len

        x    = self.values[abs_start : ctx_end]    # (ctx, N)
        y    = self.values[ctx_end   : tgt_end]    # (pred, N)
        ts_x = self.ts_feats[abs_start : ctx_end]  # (ctx, ts_dim)
        ts_y = self.ts_feats[ctx_end   : tgt_end]  # (pred, ts_dim)

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

    print(
        f"[MultivariateRealDataset] {dataset_name} | N_assets={N_assets} | "
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
