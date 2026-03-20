"""
eval_unified.py — Gold-Standard SC-Mamba Benchmark
====================================================
Single script comparing SC-Mamba (N=1, N>1) against SOTA baselines
on the 17 GluonTS benchmark datasets.

Evaluation Protocol:
  - N=1 (Univariate):  Per-series via Dataset_GluonTS.
                       IDENTICAL to Mamba4Cast / Chronos / MOIRAI protocol.
  - N>1, N==M:         multivariate_predict_aligned (in-distribution only).
  - N>1, N≠M:          Asset-dimension Chunking via MultivariateRealDataset.
                       K=⌈M_eff/N⌉ forward passes → M_eff predictions.

All three branches produce DataFrames with identical schema
→ unified metric computation (MASE, MAE, RMSE, sMAPE, NLL, CRPS, mCRPS).

Apple-to-Apple Guarantee (see implementation_plan.md §4):
  1. Identical test split: last pred_len timesteps per series.
  2. Identical cardinality: M predictions on M series.
  3. Identical MASE formula: utilsforecast.losses.mase (seasonal naive denom).
  4. Same data source: GluonTS PKL files.
  5. Same loader for train and eval (MultivariateRealDataset → no distribution shift).

Usage:
  python benchmark/eval_unified.py -c <checkpoint.pth> -cfg <config.yaml>
  python benchmark/eval_unified.py -c <checkpoint.pth> -cfg <config.yaml> --debug
"""

import os
import sys
import time
import logging
import traceback
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import yaml

# ── Resolve project root ────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
sys.path.insert(0, _PROJECT_ROOT)

from core.eval_real_dataset import (
    evaluate_real_dataset,
    REAL_DATASETS,
    REAL_DATASET_ASSETS,
    resolve_ssm_config,
    adapt_state_dict_keys,
    MAX_LENGTH,
)
from core.models import SCMamba_Forecaster


# ═════════════════════════════════════════════════════════════════════════════
# Logger Setup
# ═════════════════════════════════════════════════════════════════════════════

def setup_logger(debug: bool, out_dir: str) -> logging.Logger:
    """
    Configure dual-output logger.
    Console: INFO+ (always visible).
    File:    DEBUG+ (only when --debug, full trace).
    """
    logger = logging.getLogger('eval_unified')
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    # Avoid duplicate handlers on repeat calls
    logger.handlers.clear()

    # Console handler — clean format
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

    # File handler — timestamped, DEBUG level
    if debug:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(out_dir, f'eval_unified_debug_{ts}.log')
        fh = logging.FileHandler(log_path, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)-5s] %(message)s',
            datefmt='%H:%M:%S',
        ))
        logger.addHandler(fh)
        logger.info(f"  📝 Debug log → {log_path}")

    return logger


# ═════════════════════════════════════════════════════════════════════════════
# Model Loading
# ═════════════════════════════════════════════════════════════════════════════

def load_checkpoint(
    ckpt_path: str,
    config_yaml_path: str,
    device: torch.device,
    logger: logging.Logger,
) -> tuple:
    """
    Load SC-Mamba model from checkpoint.

    Returns
    -------
    (model, scaler, sub_day, n_assets)
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"  LOADING CHECKPOINT")
    logger.info(f"{'='*70}")
    logger.info(f"  ckpt  : {ckpt_path}")
    logger.info(f"  config: {config_yaml_path}")

    if not os.path.exists(ckpt_path):
        logger.error(f"  ❌ Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    checkpoint_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = adapt_state_dict_keys(checkpoint_data['model_state_dict'])

    # ── SSM config resolution (checkpoint → YAML → default) ────────────────
    ssm_config = resolve_ssm_config(checkpoint_data, config_yaml_path)

    # ── Training config for N_assets, scaler, sub_day ──────────────────────
    if config_yaml_path and os.path.exists(config_yaml_path):
        with open(config_yaml_path) as f:
            train_config = yaml.load(f, yaml.SafeLoader)
        n_assets = train_config.get('num_assets', 1)
        scaler = train_config.get('scaler', 'min_max')
        sub_day = train_config.get('sub_day', False)
    else:
        n_assets = checkpoint_data.get('num_assets', 1)
        scaler = 'min_max'
        sub_day = False

    logger.info(f"  N_assets : {n_assets}")
    logger.info(f"  scaler   : {scaler}")
    logger.info(f"  sub_day  : {sub_day}")
    logger.info(f"  mamba2   : {ssm_config.get('mamba2', False)}")
    logger.info(f"  d_state  : {ssm_config.get('d_state', '?')}")
    logger.debug(f"  [DEBUG] Full ssm_config: {ssm_config}")

    model = SCMamba_Forecaster(N_assets=n_assets, ssm_config=ssm_config).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  params   : {n_params:,}")
    logger.info(f"{'='*70}\n")

    return model, scaler, sub_day, n_assets


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation Loop
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_all_datasets(
    model: SCMamba_Forecaster,
    device: torch.device,
    scaler: str,
    sub_day: bool,
    logger: logging.Logger,
    context_len: int = 512,
    datasets: dict = None,
) -> pd.DataFrame:
    """
    Evaluate model on all 17 GluonTS benchmark datasets.

    Internally calls evaluate_real_dataset() which auto-routes:
      - N=1  → per-series (Dataset_GluonTS)
      - N>1, N==M → multivariate_predict_aligned
      - N>1, N≠M → multivariate_predict_chunked (Asset-dimension Chunking)

    Returns
    -------
    pd.DataFrame with columns:
      [dataset, pred_len, M_raw, N_model, route, mase, mae, rmse, smape,
       nll, crps, mcrps, time_sec, status]
    """
    if datasets is None:
        datasets = REAL_DATASETS

    n_model = getattr(model, 'N_assets', 1)
    results = []

    total = len(datasets)
    for idx, (ds_name, pred_len) in enumerate(datasets.items(), 1):
        M_raw = REAL_DATASET_ASSETS.get(ds_name, '?')

        # ── Determine expected route for debug ──────────────────────────────
        if n_model == 1:
            route = 'univariate_per_series'
        elif n_model == M_raw:
            route = 'multivariate_aligned'
        else:
            route = 'multivariate_chunked'

        logger.info(f"  [{idx:2d}/{total}] {ds_name:<30s}  P={pred_len:<3d}  M={str(M_raw):<5s}  → {route}")
        logger.debug(f"[ROUTE] {ds_name} → N_model={n_model}, M_raw={M_raw}, route={route}")

        t0 = time.time()
        try:
            out_dict, train_df, pred_df = evaluate_real_dataset(
                dataset=ds_name,
                model=model,
                scaler=scaler,
                context_len=context_len,
                eval_pred_len=pred_len,
                device=device,
                pred_style='multipoint',
                sub_day=sub_day,
            )
            elapsed = time.time() - t0

            # ── Debug: log detailed metrics ─────────────────────────────────
            logger.debug(
                f"[METRIC] {ds_name}: "
                f"mase={out_dict.get('mase', 'N/A'):.4f}  "
                f"mae={out_dict.get('mae', 'N/A'):.4f}  "
                f"rmse={out_dict.get('rmse', 'N/A'):.4f}  "
                f"smape={out_dict.get('smape', 'N/A'):.4f}  "
                f"nll={out_dict.get('nll', 'N/A'):.4f}  "
                f"crps={out_dict.get('crps', 'N/A'):.4f}  "
                f"mcrps={out_dict.get('mcrps', 'N/A'):.4f}"
            )
            logger.debug(
                f"[PRED] {ds_name}: "
                f"train_df.shape={train_df.shape}, "
                f"pred_df.shape={pred_df.shape}, "
                f"unique_ids={pred_df['id'].nunique()}"
            )
            if out_dict.get('mase_has_inf', False):
                n_series = pred_df['id'].nunique() if pred_df is not None else '?'
                logger.debug(f"[MASE_INF] {ds_name}: some series had ∞ MASE (constant training window)")

            # ── Determine M_eval (how many series were actually evaluated) ──
            M_eval = pred_df['id'].nunique() if pred_df is not None and len(pred_df) > 0 else 0

            results.append({
                'dataset': ds_name,
                'pred_len': pred_len,
                'M_raw': M_raw,
                'M_eval': M_eval,
                'N_model': n_model,
                'route': route,
                'mase': out_dict.get('mase', float('nan')),
                'mae': out_dict.get('mae', float('nan')),
                'rmse': out_dict.get('rmse', float('nan')),
                'smape': out_dict.get('smape', float('nan')),
                'nll': out_dict.get('nll', float('nan')),
                'crps': out_dict.get('crps', float('nan')),
                'mcrps': out_dict.get('mcrps', float('nan')),
                'time_sec': round(elapsed, 1),
                'status': 'OK',
            })

            logger.info(
                f"           MASE={out_dict.get('mase', float('nan')):.4f}  "
                f"mCRPS={out_dict.get('mcrps', float('nan')):.4f}  "
                f"M_eval={M_eval}  "
                f"({elapsed:.1f}s)"
            )

        except Exception as e:
            elapsed = time.time() - t0
            logger.info(f"           ❌ FAIL: {e}  ({elapsed:.1f}s)")
            logger.debug(f"[SKIP] {ds_name}: {traceback.format_exc()}")

            results.append({
                'dataset': ds_name,
                'pred_len': pred_len,
                'M_raw': M_raw,
                'M_eval': 0,
                'N_model': n_model,
                'route': route,
                'mase': float('nan'),
                'mae': float('nan'),
                'rmse': float('nan'),
                'smape': float('nan'),
                'nll': float('nan'),
                'crps': float('nan'),
                'mcrps': float('nan'),
                'time_sec': round(elapsed, 1),
                'status': f'FAIL: {e}',
            })

    return pd.DataFrame(results)


# ═════════════════════════════════════════════════════════════════════════════
# Output Formatters
# ═════════════════════════════════════════════════════════════════════════════

def print_results_table(df: pd.DataFrame, label: str, logger: logging.Logger):
    """Print formatted console summary with averages."""
    logger.info(f"\n{'═'*100}")
    logger.info(f"📊 SC-Mamba Unified Benchmark — {label}")
    logger.info(f"{'═'*100}")
    logger.info(
        f"  {'Dataset':<30s}  {'M_raw':>5s}  {'M_eval':>6s}  "
        f"{'MASE':>8s}  {'mCRPS':>8s}  {'MAE':>10s}  {'sMAPE':>8s}  "
        f"{'NLL':>8s}  {'Time':>6s}  {'Status':<12s}"
    )
    logger.info(f"{'─'*100}")

    for _, row in df.iterrows():
        status = row['status']
        is_training_src = row['dataset'] == 'exchange_rate' and row['N_model'] > 1

        if status == 'OK':
            mase_str = f"{row['mase']:.4f}" if not np.isnan(row['mase']) else '—'
            mcrps_str = f"{row['mcrps']:.4f}" if not np.isnan(row['mcrps']) else '—'
            mae_str = f"{row['mae']:.4f}" if not np.isnan(row['mae']) else '—'
            smape_str = f"{row['smape']:.4f}" if not np.isnan(row['smape']) else '—'
            nll_str = f"{row['nll']:.4f}" if not np.isnan(row['nll']) else '—'
            marker = ' †' if is_training_src else ''
            logger.info(
                f"  {row['dataset'] + marker:<30s}  {str(row['M_raw']):>5s}  {row['M_eval']:>6d}  "
                f"{mase_str:>8s}  {mcrps_str:>8s}  {mae_str:>10s}  {smape_str:>8s}  "
                f"{nll_str:>8s}  {row['time_sec']:>5.1f}s  {'OK':<12s}"
            )
        else:
            logger.info(
                f"  {row['dataset']:<30s}  {str(row['M_raw']):>5s}  {'—':>6s}  "
                f"{'—':>8s}  {'—':>8s}  {'—':>10s}  {'—':>8s}  "
                f"{'—':>8s}  {row['time_sec']:>5.1f}s  {status[:12]:<12s}"
            )

    logger.info(f"{'─'*100}")

    # ── Averages (exclude failed + in-dist training source) ────────────────
    ok_df = df[df['status'] == 'OK'].copy()
    # For N>1, exclude exchange_rate (training source) from zero-shot average
    n_model = df['N_model'].iloc[0] if len(df) > 0 else 1
    if n_model > 1:
        zs_df = ok_df[ok_df['dataset'] != 'exchange_rate']
    else:
        zs_df = ok_df

    if len(zs_df) > 0:
        avg_mase = zs_df['mase'].mean(skipna=True)
        avg_mcrps = zs_df['mcrps'].mean(skipna=True)
        avg_mae = zs_df['mae'].mean(skipna=True)
        avg_smape = zs_df['smape'].mean(skipna=True)
        avg_nll = zs_df['nll'].mean(skipna=True)
        n_ok = len(zs_df)
        logger.info(
            f"  {'Avg (zero-shot, '+ str(n_ok) +'/' + str(len(df)) + ' DS)':<30s}  "
            f"{'':>5s}  {'':>6s}  "
            f"{avg_mase:>8.4f}  {avg_mcrps:>8.4f}  {avg_mae:>10.4f}  {avg_smape:>8.4f}  "
            f"{avg_nll:>8.4f}"
        )

    logger.info(f"{'═'*100}")

    if n_model > 1:
        logger.info(f"  † Training source (in-distribution). Excluded from zero-shot average.")

    n_skip = len(df[df['status'] != 'OK'])
    if n_skip > 0:
        logger.info(f"  ⚠️  {n_skip} dataset(s) skipped (NaN-drop eliminated all series or load failure).")

    logger.info("")


def generate_latex_table(df: pd.DataFrame, label: str) -> str:
    """Generate LaTeX tabular for paper insertion."""
    lines = [
        r"\begin{tabular}{l r r r r r r r}",
        r"\toprule",
        r"Dataset & $P$ & $M$ & MASE $\downarrow$ & mCRPS $\downarrow$ & MAE $\downarrow$ & sMAPE $\downarrow$ & NLL $\downarrow$ \\",
        r"\midrule",
    ]

    n_model = df['N_model'].iloc[0] if len(df) > 0 else 1
    for _, row in df.iterrows():
        if row['status'] != 'OK':
            lines.append(
                f"{row['dataset']} & {row['pred_len']} & {row['M_raw']} & — & — & — & — & — \\\\"
            )
            continue

        is_training = row['dataset'] == 'exchange_rate' and n_model > 1
        name = row['dataset'].replace('_', r'\_')
        if is_training:
            name += r'$^\dagger$'

        def fmt(v):
            return f"{v:.4f}" if not np.isnan(v) else '—'

        lines.append(
            f"{name} & {row['pred_len']} & {row['M_eval']} & "
            f"{fmt(row['mase'])} & {fmt(row['mcrps'])} & "
            f"{fmt(row['mae'])} & {fmt(row['smape'])} & {fmt(row['nll'])} \\\\"
        )

    # Average row
    ok_df = df[df['status'] == 'OK']
    if n_model > 1:
        zs_df = ok_df[ok_df['dataset'] != 'exchange_rate']
    else:
        zs_df = ok_df

    if len(zs_df) > 0:
        lines.append(r"\midrule")
        def avgfmt(col):
            v = zs_df[col].mean(skipna=True)
            return f"{v:.4f}" if not np.isnan(v) else '—'

        n = len(zs_df)
        lines.append(
            f"\\textbf{{Avg ({n} DS)}} & & & "
            f"{avgfmt('mase')} & {avgfmt('mcrps')} & "
            f"{avgfmt('mae')} & {avgfmt('smape')} & {avgfmt('nll')} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])

    if n_model > 1:
        lines.append(r"% $\dagger$ = training source (in-distribution)")

    return '\n'.join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Main Batch Runner
# ═════════════════════════════════════════════════════════════════════════════

def main():
    # ── Path configuration ──────────────────────────────────────────────────
    colab_ckpt = '/content/drive/MyDrive/Colab Notebooks/SCMamba/sc_mamba_checkpoints'
    ckpt_dir = colab_ckpt if os.path.exists(colab_ckpt) else os.path.join(_PROJECT_ROOT, 'checkpoints')
    core_dir = os.path.join(_PROJECT_ROOT, 'core')

    # ── Evaluation Configurations ───────────────────────────────────────────
    # Format: (output_label, checkpoint_path, config_yaml_path)
    MODELS_TO_TEST = [
        (
            'SC-Mamba_N1',
            os.path.join(ckpt_dir, 'SCMamba_v2_17data_N_uni_best_mase.pth'),
            os.path.join(core_dir, 'config.based_setup.yaml'),
        ),
        (
            'SC-Mamba_N8_Chunked',
            os.path.join(ckpt_dir, 'SCMamba_v3_multi_exchange_rate_best.pth'),
            os.path.join(core_dir, 'config.v3_multi_exchange_rate.yaml'),
        ),
        # Add more configurations here...
    ]

    CONTEXT_LEN = 512
    DEBUG_MODE = True
    DATASETS_TO_RUN = None  # None executes all 17 datasets. Or provide list: ['exchange_rate', 'traffic']

    # ── Output directory ────────────────────────────────────────────────────
    out_dir = os.path.join(_PROJECT_ROOT, 'results')
    os.makedirs(out_dir, exist_ok=True)

    # ── Logger ──────────────────────────────────────────────────────────────
    logger = setup_logger(DEBUG_MODE, out_dir)

    logger.info(f"\n{'═'*80}")
    logger.info(f"  SC-Mamba Unified Batch Benchmark")
    logger.info(f"  {'Time':.<20s} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  {'Debug':.<20s} {DEBUG_MODE}")
    logger.info(f"  {'Context len':.<20s} {CONTEXT_LEN}")
    logger.info(f"{'═'*80}")

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"  Device: {device}")
    if device.type == 'cuda':
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ── Dataset filter ──────────────────────────────────────────────────────
    if DATASETS_TO_RUN:
        datasets = {k: v for k, v in REAL_DATASETS.items() if k in DATASETS_TO_RUN}
        if not datasets:
            logger.error(f"  ❌ None of {DATASETS_TO_RUN} found in REAL_DATASETS.")
            sys.exit(1)
        logger.info(f"  Evaluating subset: {list(datasets.keys())}")
    else:
        datasets = REAL_DATASETS
        logger.info(f"  Evaluating all {len(datasets)} datasets")

    # ── Batch Evaluation Loop ───────────────────────────────────────────────
    for idx, (label, ckpt, cfg) in enumerate(MODELS_TO_TEST, 1):
        logger.info(f"\n{'─'*80}")
        logger.info(f"▶️  EVALUATING MODEL [{idx}/{len(MODELS_TO_TEST)}]")
        logger.info(f"   Label : {label}")
        logger.info(f"   Ckpt  : {ckpt}")
        logger.info(f"   Config: {cfg}")
        logger.info(f"{'─'*80}\n")

        try:
            # Load model
            model, scaler, sub_day, n_assets = load_checkpoint(ckpt, cfg, device, logger)

            # Evaluate
            logger.info(f"\n  🚀 Starting evaluation: {label}\n")
            t_start = time.time()
            
            results_df = evaluate_all_datasets(
                model=model,
                device=device,
                scaler=scaler,
                sub_day=sub_day,
                logger=logger,
                context_len=CONTEXT_LEN,
                datasets=datasets,
            )
            
            total_time = time.time() - t_start

            # Display Results
            print_results_table(results_df, label, logger)
            logger.info(f"  ⏱️  Total evaluation time: {total_time:.1f}s")

            # Save Output
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            csv_path = os.path.join(out_dir, f'eval_unified_{label}_{ts}.csv')
            results_df.to_csv(csv_path, index=False, float_format='%.6f')
            logger.info(f"  📄 CSV  → {csv_path}")

            latex_str = generate_latex_table(results_df, label)
            tex_path = os.path.join(out_dir, f'eval_unified_{label}_{ts}.tex')
            with open(tex_path, 'w') as f:
                f.write(latex_str)
            logger.info(f"  📄 LaTeX → {tex_path}")

            # Safe cleanup to avoid cross-evaluation OOM
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            logger.info(f"\n  ✅ Finished: {label}\n")

        except Exception as e:
            logger.error(f"\n  ❌ Evaluation failed for: {label}")
            logger.error(f"  {traceback.format_exc()}\n")
            continue

    logger.info(f"{'='*80}")
    logger.info(f"🎉 All {len(MODELS_TO_TEST)} models evaluated.")
    logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    main()
