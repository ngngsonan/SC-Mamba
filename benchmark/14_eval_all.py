"""
14_eval_all.py
==============
Universal Zero-Shot Benchmark: evaluates ALL checkpoints (Uni N=1, Multi N=8)
across all 17 GluonTS datasets using eval_real_dataset.py.

For N=8 models, uses Asset-dimension Chunking to cover the full dataset.
Results are cached as .yml files; re-runs only compute missing datasets.

Usage:
    python benchmark/14_eval_all.py
"""

import os
import sys
import subprocess
import yaml
import pandas as pd
import numpy as np

# ── Path Setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
sys.path.insert(0, PROJECT_ROOT)

from core.eval_real_dataset import REAL_DATASETS

CKPT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')

# ── Checkpoint Registry ────────────────────────────────────────────────────────
# Format: (display_label, checkpoint_path, config_yaml_path)
# Config YAML is REQUIRED for N>1 models (must contain num_assets).
MODEL_TO_TEST = [
    (
        'SC-Mamba N=1 (best_mase)',
        os.path.join(CKPT_DIR, 'SCMamba_v2_17data_N_uni_best_mase.pth'),
        os.path.join(PROJECT_ROOT, 'core', 'config.based_setup.yaml'),
    ),
    (
        'SC-Mamba N=1 (best)',
        os.path.join(CKPT_DIR, 'SCMamba_v2_17data_N_uni_best.pth'),
        os.path.join(PROJECT_ROOT, 'core', 'config.based_setup.yaml'),
    ),
    (
        'SC-Mamba N=8 (Chunked)',
        os.path.join(CKPT_DIR, 'SCMamba_v2_multi_exchange_rate_best.pth'),
        os.path.join(PROJECT_ROOT, 'core', 'config.v_config06_multi8_exchange.yaml'),
    ),
]


# ── Utility: derive model_name from checkpoint path ──────────────────────────
# Must match the EXACT same logic as eval_real_dataset.main_evaluator
# to ensure we read the correct cache directory.
_STRIP_SUFFIXES = ('_best_mase', '_best', '_Final')

def derive_model_name(ckpt_path: str) -> str:
    """Derive cache-directory model_name from checkpoint filename."""
    name = os.path.basename(ckpt_path).replace('.pth', '')
    for suffix in _STRIP_SUFFIXES:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def parse_metrics_from_yaml(yml_path: str) -> dict:
    """Read MASE/mCRPS from a cached .yml evaluation result."""
    if not os.path.exists(yml_path):
        return {'mase': np.nan, 'mcrps': np.nan}
    try:
        with open(yml_path, 'r') as f:
            raw = yaml.safe_load(f)
        if raw:
            metrics = next(iter(raw.values()), {})
            mcrps = metrics.get('mcrps')
            if mcrps is None:
                mcrps = metrics.get('crps_scaled', np.nan)
            return {
                'mase': float(metrics.get('mase', np.nan)),
                'mcrps': float(mcrps) if mcrps is not None else np.nan,
            }
    except Exception as e:
        print(f"  ⚠️ Error parsing {yml_path}: {e}")
    return {'mase': np.nan, 'mcrps': np.nan}


def run_evaluation(ckpt_path: str, config_path: str, model_name: str) -> bool:
    """Run eval_real_dataset.py via subprocess. Returns True on success."""
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, 'core', 'eval_real_dataset.py'),
        '-c', ckpt_path,
        '-o', model_name,
    ]
    if config_path and os.path.exists(config_path):
        cmd.extend(['-cfg', config_path])
    else:
        print(f"  ⚠️ Config not found: {config_path}")
        print(f"      Model will use checkpoint-embedded config or DEFAULT fallback.")

    print(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        env=dict(os.environ, MKL_NUM_THREADS="1"),
        capture_output=False,
    )
    return result.returncode == 0


def main():
    print(f"\n{'='*100}")
    print(f"🚀 SC-Mamba Universal Zero-Shot Benchmark (17 Datasets)")
    print(f"   Asset-dimension Chunking for N>1 models")
    print(f"{'='*100}\n")

    all_results = []
    model_labels = []

    for label, ckpt_path, config_path in MODEL_TO_TEST:
        if not os.path.exists(ckpt_path):
            print(f"❌ [Skip] {label}: checkpoint not found at {ckpt_path}")
            continue

        model_name = derive_model_name(ckpt_path)
        eval_dir = os.path.join(
            PROJECT_ROOT, 'data', 'real_data_evals', model_name, 'multipoint'
        )

        # Check which datasets are missing cache
        missing = [
            ds for ds in REAL_DATASETS
            if not os.path.exists(os.path.join(eval_dir, f"{ds}_512.yml"))
        ]

        if missing:
            print(f"\n▶️ {label}: {len(missing)}/{len(REAL_DATASETS)} datasets missing cache.")
            print(f"   Running eval_real_dataset.py for model_name='{model_name}'...")
            success = run_evaluation(ckpt_path, config_path, model_name)
            if not success:
                print(f"  ⚠️ eval subprocess failed. Partial results may exist.\n")
        else:
            print(f"\n✅ {label}: all 17 datasets cached. Reading results...")

        # Collect metrics 
        model_labels.append(label)
        for ds in REAL_DATASETS:
            yml_path = os.path.join(eval_dir, f"{ds}_512.yml")
            metrics = parse_metrics_from_yaml(yml_path)
            all_results.append({
                'Model': label,
                'Dataset': ds,
                'MASE': metrics['mase'],
                'mCRPS': metrics['mcrps'],
            })

    if not all_results:
        print("❌ No results collected. Ensure checkpoints exist.")
        return

    df = pd.DataFrame(all_results)

    # ── Build Comparison Table ────────────────────────────────────────────────
    n_models = len(model_labels)
    col_w = 18  # width per model column
    ds_w = 28   # width for dataset name column
    sep_w = ds_w + 3 + (col_w + 3) * n_models * 2
    
    print(f"\n{'='*sep_w}")
    
    # Header row 1: metric group names
    mase_header = "MASE".center(col_w * n_models + 3 * (n_models - 1))
    mcrps_header = "mCRPS".center(col_w * n_models + 3 * (n_models - 1))
    print(f"{'Dataset':<{ds_w}} | {mase_header} | {mcrps_header}")
    
    # Header row 2: model names
    model_cols = " | ".join(f"{m[:col_w]:<{col_w}}" for m in model_labels)
    print(f"{'':<{ds_w}} | {model_cols} | {model_cols}")
    print(f"{'─'*sep_w}")

    pivot_mase = df.pivot(index='Dataset', columns='Model', values='MASE')
    pivot_mcrps = df.pivot(index='Dataset', columns='Model', values='mCRPS')

    for ds in REAL_DATASETS:
        if ds not in pivot_mase.index:
            continue
        row = f"{ds:<{ds_w}} | "
        for m in model_labels:
            val = pivot_mase.at[ds, m] if m in pivot_mase.columns else np.nan
            row += f"{val:>{col_w}.4f} | " if pd.notna(val) else f"{'-':>{col_w}} | "
        for m in model_labels:
            val = pivot_mcrps.at[ds, m] if m in pivot_mcrps.columns else np.nan
            row += f"{val:>{col_w}.4f} | " if pd.notna(val) else f"{'-':>{col_w}} | "
        print(row)

    print(f"{'─'*sep_w}")

    # ── Global Summary ────────────────────────────────────────────────────────
    summary = df.groupby('Model')[['MASE', 'mCRPS']].agg(
        lambda x: np.nanmean(x) if x.notna().any() else np.nan
    ).rename(columns={'MASE': 'Avg MASE', 'mCRPS': 'Avg mCRPS'})
    # Reorder rows to match MODEL_TO_TEST order
    summary = summary.reindex([lbl for lbl in model_labels if lbl in summary.index])

    print(f"\n📊 GLOBAL SUMMARY (Mean across 17 datasets):\n")
    print(summary.to_string(float_format='%.4f'))
    print(f"\n{'='*sep_w}")
    print(f"✅ Benchmark complete.\n")


if __name__ == '__main__':
    main()
