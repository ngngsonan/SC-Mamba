"""
14_eval_all.py
==============
Automated Script to run Zero-Shot Benchmark on ALL 17 datasets
for multiple Univariate (N=1) and Multivariate (N>1) checkpoints.
Handles chunked inference mathematically precisely.
"""

import os
import sys
import subprocess
import yaml
import pandas as pd
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
sys.path.insert(0, PROJECT_ROOT)

from core.eval_real_dataset import REAL_DATASETS

CKPT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')

MODEL_TO_TEST = [
    (
        'N=1 (Uni) v2',
        os.path.join(CKPT_DIR, 'SCMamba_v2_17data_N_uni_best_mase.pth'),
        os.path.join(PROJECT_ROOT, 'core', 'config.v_config06_uni_17data.yaml')
    ),
    (
        'N=1 (Uni) v2 best',
        os.path.join(CKPT_DIR, 'SCMamba_v2_17data_N_uni_best.pth'),
        os.path.join(PROJECT_ROOT, 'core', 'config.v_config06_uni_17data.yaml')
    ),
    (
        'N=8 (Multi) v2',
        os.path.join(CKPT_DIR, 'SCMamba_v2_multi_exchange_rate_best.pth'),
        os.path.join(PROJECT_ROOT, 'core', 'config.v_config06_multi8_exchange.yaml')
    )
]

def parse_metrics_from_yaml(yml_path):
    if not os.path.exists(yml_path):
        return {'mase': np.nan, 'mcrps': np.nan, 'mae': np.nan}
    try:
        with open(yml_path, 'r') as f:
            raw = yaml.safe_load(f)
            if raw:
                metrics = next(iter(raw.values()), {})
                mcrps = metrics.get('mcrps', metrics.get('crps_scaled', np.nan))
                return {
                    'mase': float(metrics.get('mase', np.nan)),
                    'mcrps': float(mcrps) if mcrps is not None else np.nan,
                    'mae': float(metrics.get('mae', np.nan))
                }
    except Exception as e:
        print(f"Error parsing YML {yml_path}: {e}")
    return {'mase': np.nan, 'mcrps': np.nan, 'mae': np.nan}

def main():
    print(f"\n{'='*95}")
    print(f"🚀 SC-Mamba Universal Benchmark Evaluation (All 17 Datasets)")
    print(f"   Includes Universal N=1 and Chunked Multivariate N=8")
    print(f"{'='*95}\n")

    all_results = []

    for label, ckpt_path, config_path in MODEL_TO_TEST:
        if not os.path.exists(ckpt_path):
            print(f"❌ [Skip] Checkpoint not found: {ckpt_path}")
            continue
        
        model_name = os.path.basename(ckpt_path).replace('.pth', '')
        for suffix in ('_best', '_Final'):
            if model_name.endswith(suffix):
                model_name = model_name[:-len(suffix)]
                break
                
        eval_base_dir = os.path.join(PROJECT_ROOT, 'data', 'real_data_evals', model_name, 'multipoint')
        
        missing = []
        for ds in REAL_DATASETS.keys():
            if not os.path.exists(os.path.join(eval_base_dir, f"{ds}_512.yml")):
                missing.append(ds)
                
        if missing:
            print(f"\n▶️ Dò thấy {len(missing)} datasets chưa có kết quả cho mô hình {label}.")
            print(f"   Khởi động eval_real_dataset.py (quét tất cả 17 datasets)...")
            cmd = [
                'python', os.path.join(PROJECT_ROOT, 'core', 'eval_real_dataset.py'),
                '-c', ckpt_path,
                '-o', model_name
            ]
            if os.path.exists(config_path):
                cmd.extend(['-cfg', config_path])
                
            res = subprocess.run(cmd, env=dict(os.environ, MKL_NUM_THREADS="1"), capture_output=False)
            if res.returncode != 0:
                print(f"⚠️ Cảnh báo: Lỗi khi chạy eval cho {label} chặn tiến độ. Có thể vài dataset bị thiếu cache.\n")

        print(f"\n✅ Đang tổng hợp số liệu YML cho {label}...")
        for ds in REAL_DATASETS.keys():
            yml_path = os.path.join(eval_base_dir, f"{ds}_512.yml")
            metrics = parse_metrics_from_yaml(yml_path)
            
            all_results.append({
                'Model': label,
                'Dataset': ds,
                'MASE': metrics['mase'],
                'mCRPS': metrics['mcrps']
            })

    if not all_results:
        print("Trống dữ liệu. Hãy tạo checkpoint trước.")
        return
        
    df = pd.DataFrame(all_results)
    pd.options.display.float_format = '{:,.4f}'.format
    
    pivot_mase = df.pivot(index='Dataset', columns='Model', values='MASE')
    pivot_mcrps = df.pivot(index='Dataset', columns='Model', values='mCRPS')
    
    print(f"\n{'='*95}")
    print(f"{'Dataset':<26} | {'MASE':^30} | {'mCRPS':^30}")
    
    headers_mase = " | ".join([f"{str(m):<13}" for m in pivot_mase.columns])
    headers_mcrps = " | ".join([f"{str(m):<13}" for m in pivot_mcrps.columns])
    print(f"{' '*26} | {headers_mase} | {headers_mcrps}")
    print(f"{'─'*95}")
    
    for ds in REAL_DATASETS.keys():
        if ds not in pivot_mase.index: continue
        row_str = f"{ds:<26} | "
        
        for m in pivot_mase.columns:
            val = pivot_mase.at[ds, m]
            s = f"{val:>13.4f}" if pd.notna(val) else f"{'-':>13}"
            row_str += f"{s} | "
            
        for m in pivot_mcrps.columns:
            val = pivot_mcrps.at[ds, m]
            s = f"{val:>13.4f}" if pd.notna(val) else f"{'-':>13}"
            row_str += f"{s} | "
            
        print(row_str)
        
    print(f"{'─'*95}")
    
    print("\nGLOBAL SUMMARY (MEAN ACROSS DATASETS):")
    summary = df.groupby('Model')[['MASE', 'mCRPS']].mean().rename(columns={'MASE': 'Avg MASE', 'mCRPS': 'Avg mCRPS'})
    print(summary.to_string(float_format='%.4f'))
    print(f"{'='*95}\n🚀 Benchmark 14 script hoàn tất.")
    
if __name__ == '__main__':
    main()
