"""
eval_unified_runner.py
======================
Interface for batch-running SC-Mamba evaluations using the unified pipeline.
Mimics the `MODEL_TO_TEST` configuration interface from previous benchmark scripts.
Ensures memory isolation between evaluations by using subprocesses.

Usage:
  python benchmark/eval_unified_runner.py
"""

import os
import sys
import subprocess

# ── Path Setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
COLAB_CKPT = '/content/drive/MyDrive/Colab Notebooks/SCMamba/sc_mamba_checkpoints'

# Auto-detect checkpoint directory (Colab vs Local)
CKPT_DIR = COLAB_CKPT if os.path.exists(COLAB_CKPT) else os.path.join(PROJECT_ROOT, 'checkpoints')
CORE_DIR = os.path.join(PROJECT_ROOT, 'core')
EVAL_SCRIPT = os.path.join(SCRIPT_DIR, 'eval_unified.py')


# ═════════════════════════════════════════════════════════════════════════════
# Configuration: Models to Evaluate
# ═════════════════════════════════════════════════════════════════════════════
# Format: (output_label, checkpoint_path, config_yaml_path)
# You can add as many models (N=1 or N>1) to this list as needed.

MODELS_TO_TEST = [
    (
        'SC-Mamba (N=1)',
        os.path.join(CKPT_DIR, 'SCMamba_v2_17data_N_uni_best_mase.pth'),
        os.path.join(CORE_DIR, 'config.based_setup.yaml'),
    ),
    (
        'SC-Mamba (N=8, Chunked)',
        os.path.join(CKPT_DIR, 'SCMamba_v3_multi_exchange_rate_best.pth'),
        os.path.join(CORE_DIR, 'config.v3_multi_exchange_rate.yaml'),
    ),
    # Add more models here...
]

# Common Arguments
CONTEXT_LEN = 512
DEBUG_MODE = True      # Set to True to generate detailed debug logs
DATASETS = None        # Set to list of datasets e.g., ['exchange_rate', 'traffic'] or None for all 17


# ═════════════════════════════════════════════════════════════════════════════
# Runner Logic
# ═════════════════════════════════════════════════════════════════════════════

def run_evaluation():
    print(f"\n{'='*80}")
    print(f"🚀 SC-Mamba Unified Evaluation Runner")
    print(f"   Engine: {EVAL_SCRIPT}")
    print(f"   Models to test: {len(MODELS_TO_TEST)}")
    print(f"{'='*80}\n")

    if not os.path.exists(EVAL_SCRIPT):
        print(f"❌ Error: Unified evaluation script not found at {EVAL_SCRIPT}")
        sys.exit(1)

    for i, (label, ckpt, cfg) in enumerate(MODELS_TO_TEST, 1):
        print(f"\n{'─'*80}")
        print(f"▶️  EVALUATING MODEL [{i}/{len(MODELS_TO_TEST)}]")
        print(f"   Label : {label}")
        print(f"   Ckpt  : {ckpt}")
        print(f"   Config: {cfg}")
        print(f"{'─'*80}\n")

        # Build command
        cmd = [
            sys.executable, EVAL_SCRIPT,
            '-c', ckpt,
            '-cfg', cfg,
            '-o', label,
            '--ctx_len', str(CONTEXT_LEN),
        ]

        if DEBUG_MODE:
            cmd.append('--debug')

        if DATASETS:
            cmd.extend(['--datasets'] + DATASETS)

        # Execute subprocess
        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=None,  # Inherit stdout to stream the output live to the console
                stderr=None,
            )
            print(f"\n✅ Finished evaluating: {label}\n")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Evaluation failed for: {label}")
            print(f"   Command returned non-zero exit status {e.returncode}\n")
        except KeyboardInterrupt:
            print(f"\n🛑 Evaluation interrupted by user for: {label}")
            break

    print(f"{'='*80}")
    print(f"🎉 All evaluations completed.")
    print(f"{'='*80}")


if __name__ == '__main__':
    run_evaluation()
