# @title 03_train_17data_N_multi.py
# ============================================================
# 🚀 SC-Mamba — Cross-Asset Causal Graph Training
# Train script for 14 remaining datasets with dynamic OOM control.
# Note: exchange_rate trained in 01, and m3/ercot missing dependencies.
# weather is heavily corrupted.
# ============================================================
import yaml, os, subprocess, time
from pathlib import Path

PROJECT_ROOT   = '.'
CHECKPOINT_DIR = '/content/drive/MyDrive/Colab Notebooks/SCMamba/sc_mamba_checkpoints'

# ── Danh sách 14 datasets Multivariate ───────────────────
# Added seq_len from GluonTS metadata to calculate max allowed context_len cleanly.
EXPERIMENTS = {
    'nn5_daily_without_missing': {'num_assets': 111, 'pred_len': 56, 'seq_len': 735},
    'nn5_weekly':                {'num_assets': 111, 'pred_len': 8,  'seq_len': 105},
    'covid_deaths':              {'num_assets': 266, 'pred_len': 30, 'seq_len': 182},
    'hospital':                  {'num_assets': 767, 'pred_len': 12, 'seq_len': 72},
    'fred_md':                   {'num_assets': 107, 'pred_len': 12, 'seq_len': 716},
    'car_parts_without_missing': {'num_assets': 2674, 'pred_len': 12, 'seq_len': 39},
    'traffic':                   {'num_assets': 862, 'pred_len': 24, 'seq_len': 14036},
    'm1_monthly':                {'num_assets': 617, 'pred_len': 18, 'seq_len': 42},
    'm1_quarterly':              {'num_assets': 203, 'pred_len': 8,  'seq_len': 40},
    'm3_monthly':                {'num_assets': 1428, 'pred_len': 18, 'seq_len': 144},
    'm3_quarterly':              {'num_assets': 756,  'pred_len': 8,  'seq_len': 72},
    'cif_2016':                  {'num_assets': 72,  'pred_len': 12, 'seq_len': 108},
    'tourism_monthly':           {'num_assets': 366, 'pred_len': 24, 'seq_len': 163},
    'tourism_quarterly':         {'num_assets': 427, 'pred_len': 8,  'seq_len': 55},
}

# ── Template cấu hình (shared across experiments) ───────────
BASE_CONFIG = {
    # Core settings
    'seed'                 : 42,
    'debugging'            : False,
    'scaler'               : 'min_max',
    'sin_pos_enc'          : False,
    'sin_pos_const'        : False,
    'encoding_dropout'     : 0.1,
    'handle_constants_model': True,

    # Backbone architecture (Mamba2 — matches existing checkpoints)
    'ssm_config': {
        'bidirectional'         : False,
        'enc_conv'              : True,
        'init_dil_conv'         : True,
        'enc_conv_kernel'       : 5,
        'init_conv_kernel'      : 5,
        'init_conv_max_dilation': 3,
        'global_residual'       : False,
        'in_proj_norm'          : False,
        'initial_gelu_flag'     : True,
        'linear_seq'            : 15,
        'mamba2'                : True,
        'norm'                  : True,
        'norm_type'             : 'layernorm',
        'num_encoder_layers'    : 2,
        'd_state'               : 128,
        'residual'              : False,
        'token_embed_len'       : 1024,
        'block_expansion'       : 2,
        'chunk_size'            : 256,  # dynamically reduced for large data
        'headdim'               : 128,
    },

    # Training schedule
    'num_epochs'        : 30,   # Train from scratch — epoch 0
    'validation_rounds' : 50,
    'real_test_interval': 1,

    # Learning rate
    'lr_scheduler' : 'cosine',
    'initial_lr'   : 5e-5,
    'learning_rate': 1e-5,      # (Tăng từ 1e-7 lên 1e-5 để kích thích tau học lại)
    't_max'        : -1,        # Auto-set to num_epochs

    # Context/prediction length
    'context_len'      : 256,
    'min_seq_len'      : 20,    # Dropped to allow extremely short sequences
    'max_seq_len'      : 256,
    'pred_len_sample'  : False, # Fixed pred_len (stable Cross-Asset windows)
    'pred_len_min'     : 10,

    # Loss
    'loss'             : 'nll',
    'multipoint'       : True,
    'sample_multi_pred': 0.3,
    'diag_prints'      : True,
    'nll_detach'       : True,      # Prevent Variance Starvation (detached sigma2/mu grads)
    'spectral_config'  : {
        'tau_init'     : 1.0,       # Initial Gumbel-Softmax temperature
        'alpha_init'   : 10.0,      # Initial mask density (log_alpha ≈ 2.3)
    },

    # Checkpoint
    'model_prefix'    : CHECKPOINT_DIR,
    'wandb'           : False,
    'continue_training': False, # ← Train từ epoch 0 (không load checkpoint cũ)

    # Synthetic prior settings (shared structure, mix_frac set per experiment)
    'prior_config': {
        'curriculum_learning'  : False,
        'mixup_prob'           : 0.0,
        'mixup_series'         : 4,
        'damp_and_spike'       : False,
        'damping_noise_ratio'  : 0.0,
        'spike_noise_ratio'    : 0.0,
        'spike_signal_ratio'   : 0.0,
        'spike_batch_ratio'    : 0.0,
        'fp_options': {
            'linear_random_walk_frac': 0,
            'seasonal_only'  : False,
            'scale_noise'    : [0.6, 0.3],
            'trend_exp'      : False,
            'harmonic_scale_ratio': 0.4,
            'harmonic_rate'  : 0.75,
            'trend_additional': True,
            'transition_ratio': 0.0,
        },
        'gp_prior_config': {
            'max_kernels'            : 6,
            'likelihood_noise_level' : 0.4,
            'noise_level'            : 'random',
            'use_original_gp'        : False,
            'gaussians_periodic'     : True,
            'peak_spike_ratio'       : 0.1,
            'subfreq_ratio'          : 0.2,
            'periods_per_freq'       : 0.5,
            'gaussian_sampling_ratio': 0.2,
            'kernel_periods'         : [4, 5, 7, 21, 24, 30, 60, 120],
            'max_period_ratio'       : 1.0,
            'kernel_bank': {
                'matern_kernel'         : 1.5,
                'linear_kernel'         : 1,
                'periodic_kernel'       : 5,
                'polynomial_kernel'     : 0,
                'spectral_mixture_kernel': 0,
            },
        },
    },
}

for dataset_name, exp_cfg in EXPERIMENTS.items():
    num_assets = exp_cfg['num_assets']
    seq_len = exp_cfg['seq_len']
    pred_len = exp_cfg['pred_len']
    
    # --- DYNAMIC DATASET HEURISTICS ---
    # Safe batch allocation strategy: Total tokens roughly = B * N * Context
    safe_target = 64  # Equivalent base: 8 batch * 8 assets (from exchange_rate)
    dynamic_batch_size = max(1, safe_target // num_assets)
    
    # Fix 1: Dynamically calculate context_len so Multivariate Loader doesn't crash on num_samples=0
    # Minimum train seq required = context_len + pred_len >= seq_len. 
    max_allowed_context = max(24, seq_len - pred_len - 1)
    context_len = min(256, max_allowed_context)

    # Note: if seq_len is extremely small (e.g. car_parts with seq=39, pred=12),
    # context will shrink to 26, enabling at least 1 training window.

    # Fix 2: Severe OOM protection for dense networks using granular chunk_size scaling
    if num_assets >= 2000:
        chunk_size = 16   # car_parts (2.6k assets)
    elif num_assets >= 1000:
        chunk_size = 32   # m3_monthly (1.4k assets)
    elif num_assets >= 700:
        chunk_size = 48   # hospital, traffic, m3_quarterly
    elif num_assets >= 500:
        chunk_size = 64   # m1_monthly
    else:
        chunk_size = BASE_CONFIG['ssm_config']['chunk_size']

    print(f"\n{'='*65}")
    print(f"🚀 CROSS-ASSET TRAINING: {dataset_name.upper()}")
    print(f"   num_assets    : {num_assets}")
    print(f"   dyn_batch_size: {dynamic_batch_size}")
    print(f"   chunk_size    : {chunk_size}")
    print(f"   context_len   : {context_len} (seq_len={seq_len})")
    print(f"   prior_mix_frac: 0.5  (synthetic 50% + real 50%)")
    print(f"   num_epochs    : {BASE_CONFIG['num_epochs']}")
    print(f"{'='*65}")

    config = {**BASE_CONFIG}
    # Create an independent copy for ssm_config
    config['ssm_config'] = {**BASE_CONFIG['ssm_config']}
    
    # Build final config by merging BASE + experiment-specific keys
    config['num_assets']          = num_assets
    config['sub_day']             = False
    config['pred_len']            = pred_len
    config['context_len']         = context_len
    config['max_seq_len']         = context_len
    config['batch_size']          = dynamic_batch_size
    config['ssm_config']['chunk_size'] = chunk_size
    
    # Generic multivariate settings from original exchange_rate config
    config['beta_kl']             = 0.2
    config['beta_anneal_epochs']  = 10
    config['gamma_sparsity']      = 0.1
    config['training_rounds']     = 10 #200
    config['version']             = f'v2_multi_{dataset_name}'

    # real_train_datasets + real_test_datasets — both point to same dataset
    config['real_train_datasets'] = [dataset_name]  # ← activates MultivariateRealDataset
    config['real_test_datasets']  = [dataset_name]

    # Inject prior_mix_frac into prior_config
    config['prior_config'] = {
        **BASE_CONFIG['prior_config'],
        'prior_mix_frac': 0.5,
    }
    
    # Write config to file
    config_path = f'{PROJECT_ROOT}/core/config.{config["version"]}.yaml'
    # Fallback if run locally without colab hierarchy
    if not os.path.exists(f'{PROJECT_ROOT}/core'):
        os.makedirs(f'{PROJECT_ROOT}/core', exist_ok=True)
        
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"📝 Config saved: {config_path}")

    # =====================================================================
    # 1. Verify real validation datasets exist (pkl files)
    # =====================================================================
    REAL_VAL_DIR = os.path.join(PROJECT_ROOT, 'data', 'real_val_datasets')
    os.makedirs(REAL_VAL_DIR, exist_ok=True)

    # Check which datasets are needed for real_test_datasets
    real_test_ds = config.get('real_test_datasets', [])
    padded = 'nopad'
    MAX_LEN = 512

    missing = []
    for ds in real_test_ds:
        pkl_name = f'{ds}_{padded}_{MAX_LEN}.pkl'
        pkl_path = os.path.join(REAL_VAL_DIR, pkl_name)
        if os.path.exists(pkl_path):
            size_mb = os.path.getsize(pkl_path) / (1024*1024)
            print(f'  ✅ {ds:30s} → {pkl_name} ({size_mb:.1f} MB)')
        else:
            missing.append(ds)
            print(f'  ❌ {ds:30s} → MISSING')

    if missing:
        print(f'\n🔄 Generating {len(missing)} missing dataset(s) from GluonTS...')
        print('   This downloads from AWS and converts to pkl. May take 1-5 min per dataset.')
        
        # Wait using subprocess run for proper error handling rather than !python
        err = os.system("python data/scripts/store_real_datasets.py")
        if err != 0:
             print("Data generation script encountered an error. Proceeding with caution.")
             
        # Verify again
        for ds in missing:
            pkl_path = os.path.join(REAL_VAL_DIR, f'{ds}_{padded}_{MAX_LEN}.pkl')
            if os.path.exists(pkl_path):
                print(f'  ✅ {ds} → OK')
            else:
                print(f'  ⚠️  {ds} → STILL MISSING. Models might crash.')
    else:
        print(f'\n✅ Real validation dataset found.')

    # =====================================================================
    # 2. RUN TRAINING
    # =====================================================================
    # Launch training — stream logs live
    cmd = ['python', f'{PROJECT_ROOT}/core/train.py', '-c', config_path]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in iter(proc.stdout.readline, ''):
        print(line, end='', flush=True)
    proc.stdout.close()
    rc = proc.wait()

    if rc == 0:
        print(f"\n✅ {dataset_name} training complete.")
    else:
        print(f"\n❌ {dataset_name} training failed (exit code {rc}). Continuing to next...")
        
    print("\n[OOM Prevention] Cool down and release memory...")
    time.sleep(3)
