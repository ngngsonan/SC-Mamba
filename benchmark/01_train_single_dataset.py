# @title training
# ============================================================
# 🚀 SC-Mamba — Cross-Asset Causal Graph Training
# Trigger: num_assets > 1 + real_train_datasets → auto-activate
#          MultivariateRealDataset (train from scratch, no finetune)
# ============================================================
import yaml, os, subprocess
from pathlib import Path

PROJECT_ROOT   = '/content/SC-Mamba'
CHECKPOINT_DIR = '/content/drive/MyDrive/Colab Notebooks/SCMamba/sc_mamba_checkpoints'

# ── Danh sách các thí nghiệm Multivariate ───────────────────
EXPERIMENTS = {
    'exchange_rate': {
        'num_assets'        : 8,     # Exchange Rate: đúng 8 đồng tiền
        'pred_len'          : 20,
        'prior_mix_frac'    : 0.5,   # 50% synthetic GP + 50% real aligned
        'batch_size'        : 8,     # B×N = 8×8 = 64 series/step (T4-safe)
        'beta_kl'           : 0.2,   # (Giảm từ 0.5 xuống 0.2 để tránh Gradient Vanishing cho tau)
        'beta_anneal_epochs': 10,    # (Kéo dài warmup)
        'gamma_sparsity'    : 0.1,   # L1 penalty on mask density → sparse causal graph
        'training_rounds'   : 200,
        'sub_day'           : False, # Daily freq → no hour/minute features
    },
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
        'chunk_size'            : 256,
        'headdim'               : 128,
    },

    # Training schedule
    'num_epochs'        : 30, #30,   # Train from scratch — epoch 0
    'validation_rounds' : 50,
    'real_test_interval': 5,

    # Learning rate
    'lr_scheduler' : 'cosine',
    'initial_lr'   : 5e-5,
    'learning_rate': 1e-5,      # (Tăng từ 1e-7 lên 1e-5 để kích thích tau học lại)
    't_max'        : -1,        # Auto-set to num_epochs

    # Context/prediction length
    'context_len'      : 256,
    'min_seq_len'      : 60,
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
    'continue_training': True, # ← Train từ epoch 0 (không load checkpoint cũ)

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



config = {**BASE_CONFIG}
# =====================================================================
# 1. Verify real validation datasets exist (pkl files)
# =====================================================================
REAL_VAL_DIR = os.path.join(PROJECT_ROOT, 'data', 'real_val_datasets')
os.makedirs(REAL_VAL_DIR, exist_ok=True)

# Check which datasets are needed for real_test_datasets
real_test_ds = config.get('real_test_datasets', [])
padded = 'pad' if False else 'nopad'  # matches real_data_args.yaml pad=false
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
    !python data/scripts/store_real_datasets.py
    # Verify again
    for ds in missing:
        pkl_path = os.path.join(REAL_VAL_DIR, f'{ds}_{padded}_{MAX_LEN}.pkl')
        if os.path.exists(pkl_path):
            print(f'  ✅ {ds} → OK')
        else:
            print(f'  ⚠️  {ds} → STILL MISSING. Check store_real_datasets.py output above.')
else:
    print(f'\n✅ All {len(real_test_ds)} real validation datasets found.')


# =====================================================================
# 2. RUN TRAINING in EXPERIMENTS setups
# =====================================================================
for dataset_name, exp_cfg in EXPERIMENTS.items():
    print(f"\n{'='*65}")
    print(f"🚀 CROSS-ASSET TRAINING: {dataset_name.upper()}")
    print(f"   num_assets    : {exp_cfg['num_assets']}")
    print(f"   prior_mix_frac: {exp_cfg['prior_mix_frac']}  "
          f"(synthetic {exp_cfg['prior_mix_frac']*100:.0f}% + "
          f"real {(1-exp_cfg['prior_mix_frac'])*100:.0f}%)")
    print(f"   num_epochs    : {BASE_CONFIG['num_epochs']}  (train from scratch)")
    print(f"{'='*65}")

    # Build final config by merging BASE + experiment-specific keys
    config['num_assets']          = exp_cfg['num_assets']
    config['sub_day']             = exp_cfg['sub_day']
    config['pred_len']            = exp_cfg['pred_len']
    config['batch_size']          = exp_cfg['batch_size']
    config['beta_kl']             = exp_cfg['beta_kl']
    config['beta_anneal_epochs']  = exp_cfg['beta_anneal_epochs']
    config['gamma_sparsity']      = exp_cfg.get('gamma_sparsity', 0.0)
    config['nll_detach']          = exp_cfg.get('nll_detach', BASE_CONFIG['nll_detach'])
    config['spectral_config']      = {
        **BASE_CONFIG['spectral_config'],
        **exp_cfg.get('spectral_config', {})
    }
    config['training_rounds']     = exp_cfg['training_rounds']
    config['version']             = f'v_multi_{dataset_name}'

    # real_train_datasets + real_test_datasets — both point to same dataset
    config['real_train_datasets'] = [dataset_name]  # ← activates MultivariateRealDataset
    config['real_test_datasets']  = [dataset_name]

    # Inject prior_mix_frac into prior_config
    config['prior_config'] = {
        **BASE_CONFIG['prior_config'],
        'prior_mix_frac': exp_cfg['prior_mix_frac'],
    }

    # Write config to file
    config_path = f'{PROJECT_ROOT}/core/config.{config["version"]}.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"📝 Config saved: {config_path}")

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
        print(f"\n❌ {dataset_name} training failed (exit code {rc}). Stopping.")
        break


