# @title 02_train_17data_N_uni.py
# =====================================================================
# 🚀 SC-Mamba — Universal Univariate Training (N=1) on 17 Datasets
# =====================================================================
import yaml, os, subprocess

PROJECT_ROOT   = '/content/SC-Mamba'
CHECKPOINT_DIR = '/content/drive/MyDrive/Colab Notebooks/SCMamba/sc_mamba_checkpoints'

# ── DATASET REGISTRY ────────────────────────────────────────────────
DATASETS = [
    'car_parts_without_missing', 'cif_2016', 'covid_deaths', 'ercot', 
    'exchange_rate', 'fred_md', 'hospital', 'm1_monthly', 'm1_quarterly', 
    'm3_monthly', 'm3_quarterly', 'nn5_daily_without_missing', 'nn5_weekly', 
    'tourism_monthly', 'tourism_quarterly', 'traffic', 'weather'
]

print(f"\n{'='*65}")
print(f"🚀 UNIVERSAL UNIVARIATE TRAINING (N=1) ON {len(DATASETS)} DATASETS")
print(f"{'='*65}")

config = {
    # Core settings
    'seed'                 : 42,
    'debugging'            : False,
    'scaler'               : 'min_max',
    'sin_pos_enc'          : False,
    'sin_pos_const'        : False,
    'encoding_dropout'     : 0.1,
    'handle_constants_model': True,
    
    'num_assets'           : 1,      # Channel-Independent (N=1)
    'sub_day'              : True,   # Required for sub-daily features (e.g., Traffic/ERCOT)
    
    # Backbone architecture
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
    'num_epochs'           : 50,
    'training_rounds'      : 200,
    'validation_rounds'    : 50,
    'real_test_interval'   : 5,
    
    # Learning rate
    'lr_scheduler'         : 'cosine',
    'initial_lr'           : 5e-5,
    'learning_rate'        : 1e-7,
    't_max'                : -1,

    # Sequence & Prediction
    'context_len'          : 256,
    'min_seq_len'          : 30,
    'max_seq_len'          : 256,
    'pred_len'             : 60,
    'pred_len_min'         : 10,
    'pred_len_sample'      : True,
    'batch_size'           : 64,
    'no_pos_enc'           : False,

    # Loss
    'loss'                 : 'nll',
    'beta_kl'              : 0.1,
    'multipoint'           : True,
    'sample_multi_pred'    : 0.5,
    'diag_prints'          : True,
    'nll_detach'           : True,

    # Checkpoint
    'model_prefix'         : CHECKPOINT_DIR,
    'wandb'                : False,
    'continue_training'    : False,
    'version'              : 'v_config06_uni_17data',

    # Dataset targets
    'real_train_datasets'  : DATASETS,
    'real_test_datasets'   : ['exchange_rate', 'fred_md', 'm1_monthly'],

    # Synthetic prior settings
    'prior_config': {
        'prior_mix_frac'       : 0.7,   # 70% Synthetic GP + 30% Real
        'curriculum_learning'  : False,
        'mixup_prob'           : 0.0,
        'mixup_series'         : 4,
        'damp_and_spike'       : True,
        'damping_noise_ratio'  : 0.05,
        'spike_noise_ratio'    : 0.05,
        'spike_signal_ratio'   : 0.05,
        'spike_batch_ratio'    : 0.05,
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

# =====================================================================
# 1. Save YAML Configuration
# =====================================================================
os.makedirs(f'{PROJECT_ROOT}/core', exist_ok=True)
config_path = f'{PROJECT_ROOT}/core/config.{config["version"]}.yaml'
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
print(f"📝 Config saved: {config_path}")

# =====================================================================
# 2. Verify Validation Datasets
# =====================================================================
REAL_VAL_DIR = os.path.join(PROJECT_ROOT, 'data', 'real_val_datasets')
os.makedirs(REAL_VAL_DIR, exist_ok=True)
MAX_LEN = 512
padded = 'nopad' 

missing = []
for ds in DATASETS + config['real_test_datasets']:
    pkl_name = f'{ds}_{padded}_{MAX_LEN}.pkl'
    if not os.path.exists(os.path.join(REAL_VAL_DIR, pkl_name)):
        missing.append(ds)

missing = list(set(missing)) 

if missing:
    print(f'\n🔄 Generating {len(missing)} missing dataset(s) from GluonTS...')
    subprocess.run(['python', f'{PROJECT_ROOT}/data/scripts/store_real_datasets.py'])
    
for ds in DATASETS:
    pkl_name = f'{ds}_{padded}_{MAX_LEN}.pkl'
    pkl_path = os.path.join(REAL_VAL_DIR, pkl_name)
    if os.path.exists(pkl_path):
        size_mb = os.path.getsize(pkl_path) / (1024*1024)
        print(f'  ✅ {ds:30s} → {pkl_name} ({size_mb:.1f} MB)')
    else:
        print(f'  ❌ {ds:30s} → MISSING')

# =====================================================================
# 3. RUN TRAINING
# =====================================================================
print('\n🚀 Starting Univariate Model Training...')
os.environ['SC_MAMBA_DIAG'] = '1'

cmd = ['python', f'{PROJECT_ROOT}/core/train.py', '-c', config_path]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in iter(proc.stdout.readline, ''):
    print(line, end='', flush=True)
proc.stdout.close()
rc = proc.wait()

if rc == 0:
    print("\n✅ Universal Univariate Training Complete.")
else:
    print(f"\n❌ Training failed (exit code {rc}). Stopping.")