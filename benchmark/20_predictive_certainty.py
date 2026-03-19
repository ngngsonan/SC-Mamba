# @title 20_predictive_certainty.py
"""
20_predictive_certainty.py — SC-Mamba Predictive Certainty Analysis
====================================================================
Compares N=1 (Univariate) vs N=8 (SC-Mamba Multivariate) on:
  PART 1: Per-asset line charts with original-scale ground truth + predictions + ±2σ bands
  PART 2: Variance reduction table (σ², MAE, 95% coverage)
  PART 3: Variance bar chart comparison
  PART 4: Quantitative ablation (MASE / MAE / RMSE / SMAPE / NLL / mCRPS)

All data is plotted in ORIGINAL SCALE (inverse-scaled via scale_data()).
Ground truth is shown with natural temporal fluctuations.

Usage: Add datasets to ANALYSIS_DATASETS list to extend analysis.
"""
import torch, os, sys, math, pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# ── Resolve project root ───────────────────────────────────────────────────
PROJECT_ROOT = '.'
if not os.path.exists(os.path.join(PROJECT_ROOT, 'core')):
    PROJECT_ROOT = '/content/SC-Mamba'
sys.path.insert(0, PROJECT_ROOT)
os.environ['TRITON_F32_DEFAULT'] = 'ieee'

from core.models import SCMamba_Forecaster
from core.eval_real_dataset import scale_data, nll_eval, crps_gaussian, REAL_DATASET_ASSETS
from utilsforecast.losses import mase, mae, smape, rmse
from data.data_provider.multivariate_loader import MultivariateRealDataset

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — Add datasets here to extend analysis
# ══════════════════════════════════════════════════════════════════════════════
CKPT_DIR = '/content/drive/MyDrive/Colab Notebooks/SCMamba/sc_mamba_checkpoints'
if not os.path.exists(CKPT_DIR):
    CKPT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')

CKPT_MULTI = os.path.join(CKPT_DIR, 'SCMamba_v3_multi_exchange_rate_best.pth')
CKPT_UNI   = os.path.join(CKPT_DIR, 'SCMamba_v_17data_N_uni_best_mase.pth')

# ┌────────────────────────────────────────────────────────────────────────────┐
# │  ADD / REMOVE datasets below to extend or narrow the analysis.            │
# │  Format: { 'name': pred_len }                                             │
# │  'labels' is optional — if omitted, uses generic "Asset_0", "Asset_1"...  │
# └────────────────────────────────────────────────────────────────────────────┘
ANALYSIS_DATASETS = [
    {
        'name': 'exchange_rate',
        'pred_len': 96,
        'labels': ['AUD', 'GBP', 'CAD', 'CHF', 'CNY', 'JPY', 'NZD', 'SGD'],
    },
    # ── Uncomment below to add more datasets ──────────────────────────────
    # {'name': 'weather',    'pred_len': 30},
    # {'name': 'traffic',    'pred_len': 24},
    # {'name': 'fred_md',    'pred_len': 12},
    # {'name': 'nn5_daily_without_missing', 'pred_len': 56},
]

CTX_LEN    = 256
SCALER     = 'min_max'
N_ASSETS   = 8

SSM_CONFIG = {
    'mamba2': True, 'num_encoder_layers': 2, 'd_state': 128,
    'headdim': 128, 'block_expansion': 2, 'token_embed_len': 1024,
    'chunk_size': 256, 'linear_seq': 15, 'norm': True,
    'norm_type': 'layernorm', 'residual': False, 'global_residual': False,
    'bidirectional': False, 'in_proj_norm': False, 'enc_conv': True,
    'enc_conv_kernel': 5, 'init_dil_conv': True, 'init_conv_kernel': 5,
    'init_conv_max_dilation': 3, 'initial_gelu_flag': True,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'benchmark', 'certainty_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
REAL_VAL_DIR = os.path.join(PROJECT_ROOT, 'data', 'real_val_datasets')
os.makedirs(REAL_VAL_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path, num_assets, device):
    """Loads a checkpoint into SCMamba_Forecaster."""
    model = SCMamba_Forecaster(N_assets=num_assets, ssm_config=SSM_CONFIG).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    return model.eval()


def get_total_assets(pkl_path):
    """Returns total asset count in a GluonTS PKL dataset."""
    with open(pkl_path, 'rb') as f:
        df = pickle.load(f)
    return len(df.index.get_level_values('Series').unique())


def build_test_sample(pkl_path, pred_len, n_assets):
    """
    Builds a test sample from the dataset PKL.
    For datasets with > n_assets series, uses the first n_assets columns
    (exchange_rate has exactly 8, so this is identity for that case).
    For datasets with < n_assets series, raises an error.
    """
    ds = MultivariateRealDataset(
        pkl_path=pkl_path, pred_len=pred_len, context_len=CTX_LEN,
        split='test', N_assets=n_assets,
    )
    sample = ds[len(ds) - 1]  # Last test window (most recent data)
    return sample


def prepare_multi_input(x, ts_x, ts_y, N, device):
    """Constructs model input for N-asset multivariate forward pass."""
    T_pred = ts_y.shape[0]
    return {
        'history':      x.permute(1, 0).to(device),
        'ts':           ts_x.unsqueeze(0).expand(N, -1, -1).to(device),
        'target_dates': ts_y.unsqueeze(0).expand(N, -1, -1).to(device),
        'task':         torch.zeros(N, T_pred, dtype=torch.int32, device=device),
    }


def prepare_uni_input(x_single, ts_x, ts_y, device):
    """Constructs model input for single-asset univariate forward pass."""
    T_pred = ts_y.shape[0]
    return {
        'history':      x_single.permute(1, 0).to(device),
        'ts':           ts_x.unsqueeze(0).to(device),
        'target_dates': ts_y.unsqueeze(0).to(device),
        'task':         torch.zeros(1, T_pred, dtype=torch.int32, device=device),
    }


def canonical_evaluate_ablation(model, pkl_path, pred_len, n_assets, device):
    """
    Full quantitative evaluation (MASE/MAE/RMSE/SMAPE/NLL/mCRPS)
    for a single dataset, matching the canonical pipeline.
    """
    from gluonts.dataset.repository.datasets import get_dataset
    from gluonts.time_feature.seasonality import get_seasonality

    ds_name = os.path.basename(pkl_path).replace('_nopad_512.pkl', '')

    ds = MultivariateRealDataset(
        pkl_path=pkl_path, pred_len=pred_len, context_len=CTX_LEN,
        split='test', N_assets=n_assets,
    )
    train_ds = MultivariateRealDataset(
        pkl_path=pkl_path, pred_len=pred_len, context_len=CTX_LEN,
        split='train', N_assets=n_assets,
    )

    try:
        gts_ds = get_dataset(ds_name, regenerate=False)
        seasonality = get_seasonality(gts_ds.metadata.freq)
        if gts_ds.metadata.freq == 'D':
            seasonality = 7
    except Exception:
        seasonality = 1

    batch_train_dfs, batch_pred_dfs = [], []

    with torch.no_grad():
        for win_idx in range(len(ds)):
            sample = ds[win_idx]
            x_w = sample['x'].to(device)
            y_w = sample['y'].to(device)
            ts_x_w = sample['ts_x'].to(device)
            ts_y_w = sample['ts_y'].to(device)
            T_pred = y_w.shape[0]

            if n_assets > 1:
                data = prepare_multi_input(x_w, ts_x_w, ts_y_w, n_assets, device)
            else:
                # For univariate, iterate over assets
                pass  # handled below

            if n_assets > 1:
                output = model(data, prediction_length=T_pred)
                mu_s, sig_s = scale_data(output, SCALER)
                mu_np = mu_s.cpu().numpy()
                sig_np = sig_s.cpu().numpy()
                y_np_w = y_w.cpu().numpy()

                for ai in range(n_assets):
                    aid = f"asset_{ai}"
                    train_hist = train_ds[len(train_ds)-1]['x'][:, ai].numpy() if len(train_ds) > 0 else x_w[:, ai].cpu().numpy()
                    batch_train_dfs.append(pd.DataFrame({'id': [aid]*len(train_hist), 'target': train_hist}))

                    sigma_i = np.sqrt(np.clip(sig_np[ai], 1e-6, None))
                    crps_v = crps_gaussian(mu_np[ai], sigma_i, y_np_w[:, ai])
                    nll_v = nll_eval(torch.tensor(mu_np[ai]), torch.tensor(sig_np[ai]),
                                     torch.tensor(y_np_w[:, ai])).numpy()

                    batch_pred_dfs.append(pd.DataFrame({
                        'id': [aid]*T_pred, 'pred': mu_np[ai], 'target': y_np_w[:, ai],
                        'variance': sig_np[ai], 'nll': nll_v, 'crps': crps_v,
                    }))
            else:
                y_np_w = y_w.cpu().numpy()
                for ai in range(y_np_w.shape[1]):
                    aid = f"asset_{ai}"
                    data_u = prepare_uni_input(x_w[:, ai:ai+1], ts_x_w, ts_y_w, device)
                    output = model(data_u, prediction_length=T_pred)
                    mu_s, sig_s = scale_data(output, SCALER)
                    mu_np_i = mu_s[0].cpu().numpy()
                    sig_np_i = sig_s[0].cpu().numpy()

                    train_hist = train_ds[len(train_ds)-1]['x'][:, ai].numpy() if len(train_ds) > 0 else x_w[:, ai].cpu().numpy()
                    batch_train_dfs.append(pd.DataFrame({'id': [aid]*len(train_hist), 'target': train_hist}))

                    sigma_i = np.sqrt(np.clip(sig_np_i, 1e-6, None))
                    crps_v = crps_gaussian(mu_np_i, sigma_i, y_np_w[:, ai])
                    nll_v = nll_eval(torch.tensor(mu_np_i), torch.tensor(sig_np_i),
                                     torch.tensor(y_np_w[:, ai])).numpy()

                    batch_pred_dfs.append(pd.DataFrame({
                        'id': [aid]*T_pred, 'pred': mu_np_i, 'target': y_np_w[:, ai],
                        'variance': sig_np_i, 'nll': nll_v, 'crps': crps_v,
                    }))

    train_df = pd.concat(batch_train_dfs)
    pred_df = pd.concat(batch_pred_dfs)

    # Compute metrics (with robust inf/nan handling)
    try:
        mase_res = mase(pred_df, ['pred'], seasonality, train_df, 'id', 'target')
        mase_vals = mase_res['pred'].replace([np.inf, -np.inf], np.nan)
        mase_mean = float(mase_vals.mean(skipna=True))
    except Exception:
        mase_mean = float('nan')

    raw_crps = float(pred_df['crps'].replace([np.inf, -np.inf], np.nan).mean(skipna=True))
    mean_abs_t = float(train_df['target'].abs().mean(skipna=True))
    if mean_abs_t < 1e-8:
        mean_abs_t = 1.0
    mcrps = raw_crps / mean_abs_t

    return {
        'MASE':  mase_mean,
        'MAE':   float(mae(pred_df, ['pred'], 'id', 'target')['pred'].mean()),
        'RMSE':  float(rmse(pred_df, ['pred'], 'id', 'target')['pred'].mean()),
        'SMAPE': float(smape(pred_df, ['pred'], 'id', 'target')['pred'].mean()),
        'NLL':   float(pred_df['nll'].mean()),
        'mCRPS': mcrps,
    }


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print(" LOADING MODELS")
print("=" * 80)

model_uni   = load_model(CKPT_UNI, 1, device)
model_multi = load_model(CKPT_MULTI, N_ASSETS, device)
print(f"  Multi model: N_assets={model_multi.N_assets}")
print(f"  Uni   model: N_assets={model_uni.N_assets}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS LOOP — Process each dataset
# ══════════════════════════════════════════════════════════════════════════════

for ds_cfg in ANALYSIS_DATASETS:
    ds_name  = ds_cfg['name']
    pred_len = ds_cfg['pred_len']
    labels   = ds_cfg.get('labels', [f'Asset_{i}' for i in range(N_ASSETS)])

    pkl_path = os.path.join(REAL_VAL_DIR, f'{ds_name}_nopad_512.pkl')
    if not os.path.exists(pkl_path):
        print(f"\n  ⚠️ PKL missing for {ds_name} — generating...")
        os.system(f"python {os.path.join(PROJECT_ROOT, 'data', 'scripts', 'store_real_datasets.py')}")

    if not os.path.exists(pkl_path):
        print(f"  ❌ Still missing: {pkl_path}, skipping.")
        continue

    print(f"\n\n{'═' * 80}")
    print(f"  DATASET: {ds_name} (pred_len={pred_len})")
    print(f"{'═' * 80}")

    try:
        # ── Load test sample ────────────────────────────────────────────────
        n_total = get_total_assets(pkl_path)
        N = min(N_ASSETS, n_total)
        labels_use = labels[:N] if len(labels) >= N else [f'Asset_{i}' for i in range(N)]

        sample = build_test_sample(pkl_path, pred_len, N)
        x, y       = sample['x'], sample['y']
        ts_x, ts_y = sample['ts_x'], sample['ts_y']
        y_np       = y.numpy()  # Ground truth — original scale

        # ════════════════════════════════════════════════════════════════════
        # MULTIVARIATE (N=8) INFERENCE
        # ════════════════════════════════════════════════════════════════════
        with torch.no_grad():
            data_m = prepare_multi_input(x, ts_x, ts_y, N, device)
            out_m  = model_multi(data_m, prediction_length=pred_len)

        # Inverse-scale to original data range
        mu_m_scaled, sig_m_scaled = scale_data(out_m, SCALER)
        mu_m  = mu_m_scaled.cpu().numpy()   # (N, pred_len) — original scale
        sig_m = sig_m_scaled.cpu().numpy()   # (N, pred_len) — original scale variance

        # ════════════════════════════════════════════════════════════════════
        # UNIVARIATE (N=1) INFERENCE — per asset
        # ════════════════════════════════════════════════════════════════════
        mu_u  = np.zeros((N, pred_len))
        sig_u = np.zeros((N, pred_len))

        with torch.no_grad():
            for i in range(N):
                data_u = prepare_uni_input(x[:, i:i+1], ts_x, ts_y, device)
                out_u  = model_uni(data_u, prediction_length=pred_len)
                m_u, s_u = scale_data(out_u, SCALER)
                mu_u[i]  = m_u[0].cpu().numpy()
                sig_u[i] = s_u[0].cpu().numpy()

        # ════════════════════════════════════════════════════════════════════
        # PART 1: VARIANCE COMPARISON TABLE (printed before any plots)
        # ════════════════════════════════════════════════════════════════════
        print(f"\n  ┌{'─'*78}┐")
        print(f"  │  PART 1: PREDICTIVE CERTAINTY — N=1 vs N=8 (Original Scale){'':>17}│")
        print(f"  ├{'─'*78}┤")
        print(f"  │  {'Asset':<8}{'σ²_Uni':>12}{'σ²_Multi':>12}{'Δσ²':>10}"
              f"{'MAE_Uni':>12}{'MAE_Multi':>12}{'95%Cov_U':>10}{'95%Cov_M':>10} │")
        print(f"  ├{'─'*78}┤")

        reductions = []
        coverage_u_list, coverage_m_list = [], []
        mae_u_list, mae_m_list = [], []

        for i in range(N):
            gt_i = y_np[:, i]  # Ground truth in original scale
            su = np.mean(sig_u[i])
            sm = np.mean(sig_m[i])
            rd = (1 - sm / su) * 100 if su > 1e-12 else 0.0
            reductions.append(rd)

            # 95% coverage
            std_u_i = np.sqrt(np.abs(sig_u[i]))
            std_m_i = np.sqrt(np.abs(sig_m[i]))
            cv_u = np.mean((gt_i >= mu_u[i] - 2*std_u_i) & (gt_i <= mu_u[i] + 2*std_u_i)) * 100
            cv_m = np.mean((gt_i >= mu_m[i] - 2*std_m_i) & (gt_i <= mu_m[i] + 2*std_m_i)) * 100
            coverage_u_list.append(cv_u)
            coverage_m_list.append(cv_m)

            mae_ui = np.mean(np.abs(gt_i - mu_u[i]))
            mae_mi = np.mean(np.abs(gt_i - mu_m[i]))
            mae_u_list.append(mae_ui)
            mae_m_list.append(mae_mi)

            print(f"  │  {labels_use[i]:<8}{su:>12.6f}{sm:>12.6f}{rd:>+9.1f}%"
                  f"{mae_ui:>12.6f}{mae_mi:>12.6f}{cv_u:>9.1f}%{cv_m:>9.1f}% │")

        print(f"  ├{'─'*78}┤")
        avg_su  = np.mean([np.mean(sig_u[i]) for i in range(N)])
        avg_sm  = np.mean([np.mean(sig_m[i]) for i in range(N)])
        avg_rd  = np.mean(reductions)
        avg_mae_u = np.mean(mae_u_list)
        avg_mae_m = np.mean(mae_m_list)
        avg_cu  = np.mean(coverage_u_list)
        avg_cm  = np.mean(coverage_m_list)

        print(f"  │  {'AVG':<8}{avg_su:>12.6f}{avg_sm:>12.6f}{avg_rd:>+9.1f}%"
              f"{avg_mae_u:>12.6f}{avg_mae_m:>12.6f}{avg_cu:>9.1f}%{avg_cm:>9.1f}% │")
        print(f"  └{'─'*78}┘")

        print(f"\n  Summary:")
        print(f"    Average σ² reduction:  {avg_rd:+.1f}%")
        print(f"    Average 95% coverage:  N=1 ({avg_cu:.1f}%) vs N=8 ({avg_cm:.1f}%)")
        print(f"    Average MAE:           N=1 ({avg_mae_u:.6f}) vs N=8 ({avg_mae_m:.6f})")

        # ════════════════════════════════════════════════════════════════════
        # PART 2: LINE CHARTS — Original-scale ground truth with predictions
        # ════════════════════════════════════════════════════════════════════
        print(f"\n  Generating per-asset line charts (original scale)...")

        n_rows = (N + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(18, 5 * n_rows), sharex=True)
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(
            f"SC-Mamba: Predictive Certainty — N=1 (Univariate) vs N=8 (Multivariate)\n"
            f"Dataset: {ds_name} | Horizon: {pred_len} steps | Original Scale",
            fontsize=14, fontweight='bold'
        )

        t = np.arange(pred_len)

        for i in range(N):
            ax = axes[i // 2, i % 2]
            gt_i = y_np[:, i]

            # ── Ground truth — black solid line showing real temporal fluctuations
            ax.plot(t, gt_i, color='black', linewidth=2.2, label='Ground Truth',
                    zorder=5)

            # ── N=1 Univariate — blue dashed + light blue confidence band
            ax.plot(t, mu_u[i], color='#2980b9', linewidth=1.3, linestyle='--',
                    alpha=0.85, label='N=1 Prediction', zorder=4)
            std_u_i = np.sqrt(np.abs(sig_u[i]))
            ax.fill_between(t,
                            mu_u[i] - 2 * std_u_i,
                            mu_u[i] + 2 * std_u_i,
                            color='#3498db', alpha=0.10,
                            label=r'N=1 $\pm 2\sigma$', zorder=2)

            # ── N=8 SC-Mamba — red dashed + light red confidence band
            ax.plot(t, mu_m[i], color='#c0392b', linewidth=1.3, linestyle='--',
                    alpha=0.85, label='N=8 Prediction', zorder=4)
            std_m_i = np.sqrt(np.abs(sig_m[i]))
            ax.fill_between(t,
                            mu_m[i] - 2 * std_m_i,
                            mu_m[i] + 2 * std_m_i,
                            color='#e74c3c', alpha=0.10,
                            label=r'N=8 $\pm 2\sigma$', zorder=2)

            # ── Styling
            rd_i = reductions[i]
            mae_ui, mae_mi = mae_u_list[i], mae_m_list[i]
            ax.set_title(
                f"{labels_use[i]}  |  σ² reduction: {rd_i:+.1f}%  |  "
                f"MAE: {mae_ui:.4f} → {mae_mi:.4f}",
                fontsize=10, fontweight='bold'
            )
            ax.set_ylabel('Value (Original Scale)', fontsize=9)
            ax.grid(alpha=0.25, linestyle='-')
            ax.tick_params(labelsize=8)

            # Legend only on first two subplots (avoids clutter)
            if i < 2:
                ax.legend(fontsize=7, loc='best', framealpha=0.9)

        # Hide empty subplot if N is odd
        if N % 2 == 1:
            axes[n_rows - 1, 1].set_visible(False)

        axes[-1, 0].set_xlabel('Time Step', fontsize=10)
        if N % 2 == 0:
            axes[-1, 1].set_xlabel('Time Step', fontsize=10)

        plt.tight_layout()
        line_path = os.path.join(OUTPUT_DIR, f'20_predictive_certainty_{ds_name}.png')
        fig.savefig(line_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved → {line_path}")
        plt.close(fig)

        # ════════════════════════════════════════════════════════════════════
        # PART 3: VARIANCE BAR CHART COMPARISON
        # ════════════════════════════════════════════════════════════════════
        print(f"  Generating variance bar chart...")

        fig2, ax2 = plt.subplots(figsize=(12, 5))
        xp = np.arange(N)
        w = 0.35
        bu = [np.mean(sig_u[i]) for i in range(N)]
        bm = [np.mean(sig_m[i]) for i in range(N)]

        bars_u = ax2.bar(xp - w/2, bu, w, label='N=1 (Univariate)', color='#3498db', alpha=0.75, edgecolor='white')
        bars_m = ax2.bar(xp + w/2, bm, w, label='N=8 (SC-Mamba)',   color='#e74c3c', alpha=0.75, edgecolor='white')

        ax2.set_xlabel('Asset', fontsize=11)
        ax2.set_ylabel(r'Mean $\sigma^2$ (Original Scale)', fontsize=11)
        ax2.set_title(f'Per-Asset Epistemic Variance — {ds_name}\nN=1 (Univariate) vs N=8 (SC-Mamba)',
                       fontweight='bold', fontsize=12)
        ax2.set_xticks(xp)
        ax2.set_xticklabels(labels_use, fontsize=9)
        ax2.legend(fontsize=10)
        ax2.grid(axis='y', alpha=0.3)

        # Annotate reduction percentages
        for i in range(N):
            peak = max(bu[i], bm[i])
            color = '#27ae60' if reductions[i] > 0 else '#c0392b'
            ax2.text(i, peak * 1.03, f"{reductions[i]:+.0f}%",
                     ha='center', fontsize=8, fontweight='bold', color=color)

        plt.tight_layout()
        bar_path = os.path.join(OUTPUT_DIR, f'20_variance_bar_{ds_name}.png')
        fig2.savefig(bar_path, dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved → {bar_path}")
        plt.close(fig2)

        # ════════════════════════════════════════════════════════════════════
        # PART 4: QUANTITATIVE ABLATION (MASE / MAE / RMSE / SMAPE / NLL / mCRPS)
        # ════════════════════════════════════════════════════════════════════
        print(f"\n  Running quantitative ablation (canonical metrics)...")

        try:
            eval_pl = min(pred_len, 30)  # Capped to match zero-shot benchmark protocol
            print(f"    Evaluating N=1 (pred_len={eval_pl})...")
            res_uni = canonical_evaluate_ablation(model_uni, pkl_path, eval_pl, 1, device)

            print(f"    Evaluating N=8 (pred_len={eval_pl})...")
            res_multi = canonical_evaluate_ablation(model_multi, pkl_path, eval_pl, N, device)

            res_uni['Model']   = 'N=1 (Univariate)'
            res_multi['Model'] = 'N=8 (SC-Mamba)'

            df_abl = pd.DataFrame([res_uni, res_multi]).set_index('Model')
            metric_cols = ['MASE', 'MAE', 'RMSE', 'SMAPE', 'NLL', 'mCRPS']

            print(f"\n  ┌{'─'*76}┐")
            print(f"  │  ABLATION: {ds_name} — N=1 vs N=8 (pred_len={eval_pl}){' '*(38-len(ds_name))}│")
            print(f"  ├{'─'*76}┤")
            for line in df_abl[metric_cols].to_string(float_format='{:.4f}'.format).split('\n'):
                print(f"  │  {line:<73} │")
            print(f"  ├{'─'*76}┤")
            print(f"  │  ↓ lower is better for ALL metrics{' '*39}│")
            print(f"  ├{'─'*76}┤")

            base = df_abl.iloc[0]
            multi = df_abl.iloc[1]
            for m in metric_cols:
                d = (multi[m] - base[m]) / (abs(base[m]) + 1e-10) * 100
                icon = '🟢' if d < 0 else '🔴'
                print(f"  │  {icon}  {m:6s}: {d:+.1f}%{' '*(62-len(f'{d:+.1f}%'))}│")
            print(f"  └{'─'*76}┘")

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ❌ Ablation error: {e}")

    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  ❌ Dataset error for {ds_name}: {e}")


print(f"\n{'═' * 80}")
print(f"  PREDICTIVE CERTAINTY ANALYSIS COMPLETE")
print(f"  Outputs: {OUTPUT_DIR}")
print(f"{'═' * 80}")
