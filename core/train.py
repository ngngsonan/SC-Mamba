"""
Module to train the model
"""
import sys
import os

# CRITICAL: Programmatic workaround for Triton 3.x compiler bug on Colab T4 (Turing) GPUs.
# Prevents Triton from compiling TF32 paths that crash with `IndexError: map::at`.
os.environ["TRITON_F32_DEFAULT"] = "ieee"

from pathlib import Path

# Get the directory of the current script
current_dir = Path(__file__).parent

# Get the parent directory (project_root)
parent_dir = current_dir.parent

# Add the root directory to the sys.path to access 'data' natively
# Using a robust absolute path for sys.path.append
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import wandb
import torchmetrics
import yaml
import argparse
import torch
import torch.nn as nn
import numpy as np
import random
import torch.optim as optim
import time
import pprint
import time
import pprint
from core.models import SCMamba_Forecaster  # Modified for SC-Mamba
from create_train_test_batch import create_train_test_batch_dl
from real_data_val_pipeline import validate_on_real_dataset
from utils import SMAPEMetric, generate_model_save_name, avoid_constant_inputs
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

def nll_loss(mu, sigma2, target):
    """
    Negative Log-Likelihood for Multivariate Gaussian (Independent across series)
    mu: [Batch, Pred_Len]
    sigma2: [Batch, Pred_Len]
    target: [Batch, Pred_Len]
    """
    # Adding epsilon for numerical stability
    sigma2 = torch.clamp(sigma2, min=1e-6)
    loss = 0.5 * torch.log(2 * np.pi * sigma2) + 0.5 * ((target - mu) ** 2) / sigma2
    return loss.mean()



def train_model(config):
    print("config:")
    print(pprint.pformat(config))

    # Enable model-level diagnostics (padding, spectral layer) if diag_prints is set
    # Must be set before models.py is imported/used, since _is_diag() checks at runtime
    if config.get('diag_prints', False):
        os.environ['SC_MAMBA_DIAG'] = '1'

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])  # FIX: seed Python's random module for reproducibility

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'cuda device usage (before model load): {torch.cuda.memory_allocated() / 2**20}')

    # loading the datasets as dataloaders
    if config["debugging"]:
        available_cpus = 1
    else:
        available_cpus = os.cpu_count()

    base_model_configs = {
        "scaler": config['scaler'],
        "sin_pos_enc": config['sin_pos_enc'],
        "sin_pos_const": config['sin_pos_const'],
        "sub_day": config["sub_day"],
        "encoding_dropout": config["encoding_dropout"],
        "handle_constants_model": config["handle_constants_model"],
        }
    
    # Load the model
    if config.get('num_assets') is None:
        raise ValueError("num_assets must be provided in config for Spectral Causal filtering.")
        
    model = SCMamba_Forecaster(
        N_assets=config['num_assets'], 
        ssm_config={**base_model_configs, **config['ssm_config']}
    ).to(device)
    print("Using SCMamba_Forecaster (Spectral Variational Graph)")
    # Assuming your train_loader and test_loader are already defined
    if config['lr_scheduler'] == "cosine":
        optimizer = optim.AdamW(model.parameters(), lr=config["initial_lr"])
        if config["t_max"] == -1:
            config["t_max"] = config['num_epochs']
        scheduler = CosineAnnealingLR(optimizer, T_max=config['t_max'], eta_min=config['learning_rate'])
    elif config['lr_scheduler'] == "cosine_warm_restarts":
        optimizer = optim.AdamW(model.parameters(), lr=config["initial_lr"])
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config['t_max'], eta_min=config['learning_rate'])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])

    initial_epoch = 0
    best_val_loss = float('inf')
    # Load state dicts if we are resuming training — prefer best checkpoint over periodic save
    config['model_save_name'] = generate_model_save_name(config)

    # ── Checkpoint diagnostics ────────────────────────────────────────
    best_ckpt_path = f"{config['model_prefix']}/{config['model_save_name']}_best.pth"
    periodic_ckpt_path = f"{config['model_prefix']}/{config['model_save_name']}.pth"
    final_ckpt_path = f"{config['model_prefix']}/{config['model_save_name']}_Final.pth"

    print(f"\n{'='*60}")
    print(f"  CHECKPOINT CONFIGURATION")
    print(f"{'='*60}")
    print(f"  model_save_name   : {config['model_save_name']}")
    print(f"  model_prefix      : {config['model_prefix']}")
    print(f"  continue_training : {config['continue_training']}")
    print(f"  best  checkpoint  : {best_ckpt_path}")
    print(f"       exists?      : {os.path.exists(best_ckpt_path)}")
    print(f"  periodic checkpoint: {periodic_ckpt_path}")
    print(f"       exists?      : {os.path.exists(periodic_ckpt_path)}")
    print(f"  final checkpoint  : {final_ckpt_path}")
    print(f"       exists?      : {os.path.exists(final_ckpt_path)}")
    print(f"{'='*60}\n")

    resume_path = best_ckpt_path if os.path.exists(best_ckpt_path) else periodic_ckpt_path
    if config['continue_training'] and os.path.exists(resume_path):
        print(f'✅ Loading checkpoint: {resume_path}')
        ckpt = torch.load(resume_path, map_location=device)
        # load states
        ckpt_keys = list(ckpt.keys())
        print(f'   Checkpoint keys: {ckpt_keys}')
        print(f'   Checkpoint epoch: {ckpt.get("epoch", "N/A")}')
        print(f'   Checkpoint best_val_loss: {ckpt.get("best_val_loss", "N/A")}')

        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if config['lr_scheduler'] in ('cosine', 'cosine_warm_restarts'):
            if ckpt.get('scheduler_state_dict') is not None:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                print(f'   Scheduler state loaded (last_lr={scheduler.get_last_lr()[0]:.2e})')
            else:
                # Fallback for old checkpoints that saved None due to bug
                if config['lr_scheduler'] == 'cosine_warm_restarts':
                    scheduler.step(ckpt.get('epoch', 0))
                elif config['lr_scheduler'] == 'cosine':
                    for _ in range(ckpt.get('epoch', 0)):
                        scheduler.step()
                print(f"   ⚠️  Recovered lost scheduler state by fast-forwarding to epoch {ckpt.get('epoch', 0)}")
        initial_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f'   ▶ Resuming from epoch {initial_epoch}, best_val_loss={best_val_loss:.4f}')
    else:
        if config['continue_training']:
            print(f'⚠️  continue_training=True but no checkpoint found at:')
            print(f'     {best_ckpt_path}')
            print(f'     {periodic_ckpt_path}')
        print('Starting fresh training (no checkpoint loaded)')
        model = model.to(device)
    
    # Fast-forward RNG state to avoid repeating identical synthetic data batches when resuming
    if initial_epoch > 0:
        new_seed = config['seed'] + initial_epoch * 1000
        print(f"Advancing random seeds by {initial_epoch} epochs to avoid repeating data on resume.")
        torch.manual_seed(new_seed)
        np.random.seed(new_seed % (2**32 - 1))
        random.seed(new_seed % (2**32 - 1))
    
    train_dataloader, test_dataloader = create_train_test_batch_dl(config=config,
                                                                   initial_epoch=initial_epoch,
                                                                   cpus_available=available_cpus,
                                                                   device=device,
                                                                   multipoint=config['multipoint'])

    print(f'cuda device usage (after model load): {torch.cuda.memory_allocated() / 2**20}')
    config['model_param_size'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size: {config['model_param_size']}")
    #wandb hyperparam init
    if config["wandb"]:
        run = wandb.init(
            project="SeriesPFN",
            # Track hyperparameters and run metadata
            config=config,
            name=config['model_save_name']
        )

    
    criterion = nn.L1Loss().to(device) if (config["loss"] == "mae") else nn.MSELoss().to(device) # For Mean Squared Error Loss
    # Metric initialization
    train_mape = torchmetrics.MeanAbsolutePercentageError().to(device)
    train_mse = torchmetrics.MeanSquaredError().to(device)
    train_smape = SMAPEMetric().to(device)
    
    val_mape = torchmetrics.MeanAbsolutePercentageError().to(device)
    val_mse = torchmetrics.MeanSquaredError().to(device)
    val_smape = SMAPEMetric().to(device)
 
    print(f'cuda device usage (before training start): {torch.cuda.memory_allocated() / 2**20}') 

    # Training Loop 
    for epoch in range(initial_epoch, config['num_epochs']):
        print("============Training==================")
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        running_nll_loss = 0.0
        running_kl_loss = 0.0
        train_epoch_loss = 0.0
        full_epoch_accumulated_loss = 0.0  # FIX: true full-epoch loss for WandB logging
        epoch_nll_sum = 0.0      # DIAG: track NLL component separately
        epoch_kl_sum = 0.0       # DIAG: track KL component separately
        epoch_grad_norm_sum = 0.0  # DIAG: track gradient norms
        epoch_sigma2_min = float('inf')  # DIAG: track sigma2 range
        epoch_sigma2_max = 0.0
        
        # New Diagnostics
        epoch_tau_sum = 0.0
        epoch_alpha_sum = 0.0
        epoch_sparsity_sum = 0.0
        epoch_collapse_sum = 0.0

        batch_idx = 0
        if config.get('diag_prints', False):
            print("Waiting for first batch from dataloader...", flush=True)
        for batch_id, batch in enumerate(train_dataloader):
            # if config.get('diag_prints', False):
            #     print(f"[{time.time() - epoch_start_time:.2f}s] Fetched batch {batch_id}", flush=True)
            data, target = {k: v.to(device) for k, v in batch.items() if k != 'target_values'}, batch['target_values'].to(device)           
            avoid_constant_inputs(data['history'], target)
            
            pred_len = target.size(1)
            optimizer.zero_grad()
            if isinstance(model, SCMamba_Forecaster):
                drop_enc_allow = True
                if config["sample_multi_pred"] > np.random.rand():
                    drop_enc_allow = False
                    # randomly sample 2 numbers between 4 and pred_length (inclusive)
                    pred_limits = np.random.randint(4, pred_len+1, 2)
                    start_pred = min(pred_limits)
                    end_pred = max(pred_limits)

                    if end_pred == start_pred:
                        if start_pred == pred_len:
                            start_pred = start_pred - 1
                        else:
                            end_pred = end_pred + 1
                    pred_len = end_pred - start_pred
                    target = target[:, start_pred:end_pred].contiguous()
                    data['target_dates'] = data['target_dates'][:, start_pred:end_pred].contiguous()
                    data['complete_target'] = data['complete_target'][:, start_pred:end_pred].contiguous()
                    data['task'] = data['task'][:, start_pred:end_pred].contiguous()
                # if config.get('diag_prints', False):
                #     print(f"[{time.time() - epoch_start_time:.2f}s] Running forward pass (Mamba2 compilation may take up to 10 mins on first batch)...", flush=True)
                output = model(data, prediction_length=pred_len)
                # if config.get('diag_prints', False):
                #     print(f"[{time.time() - epoch_start_time:.2f}s] Forward pass complete", flush=True)
                # DIAG: log first batch shapes and spectral layer info
                if batch_idx == 0 and config.get('diag_prints', False):
                    print(f"  [DIAG] history={data['history'].shape}, pred_len={pred_len}, mu={output['mu'].shape}, sigma2 range=[{output['sigma2'].min().item():.6f}, {output['sigma2'].max().item():.4f}], kl={output['kl_loss'].item():.6f}")
            else:
                # if config.get('diag_prints', False):
                #     print(f"[{time.time() - epoch_start_time:.2f}s] Running forward pass...", flush=True)
                output = model(data, prediction_length=pred_len)
                # if config.get('diag_prints', False):
                #     print(f"[{time.time() - epoch_start_time:.2f}s] Forward pass complete", flush=True)

            if config['scaler'] == 'min_max':
                max_scale = output['scale'][0].squeeze(-1)
                min_scale = output['scale'][1].squeeze(-1)
                scaled_target = (target - min_scale) / (max_scale - min_scale)
            else:                
                scaled_target = (target - output['scale'][0].squeeze(-1)) / output['scale'][1].squeeze(-1)

            # Calculate Loss Components
            nll_loss_val = nll_loss(output['mu'], output['sigma2'], scaled_target.float())
            kl_divergence = output['kl_loss']

            # beta_kl annealing: ramp 0 → beta_kl_target over the first half of training.
            # Rationale: starting with β=0 lets the model first learn a good NLL landscape
            # before the KL term constrains the spectral distribution. This prevents
            # KL Collapse (model trivially maps μ_F → 0, σ_F → 1 to zero KL at the cost
            # of learning no cross-asset structure). See β-VAE / NVAE annealing literature.
            beta_target = config.get('beta_kl', 0.1)
            beta_anneal_epochs = config.get('beta_anneal_epochs', config['num_epochs'] // 2)
            global_step = epoch * config['training_rounds'] + batch_idx
            total_warmup_steps = beta_anneal_epochs * config['training_rounds']
            beta = min(beta_target, beta_target * global_step / max(total_warmup_steps, 1))
            
            loss = nll_loss_val + beta * kl_divergence

            loss.backward()
            # Clip gradients to prevent NLL instability when σ² approaches floor (1e-6 → gradient ≈ 1e6)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # DIAG: accumulate per-epoch diagnostics
            epoch_nll_sum += nll_loss_val.item()
            epoch_kl_sum += kl_divergence.item()
            epoch_grad_norm_sum += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            s2_min = output['sigma2'].min().item()
            s2_max = output['sigma2'].max().item()
            if s2_min < epoch_sigma2_min: epoch_sigma2_min = s2_min
            if s2_max > epoch_sigma2_max: epoch_sigma2_max = s2_max
            
            if 'spectral_stats' in output:
                epoch_tau_sum += output['spectral_stats']['tau'].item()
                epoch_alpha_sum += output['spectral_stats']['alpha'].item()
                epoch_sparsity_sum += output['spectral_stats']['sparsity'].item()
            
            # Collapse tracking: sigma2 < 1e-3
            collapse_rate = (output['sigma2'] < 1e-3).float().mean().item()
            epoch_collapse_sum += collapse_rate
            
            if config['scaler'] == 'min_max':
                inv_scaled_output = (output['mu'] * (max_scale - min_scale)) + min_scale
            else:
                inv_scaled_output = (output['mu'] * output['scale'][1].squeeze(-1)) + output['scale'][0].squeeze(-1)
            
            inv_scaled_output = inv_scaled_output.detach()

            # Update metrics
            train_mape.update(inv_scaled_output, target)
            train_mse.update(inv_scaled_output, target)
            train_smape.update(inv_scaled_output, target)
            running_loss += loss.item()
            running_nll_loss += nll_loss_val.item()
            running_kl_loss += kl_divergence.item()
            full_epoch_accumulated_loss += loss.item()  # FIX: track full-epoch sum

            if batch_idx == config['training_rounds'] - 1:
                train_epoch_loss = full_epoch_accumulated_loss / config['training_rounds']

            if batch_idx % 10 == 9:  # Log every 10 batches
                avg_loss = running_loss / 10
                avg_nll = running_nll_loss / 10
                avg_kl = running_kl_loss / 10
                print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}, Total Loss: {avg_loss:.4f}, NLL: {avg_nll:.4f}, KL: {avg_kl:.4f} From torchmetric: {train_mse.compute():.4f}')
                if config["wandb"]:    
                    wandb_dict = {
                        'train_batch_metrics': {
                            'sc_loss': avg_loss, 
                            'nll': avg_nll, 
                            'kl': avg_kl, 
                            'mape': train_mape.compute(), 
                            'smape': train_smape.compute()
                        },
                        'step': epoch * config['training_rounds'] + batch_idx
                    }
                    if 'spectral_stats' in output and config.get('diag_prints', False):
                        wandb_dict['train_batch_metrics']['tau'] = output['spectral_stats']['tau'].item()
                        wandb_dict['train_batch_metrics']['alpha'] = output['spectral_stats']['alpha'].item()
                        wandb_dict['train_batch_metrics']['sparsity'] = output['spectral_stats']['sparsity'].item()
                        wandb_dict['train_batch_metrics']['sigma2_collapse'] = collapse_rate
                    wandb.log(wandb_dict)
                running_loss = 0.0
                running_nll_loss = 0.0
                running_kl_loss = 0.0
            
            batch_idx += 1
            #end of epoch at max training rounds
            if batch_idx == config['training_rounds']:
                break

        # Validation loop (after each epoch)
        print("============Validation==================")
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            val_batch_idx = 0
            for batch_id, batch in enumerate(test_dataloader):
                data, target = {k: v.to(device) for k, v in batch.items() if k != 'target_values'}, batch['target_values'].to(device)
                avoid_constant_inputs(data['history'], target)
                pred_len = target.size(1)
                output = model(data, prediction_length=pred_len)
                
                if config['scaler'] == 'min_max':
                    max_scale = output['scale'][0].squeeze(-1)
                    min_scale = output['scale'][1].squeeze(-1)
                    scaled_target = (target - min_scale) / (max_scale - min_scale)
                else:                
                    scaled_target = (target - output['scale'][0].squeeze(-1)) / output['scale'][1].squeeze(-1)
                
                # Validation Loss (NLL) — use annealed beta consistent with training
                val_nll = nll_loss(output['mu'], output['sigma2'], scaled_target.float()).item()
                val_loss = val_nll + beta * output['kl_loss'].item()
                total_val_loss += val_loss

                if batch_id % 10 == 9:
                    print(f'val loss for batch {batch_id}: {val_loss}')

                if config['scaler'] == 'min_max':
                    inv_scaled_output = (output['mu'] * (max_scale - min_scale)) + min_scale
                else:
                    inv_scaled_output = (output['mu'] * output['scale'][1].squeeze(-1)) + output['scale'][0].squeeze(-1)
                # Update validation metrics
                val_mape.update(inv_scaled_output, target)
                val_mse.update(inv_scaled_output, target)
                val_smape.update(inv_scaled_output, target)

                val_batch_idx += 1  # FIX: was never incremented, causing only 1 validation batch
                if val_batch_idx == config['validation_rounds']:
                    break

        # Compute and log validation metrics
        avg_val_loss = total_val_loss / max(1, val_batch_idx)
        print(f'Epoch: {epoch+1}, SC. Validation Loss: {avg_val_loss:.4f} From torchmetric: {val_mse.compute():.4f}')
        if config["wandb"]:    
            wandb.log({'epoch_metrics': {
                    'train': {'sc_loss': train_epoch_loss,'mape': train_mape.compute(),'smape': train_smape.compute()},
                    'val': {'sc_loss': avg_val_loss,'mape': val_mape.compute(),'smape': val_smape.compute()}
                    },
                   'epoch': epoch,
                   'lr': optimizer.param_groups[0]['lr']})
        
        epoch_time = time.time() - epoch_start_time
        print(f'Time taken for epoch: {epoch_time/60:.1f} mins {epoch_time%60:.0f} secs.')

        # ── DIAGNOSTIC EPOCH SUMMARY ──────────────────────────────────
        if config.get('diag_prints', False) and batch_idx > 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  ╔══════════════ EPOCH {epoch+1} DIAGNOSTICS ══════════════╗")
            print(f"  ║ LR           = {current_lr:.2e}")
            print(f"  ║ beta (KL wt) = {beta:.4f}")
            print(f"  ║ avg NLL      = {epoch_nll_sum / batch_idx:.4f}")
            print(f"  ║ avg KL       = {epoch_kl_sum / batch_idx:.4f}")
            print(f"  ║ avg KL×beta  = {beta * epoch_kl_sum / batch_idx:.6f}")
            print(f"  ║ avg Loss     = {full_epoch_accumulated_loss / batch_idx:.4f}")
            print(f"  ║ avg GradNorm = {epoch_grad_norm_sum / batch_idx:.4f}")
            print(f"  ║ sigma2 range = [{epoch_sigma2_min:.6f}, {epoch_sigma2_max:.4f}]")
            if epoch_tau_sum > 0 or epoch_sparsity_sum >= 0:
                print(f"  ║ avg Tau      = {epoch_tau_sum / batch_idx:.4f} (Alpha = {epoch_alpha_sum / batch_idx:.2f})")
                print(f"  ║ Mask Sparsity= {epoch_sparsity_sum / batch_idx * 100:.2f}%")
                print(f"  ║ S2 Collapse  = {epoch_collapse_sum / batch_idx * 100:.2f}% (<1e-3)")
            print(f"  ║ val_loss     = {avg_val_loss:.4f}")
            print(f"  ║ best_val     = {best_val_loss:.4f}")
            print(f"  ╚═══════════════════════════════════════════════════════╝")

        # Reset metrics for the next epoch
        train_mape.reset()
        train_mse.reset()
        train_smape.reset()
        val_mape.reset()
        val_mse.reset()
        val_smape.reset()

        # Save best checkpoint whenever validation improves
        # FIX: Guard against val_batch_idx=0 (empty dataloader) producing spurious 0.0 loss
        if val_batch_idx > 0 and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            has_scheduler = config['lr_scheduler'] in ('cosine', 'cosine_warm_restarts')
            best_ckpt = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if has_scheduler else None,
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'ssm_config': config.get('ssm_config', {}),
            }
            os.makedirs(config['model_prefix'], exist_ok=True)
            torch.save(best_ckpt, f"{config['model_prefix']}/{config['model_save_name']}_best.pth")
            print(f'  🏆 Best checkpoint saved (epoch {epoch+1}, val_loss={best_val_loss:.4f})')

        if epoch % config['real_test_interval'] == config['real_test_interval'] - 1:
            res_dict = {
                'real_dataset_metrics': {
                    'mase': {}, 'mae': {}, 'rmse': {}, 'smape': {},
                    'nll': {}, 'crps': {},   # Probabilistic metrics unique to SC-Mamba
                },
                'epoch': epoch
            }
            for real_dataset in config['real_test_datasets']:
                print(f'Evaluating on real dataset: {real_dataset}')
                real_mase, real_mae, real_rmse, real_smape, real_nll, real_crps = validate_on_real_dataset(
                    real_dataset, model, device, config['scaler'], subday=config["sub_day"]
                )
                print(
                    f"MASE: {real_mase:.4f}, MAE: {real_mae:.4f}, "
                    f"RMSE: {real_rmse:.4f}, SMAPE: {real_smape:.4f}, "
                    f"NLL: {real_nll:.4f}, CRPS: {real_crps:.4f}"
                )
                res_dict['real_dataset_metrics']['mase'][real_dataset] = real_mase
                res_dict['real_dataset_metrics']['mae'][real_dataset] = real_mae
                res_dict['real_dataset_metrics']['rmse'][real_dataset] = real_rmse
                res_dict['real_dataset_metrics']['smape'][real_dataset] = real_smape
                res_dict['real_dataset_metrics']['nll'][real_dataset] = real_nll
                res_dict['real_dataset_metrics']['crps'][real_dataset] = real_crps
                if config["wandb"]:
                    wandb.log(res_dict)
            # FIX: Explicitly restore to train mode after real dataset evaluation
            # validate_on_real_dataset calls model.eval() internally; next epoch's model.train()
            # at the top of the loop covers this, but being explicit here is safer.
            model.train()
        
        # LR Scheduler step — separated by scheduler type to avoid calling
        # CosineAnnealingWarmRestarts without the required epoch arg.
        if config['lr_scheduler'] == "cosine":
            if scheduler.get_last_lr()[0] <= config['learning_rate'] + 1e-10:
                print("Learning rate has reached the minimum value. No more steps.")
            else:
                scheduler.step()
        elif config['lr_scheduler'] == "cosine_warm_restarts":
            # CosineAnnealingWarmRestarts expects epoch as the fractional epoch index
            scheduler.step(epoch + 1)
            
        if epoch % 5 == 4:
            has_scheduler = config['lr_scheduler'] in ('cosine', 'cosine_warm_restarts')
            ckpt = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if has_scheduler else None,
                'epoch': epoch,
                'best_val_loss': best_val_loss,  # FIX: Preserve best_val_loss across session breaks
                'ssm_config': config.get('ssm_config', {}),
            }
            torch.save(ckpt, f"{config['model_prefix']}/{config['model_save_name']}.pth")

    if config["wandb"]:
        wandb.finish()

    # Save the final model — guard against empty loop (initial_epoch >= num_epochs)
    final_epoch = epoch if initial_epoch < config['num_epochs'] else max(initial_epoch - 1, 0)
    if initial_epoch < config['num_epochs']:
        has_scheduler = config['lr_scheduler'] in ('cosine', 'cosine_warm_restarts')
        ckpt = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if has_scheduler else None,
            'epoch': final_epoch,
            'best_val_loss': best_val_loss,
            'ssm_config': config.get('ssm_config', {}),
        }
        torch.save(ckpt, f"{config['model_prefix']}/{config['model_save_name']}_Final.pth")
    else:
        print(f"⚠️  No training epochs ran (initial_epoch={initial_epoch} >= num_epochs={config['num_epochs']}). Final checkpoint NOT saved.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="./config.batch_ddp.yaml", help="Path to config file")
    parser.add_argument("--diag_prints", action="store_true", help="Enable verbose diagnostic prints (timing, shapes, spectral layer stats)")
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.load(config_file, yaml.loader.SafeLoader)
    
    # Inject diag_prints into config (CLI flag overrides YAML)
    if args.diag_prints:
        config['diag_prints'] = True
    
    # for wandb offline mode (can comment if needed):
    os.environ['WANDB_MODE'] = "online" if config['wandb'] else 'offline'
    train_model(config)
