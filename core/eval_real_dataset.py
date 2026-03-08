import torch
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import yaml
import argparse
import submitit
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.time_feature.seasonality import get_seasonality
from data.data_provider.data_factory import data_provider
from utilsforecast.losses import mase, mae, smape, rmse
import csv
from core.models import SCMamba_Forecaster
from tqdm import tqdm
import time
from scipy.stats import norm as scipy_norm

REAL_DATASETS = {
    "nn5_daily_without_missing": 56,
    "nn5_weekly": 8,
    "covid_deaths": 30,
    "weather": 30,
    "hospital": 12,
    "fred_md": 12,
    "car_parts_without_missing": 12,
    "traffic": 24,
    "m3_monthly": 18,
    "ercot": 24,
    "m1_monthly": 18,
    "m1_quarterly": 8,
    "cif_2016": 12,
    "exchange_rate": 30,
    "m3_quarterly": 8,
    "tourism_monthly": 24,
    "tourism_quarterly": 8,
}

REAL_DATASET_ASSETS = {
    "nn5_daily_without_missing": 111,
    "nn5_weekly": 111,
    "covid_deaths": 108,
    "weather": 21,
    "hospital": 767,
    "fred_md": 107,
    "car_parts_without_missing": 2674,
    "traffic": 862,
    "m3_monthly": 1428,
    "ercot": 39,
    "m1_monthly": 617,
    "m1_quarterly": 203,
    "cif_2016": 72,
    "exchange_rate": 8,
    "m3_quarterly": 756,
    "tourism_monthly": 366,
    "tourism_quarterly": 427,
}

MAX_LENGTH = 512

ssm_config = {
    "bidirectional":False,
    "enc_conv" : True,
    "init_dil_conv" : True,
    "enc_conv_kernel" : 5,
    "init_conv_kernel" : 5,
    "init_conv_max_dilation" : 3,
    "global_residual":False,
    "in_proj_norm":False,
    "initial_gelu_flag":True,
    "linear_seq":15,
    "mamba2":True,
    "norm":True,
    "norm_type":"layernorm",
    "num_encoder_layers":2,
    "d_state":128,
    "residual":False,
    "token_embed_len":1024,
}
# model_name and sub_day will be parsed logically or passed dynamically
DEFAULT_MODEL_NAME = "latest_mi_2l_1024e_nores_30-512cl_m2_norm_dconv_i5e5_lr1e-07_mp0.5_initlr1e-05_t300r_pm0.7_nPer_pl10-60_subday_subfreq0.2_perfreq0.5_d0.05_s0.05"

def set_queue(q_, log_folder, maximum_runtime=None):
    global ex
    global q
    if q_ == 'all':
        q = 'alldlc_gpu-rtx2080'
    if q_ == 'ml':
        q = 'mldlc_gpu-rtx2080'
    if q_ == 'mlhiwi':
        q = "mlhiwidlc_gpu-rtx2080"

    if maximum_runtime is None:
        if q == 'alldlc_gpu-rtx2080' or q == 'mlhiwidlc_gpu-rtx2080':
            maximum_runtime = 24*60*1-1
        else:
            maximum_runtime = 24*60*4-1

    ex = submitit.AutoExecutor(folder=log_folder)
    ex.update_parameters(timeout_min=maximum_runtime,
                        slurm_partition=q, #  mldlc_gpu-rtx2080
                        slurm_signal_delay_s=180, # time to pass the USR2 signal to slurm before the job times out so that it can finish the run
                        tasks_per_node=1,
                        nodes=1,
                        cpus_per_task=30, #24
                        mem_per_cpu=4096,
                        slurm_gres=f'gpu:{1}'
       )

    return maximum_runtime

def scale_data(output, scaler):
    # Scale both mu and sigma2
    if scaler == 'custom_robust':
        scaled_mu = (output['mu'] * output['scale'][1].squeeze(-1)) + output['scale'][0].squeeze(-1)
        scaled_sigma2 = output['sigma2'] * (output['scale'][1].squeeze(-1) ** 2)
    elif scaler == 'min_max':
        scaling_span = (output['scale'][0].squeeze(-1) - output['scale'][1].squeeze(-1))
        scaled_mu = (output['mu'] * scaling_span) + output['scale'][1].squeeze(-1)
        scaled_sigma2 = output['sigma2'] * (scaling_span ** 2)
    elif scaler == 'identity':
        scaled_mu = output['mu']
        scaled_sigma2 = output['sigma2']
        
    return scaled_mu, scaled_sigma2
    
def nll_eval(mu, sigma2, target):
    """
    Per-element Gaussian NLL used as a probabilistic quality metric at evaluation.
    Not used for gradient computation (eval only).
    """
    sigma2 = torch.clamp(sigma2, min=1e-6)
    loss = 0.5 * torch.log(torch.tensor(2 * np.pi) * sigma2) + 0.5 * ((target - mu) ** 2) / sigma2
    return loss


def crps_gaussian(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Closed-form CRPS for a Gaussian predictive distribution.

    CRPS(N(mu, sigma^2), y) = sigma * [
        (z * (2*Phi(z) - 1)) + 2*phi(z) - 1/sqrt(pi)
    ]
    where z = (y - mu) / sigma, Phi = Gaussian CDF, phi = Gaussian PDF.

    References
    ----------
    Gneiting & Raftery (2007), "Strictly Proper Scoring Rules, Prediction, and Estimation".
    Jordan et al. (2019), properscoring library closed-form derivation.

    Parameters
    ----------
    mu     : point forecast mean,  shape [...]
    sigma  : predictive std-dev (> 0), shape [...]
    y      : ground-truth observation, shape [...]

    Returns
    -------
    crps   : per-element CRPS scores, shape [...] (lower is better)
    """
    sigma = np.clip(sigma, a_min=1e-6, a_max=None)
    z = (y - mu) / sigma
    crps = sigma * (
        z * (2.0 * scipy_norm.cdf(z) - 1.0)
        + 2.0 * scipy_norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )
    return crps


def auto_regressive_predict(model, batch_x, batch_y, batch_x_mark, batch_y_mark, eval_pred_len, real_data_args, scaler, device):
        # decoder input
        dec_inp = torch.zeros_like(
            batch_y[-eval_pred_len:]).unsqueeze(0).int().to(device)

        x = {}
        x['ts'] = batch_x_mark.to(device)
        x['history'] = batch_x.reshape(1,batch_x.size(1)).to(device)
        outputs = []
        for pred_ind in range(0, eval_pred_len):
            x['target_dates'] = batch_y_mark[:, pred_ind].unsqueeze(1).to(device)
            x['task'] = dec_inp[:, pred_ind].unsqueeze(1)
            multipoint = isinstance(model, SCMamba_Forecaster)
            if multipoint:
                output = model(x, prediction_length=1)
            else:
                output = model(x)
                
            output = scale_data(output, scaler)
            outputs.append(output)
            
            x['history'] = torch.cat([x['history'], output], dim=1)
            x['ts'] = torch.cat([x['ts'], x['target_dates']], dim=1)
    
        outputs = torch.stack(outputs, dim=1).detach().cpu().squeeze()
            
        return outputs
    

def batch_predict(model, batch_x, batch_x_mark, batch_y_mark, eval_pred_len, scaler, device):
    x = {}
    x['ts'] = batch_x_mark.repeat(eval_pred_len, 1, 1).to(device)
    x['history'] = batch_x.reshape(1,batch_x.size(1)).repeat(eval_pred_len, 1).to(device)

    x['target_dates'] = batch_y_mark.transpose(0, 1).to(device)
    x['task'] = torch.zeros(eval_pred_len,1).int().to(device)

    output = model(x)
    scaled_mu, scaled_sigma2 = scale_data(output, scaler)
    
    mu_out = scaled_mu.detach().cpu().squeeze()
    sigma2_out = scaled_sigma2.detach().cpu().squeeze()
    
    return mu_out, sigma2_out

def multipoint_predict(model, batch_x, batch_x_mark, batch_y_mark, pred_len, scaler, device):
    x = {}
    x['ts'] = batch_x_mark.to(device)
    x['history'] = batch_x.reshape(1,batch_x.size(1)).to(device)
    x['target_dates'] = batch_y_mark.to(device)
    x['task'] = torch.zeros(1,pred_len).int().to(device)
    
    output = model(x, prediction_length=pred_len)
    scaled_mu, scaled_sigma2 = scale_data(output, scaler)
    
    mu_out = scaled_mu.detach().cpu().squeeze()
    sigma2_out = scaled_sigma2.detach().cpu().squeeze()
    return mu_out, sigma2_out


def efficient_ensemble_predict(model, batch_x, batch_x_mark, batch_y_mark, pred_len, scaler, device, ensemble_config):
    x = {}
    x['ts'] = batch_x_mark.to(device).repeat(ensemble_config["num_ensembles"], 1, 1)
    x['history'] = batch_x.reshape(1,batch_x.size(1)).to(device).repeat(ensemble_config["num_ensembles"], 1)
    x['target_dates'] = batch_y_mark.to(device).repeat(ensemble_config["num_ensembles"], 1, 1)
    x['task'] = torch.zeros(1,pred_len).int().to(device).repeat(ensemble_config["num_ensembles"], 1)
    
    cl = x["history"].size(1)
    
    drops = np.linspace(0,cl*ensemble_config["max_fraction"], ensemble_config["num_ensembles"]).astype(int)
    assert isinstance(model, SCMamba_Forecaster), "Model must be an instance of SCMamba_Forecaster"
  
    outputs = []
    z = x.copy()
    for i, drop in enumerate(drops):
        drop_indices = np.random.choice(cl, size=drop, replace=False)  # Randomly choose 'drop' indices to remove
        keep_indices = np.setdiff1d(np.arange(cl), drop_indices)
        z['ts'][i, drop:, :] = x['ts'][i, keep_indices, :]
        z['ts'][i, :drop, :] = x['ts'][i, keep_indices, :].min(dim=0, keepdim=True).values
        z['history'][i, drop:] = x['history'][i, keep_indices]
        z['history'][i, :drop] = x['history'][i, keep_indices].min(dim=0, keepdim=True).values
    output = model(z, pred_len)
    output = scale_data(output, scaler)
    outputs = output.detach().cpu().squeeze()
    if ensemble_config["method"] == "median":
        output = outputs.median(dim=0)
    else:
        output = outputs.mean(dim=0)
    output = output.detach().cpu().squeeze()
    return output


def ensemble_predict(model, batch_x, batch_x_mark, batch_y_mark, pred_len, scaler, device, ensemble_config):
    x = {}
    x['ts'] = batch_x_mark.to(device)
    x['history'] = batch_x.reshape(1,batch_x.size(1)).to(device)
    x['target_dates'] = batch_y_mark.to(device)
    x['task'] = torch.zeros(1,pred_len).int().to(device)
    
    cl = x["history"].size(1)
    
    drops = np.linspace(0,cl*ensemble_config["max_fraction"], ensemble_config["num_ensembles"]).astype(int)
    assert isinstance(model, SCMamba_Forecaster), "Model must be an instance of SCMamba_Forecaster"
  
    outputs = []
    z = {}
    for drop in drops:
        drop_indices = np.random.choice(cl, size=drop, replace=False)  # Randomly choose 'drop' indices to remove
        keep_indices = np.setdiff1d(np.arange(cl), drop_indices)
        z['ts'] = x['ts'][:, keep_indices, :]
        z['history'] = x['history'][:, keep_indices]
        z['target_dates'] = x['target_dates']
        z['task'] = x['task']
        output = model(z, pred_len)
        output = scale_data(output, scaler)
        output = output.detach().cpu().squeeze()
        outputs.append(output)
    outputs = torch.stack(outputs, dim=1)
    if ensemble_config["method"] == "median":
        output = outputs.median(dim=1)
    else:
        output = outputs.mean(dim=1)
    output = output.detach().cpu().squeeze()
    return output


def evaluate_real_dataset(dataset: str, model, scaler, context_len, eval_pred_len, device, pred_style=None):
    # Use absolute path for real_data_args.yaml (same dir as this script)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(base_dir, 'real_data_args.yaml')
    
    with open(config_file) as file:
        real_data_args = yaml.load(file, yaml.loader.SafeLoader)

    if pred_style is None:
        pred_style = real_data_args['pred_style']
        
    if real_data_args['pad']:
        real_data_args['data_path'] = dataset + f'_pad_{MAX_LENGTH}.pkl'
    else:
        real_data_args['data_path'] = dataset + f'_nopad_{MAX_LENGTH}.pkl'

    pred_len = REAL_DATASETS[dataset]
    real_data_args['data'] = dataset
    real_data_args['pred_len'] = pred_len
    
    # Extract sub_day from model_name dynamically
    sub_day = "subday" in real_data_args.get("model_name", "")
    test_dataset, test_dataloader = data_provider(real_data_args, real_data_args['flag'], subday=sub_day)

    gts_dataset = get_dataset(real_data_args['data'], regenerate=False)
    seasonality = get_seasonality(gts_dataset.metadata.freq) if gts_dataset.metadata.freq != 'D' else 7
    print(seasonality)

    batch_train_dfs = []
    batch_pred_dfs = []
    model.eval()
    j = 0
    print(f"pred_style: {pred_style}")
    with torch.no_grad():
        # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        for i, batch in tqdm(enumerate(test_dataloader)):
            cl = min(context_len, batch['x'].shape[1])
            ids = batch["id"]
            batch_x = batch["x"][:, -cl:, :].float()
            batch_y = batch["y"][:, :eval_pred_len, :].squeeze().float()

            batch_x_mark = batch["ts_x"][:, -cl:, :].float()
            batch_y_mark = batch["ts_y"][:, :eval_pred_len, :].float()
            
            if pred_style == 'multipoint':
                mu_out, sigma2_out = multipoint_predict(model, batch_x, batch_x_mark, batch_y_mark, eval_pred_len, scaler, device)
            else:
                # We default to multipoint for SCMamba since it handles sequence directly
                mu_out, sigma2_out = multipoint_predict(model, batch_x, batch_x_mark, batch_y_mark, eval_pred_len, scaler, device)
            
            # create dfs used to calculate the MASE such that the batch['x'] and pred are squeezed into 1 column and the ids are repeated 
            # for batch['x'].shape[1] times and pred_len times respectively
            batch_train_dfs.append(pd.DataFrame({
                'id': ids.repeat_interleave(batch_x.size(1)).numpy(),
                'target': batch_x.flatten().numpy()
            }))
            
            nll_vals = nll_eval(mu_out, sigma2_out, batch_y.cpu()).flatten().numpy()
            
            batch_pred_dfs.append(pd.DataFrame({
                'id': ids.repeat_interleave(eval_pred_len).numpy(),
                'pred': mu_out.flatten().numpy(),
                'target': batch_y.flatten().numpy(),
                'variance': sigma2_out.flatten().numpy(),
                'nll': nll_vals
            }))

    train_df = pd.concat(batch_train_dfs)
    pred_df = pd.concat(batch_pred_dfs)

    mase_loss = mase(pred_df, ['pred'], seasonality, train_df, 'id', 'target')
    mae_loss = mae(pred_df, ['pred'], 'id', 'target')
    rmse_loss = rmse(pred_df, ['pred'], 'id', 'target')
    smape_loss = smape(pred_df, ['pred'], 'id', 'target')
    mean_nll = pred_df['nll'].mean()

    # CRPS (Continuous Ranked Probability Score) — closed-form Gaussian CRPS.
    # This is the primary probabilistic metric distinguishing SC-Mamba from
    # deterministic baselines (Mamba4Cast, etc.) which cannot report CRPS.
    # Lower CRPS = better calibrated distributional forecast.
    mu_np = pred_df['pred'].values
    sigma_np = np.sqrt(np.clip(pred_df['variance'].values, 1e-6, None))
    y_np = pred_df['target'].values
    crps_vals = crps_gaussian(mu_np, sigma_np, y_np)
    mean_crps = float(crps_vals.mean())

    out_dict = {'mase': mase_loss['pred'].mean(), 'mae': mae_loss['pred'].mean(), 
                'rmse': rmse_loss['pred'].mean(), 'smape': smape_loss['pred'].mean(),
                'nll': mean_nll, 'crps': mean_crps}

    return out_dict, train_df, pred_df


def adapt_state_dict_keys(old_state_dict):
    new_state_dict = {}

    for key in old_state_dict.keys():
        if "linear_layer" in key:
            # Extract the layer index
            layer_idx = key.split('.')[1]
            
            # Replace "linear_layer" with "stage_2_layer.0"
            new_key = key.replace(f"linear_layer", f"stage_2_layer.0")
            
            # Add the updated key to the new state dict
            new_state_dict[new_key] = old_state_dict[key]
        else:
            # Keep other keys unchanged
            new_state_dict[key] = old_state_dict[key]

    return new_state_dict

def csv_writer(csv_file, result_dict):
    fieldnames = set()
    for entry in result_dict.values():
        fieldnames.update(entry.keys())
    # Convert set to list and sort or maintain any specific order if required
    fieldnames = list(fieldnames)
    fieldnames.insert(0, 'ID')
    # Open the file in write mode
    with open(csv_file, 'w', newline='') as csvfile:
        # Create a DictWriter object with the extended fieldnames
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header
        writer.writeheader()
        
        # Write the data
        for key, value in result_dict.items():
            # Insert the outer key into the row dictionary
            row = {'ID': key}
            row.update(value)
            writer.writerow(row)

def main_evaluator(pred_style=None, model_name=DEFAULT_MODEL_NAME, model_prefix='models', checkpoint_path=None):
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'real_data_args.yaml')
    with open(config_path) as file:
        real_data_args = yaml.load(file, yaml.loader.SafeLoader)
    real_data_args["model_name"] = model_name

    if pred_style is None:
        pred_style = real_data_args['pred_style']

    # --- Robust Checkpoint Resolution ---
    if checkpoint_path and os.path.exists(checkpoint_path):
        model_string = checkpoint_path
        print(f"🎯 Using explicit checkpoint: {model_string}")
    else:
        # Fallback search chain: _best.pth -> .pth -> _Final.pth
        candidates = [
            f"{model_prefix}/{model_name}_best.pth",
            f"{model_prefix}/{model_name}.pth",
            f"{model_prefix}/{model_name}_Final.pth"
        ]
        model_string = None
        for cand in candidates:
            if os.path.exists(cand):
                model_string = cand
                break
        
        if model_string is None:
            print(f"\n[ERROR] No checkpoint found for model '{model_name}' in prefix '{model_prefix}'", file=sys.stderr)
            print(f"Checked candidates: {candidates}", file=sys.stderr)
            sys.exit(1)
        print(f"✅ Resolved checkpoint: {model_string}")
    
    # Use model_name for result directory, but ensure it handles path-like model_names from checkpoint_path
    safe_model_name = os.path.basename(model_name) if checkpoint_path else model_name
    eval_dir = os.path.join(base_dir, f'../data/real_data_evals/{safe_model_name}/{pred_style}')
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    new_state_dict = adapt_state_dict_keys(torch.load(model_string, map_location=device, weights_only=True)['model_state_dict'])    
    context_lens = [512]
    
    for dataset_name in REAL_DATASETS.keys():
        
        n_assets = REAL_DATASET_ASSETS.get(dataset_name, 1)
        model = SCMamba_Forecaster(
            N_assets=n_assets, 
            ssm_config=ssm_config
        ).to(device)
        model.load_state_dict(new_state_dict, strict=False)
        
        res_dict = {}
        print(f'pred_style: {pred_style}')
        yml_path = os.path.join(eval_dir, f'{dataset_name}_{MAX_LENGTH}.yml')
        if not os.path.exists(yml_path):
            pred_lens = [REAL_DATASETS[dataset_name]] #range(1, REAL_DATASETS[dataset_name]+1)
            for cl in context_lens:
                for pl in pred_lens:
                    print(f'evaluating {dataset_name} for context length:{cl} and prediction length:{pl}')
                    start_time = time.time()
                    out_dict, train_df, pred_df = evaluate_real_dataset(dataset_name, model, 'min_max', cl, pl, device, pred_style=pred_style)
                    end_time = time.time()
                    res_dict[f'{cl}_{pl}'] = out_dict
                    print(f"Time taken by {pred_style}: {end_time - start_time}")
                    #saving train and pred dfs if its max pred_len of entire series
                    if pl == REAL_DATASETS[dataset_name]:
                        train_df.to_csv(os.path.join(eval_dir, f'{dataset_name}_train_df_{cl}_{pl}.csv'), index=False)
                        pred_df.to_csv(os.path.join(eval_dir, f'{dataset_name}_pred_df_{cl}_{pl}.csv'), index=False)
        
            with open(yml_path, 'w') as outfile:
                yaml.dump(res_dict, outfile, default_flow_style=True)
            
            # Specify the file name
            filename = os.path.join(eval_dir, f'{dataset_name}_{MAX_LENGTH}.csv')

            # Write the dictionary to the CSV file
            csv_writer(filename, res_dict)
                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--slurm", type=bool, default=False, help="flag to run training on slurm")
    parser.add_argument("-m", "--model_name", type=str, default=DEFAULT_MODEL_NAME, help="Model name to evaluate")
    parser.add_argument("-p", "--model_prefix", type=str, default='models', help="Directory containing models")
    parser.add_argument("-c", "--checkpoint_path", type=str, default=None, help="Explicit path to a .pth file (overrides -m and -p)")
    args = parser.parse_args()

    model_name = args.model_name
    model_prefix = args.model_prefix
    checkpoint_path = args.checkpoint_path
    
    script_basedir = os.path.dirname(os.path.abspath(__file__))
    
    # Determine result folder name
    log_name = os.path.basename(checkpoint_path).replace('.pth', '') if checkpoint_path else model_name
    directory = os.path.join(script_basedir, f'../data/real_data_evals/{log_name}')
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    print(args.slurm)
    if args.slurm:
        print("Running on slurm")
        global ex
        global q
        maximum_runtime = 0
        log_folder = os.path.join(script_basedir, '../logs/')
        maximum_runtime = set_queue('mlhiwi', log_folder)
        submit_func = ex.submit
        job = submit_func(main_evaluator, model_name=model_name, model_prefix=model_prefix, checkpoint_path=checkpoint_path)

        print(job)
    else:
        print("Running on local machine")
        for pred_style in ['multipoint']: # Defaulting to multipoint for SC-Mamba
            main_evaluator(pred_style, model_name, model_prefix, checkpoint_path)


    