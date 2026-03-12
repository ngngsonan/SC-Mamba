import torch
import numpy as np
import pandas as pd
import yaml
import os
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.time_feature.seasonality import get_seasonality
from data.data_provider.data_factory import data_provider
from utilsforecast.losses import mase, mae, smape, rmse
from scipy.stats import norm as scipy_norm
from core.models import SCMamba_Forecaster

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
    "electricity_hourly": 24,
    "m4_daily": 14,
    "exchange_rate": 30,
    ## "ett_hourly": 24,
    "m3_quarterly": 8,
    "tourism_monthly": 24,
    "tourism_quarterly": 8,
}

MAX_LENGTH = 512

def scale_data(output, scaler):
    # SCMamba_Forecaster returns 'mu' key; legacy models return 'result' key
    result_key = 'mu' if 'mu' in output else 'result'
    if scaler == 'custom_robust':
        output = (output[result_key] * output['scale'][1].squeeze(-1)) + output['scale'][0].squeeze(-1)
    elif scaler == 'min_max':
        output = (output[result_key] * (output['scale'][0].squeeze(-1) - output['scale'][1].squeeze(-1))) + output['scale'][1].squeeze(-1)
    elif scaler == 'identity':
        output = output[result_key]
    return output

def auto_regressive_predict(model, batch_x, batch_y, batch_x_mark, batch_y_mark, pred_len, real_data_args, scaler, device):

    # decoder input
    dec_inp = torch.zeros_like(
        batch_y[:, -pred_len:]).int()
    dec_inp = torch.cat(
        [batch_y[:, :real_data_args['label_len']], dec_inp], dim=1).int().to(device)
    
    x = {}
    x['ts'] = batch_x_mark
    x['history'] = batch_x.reshape(1,batch_x.size(1))
    outputs = []
    for pred_ind in range(0, pred_len):
            
        x['target_dates'] = batch_y_mark[:, real_data_args['label_len'] + pred_ind].unsqueeze(1)
        x['task'] = dec_inp[:, real_data_args['label_len'] + pred_ind]
        
        output = model(x)
        outputs.append(scale_data(output, scaler))
        
        x['history'] = torch.cat([x['history'], output['result']], dim=1)
        x['ts'] = torch.cat([x['ts'], x['target_dates']], dim=1)

    outputs = torch.stack(outputs, dim=1).detach().cpu().numpy()
        
    return outputs
            
def batch_predict(model, batch_x, batch_y, batch_x_mark, batch_y_mark, pred_len, real_data_args, scaler, device):
    x = {}
    x['ts'] = batch_x_mark.repeat(pred_len, 1, 1).to(device)
    x['history'] = batch_x.reshape(1,batch_x.size(1)).repeat(pred_len, 1).to(device)

    x['target_dates'] = batch_y_mark.transpose(0, 1).to(device)
    x['task'] = torch.zeros(pred_len,1).int().to(device)

    output = model(x)
    output = scale_data(output, scaler)
        
    output = output.detach().cpu().squeeze()
    
    return output
    

def multipoint_predict(model, batch_x, batch_y, batch_x_mark, batch_y_mark, pred_len, scaler, device):
    """
    Run SCMamba_Forecaster in multipoint mode.
    Returns (mu_out, sigma2_out) both inverse-scaled to original data space.
    sigma2 is scaled by span^2 (variance transforms as square of linear scale).
    """
    x = {}
    x['ts'] = batch_x_mark.to(device)
    x['history'] = batch_x.reshape(1, batch_x.size(1)).to(device)
    x['target_dates'] = batch_y_mark.to(device)
    x['task'] = torch.zeros(1, pred_len).int().to(device)
    if not isinstance(model, SCMamba_Forecaster):
        raise ValueError("Model must be an instance of SCMamba_Forecaster")

    output = model(x, pred_len)
    # Inverse-scale mu
    mu_out = scale_data(output, scaler).detach().cpu().squeeze()
    # Inverse-scale sigma2: Var[aX + b] = a^2 * Var[X]
    if scaler == 'min_max':
        span = (output['scale'][0].squeeze(-1) - output['scale'][1].squeeze(-1)).detach().cpu()
        sigma2_out = (output['sigma2'].detach().cpu().squeeze() * span.squeeze() ** 2)
    elif scaler == 'custom_robust':
        iqr = output['scale'][1].squeeze(-1).detach().cpu()
        sigma2_out = (output['sigma2'].detach().cpu().squeeze() * iqr.squeeze() ** 2)
    else:
        sigma2_out = output['sigma2'].detach().cpu().squeeze()
    return mu_out, sigma2_out


def validate_on_real_dataset(dataset: str, model, device, scaler, subday=False):

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'real_data_args.yaml')
    with open(config_path) as file:
        real_data_args = yaml.load(file, yaml.loader.SafeLoader)

    if real_data_args['pad']:
        real_data_args['data_path'] = dataset + f'_pad_{MAX_LENGTH}.pkl'
    else:
        real_data_args['data_path'] = dataset + f'_nopad_{MAX_LENGTH}.pkl'

    pred_len = REAL_DATASETS[dataset]
    real_data_args['data'] = dataset
    real_data_args['pred_len'] = pred_len
    test_dataset, test_dataloader = data_provider(real_data_args, real_data_args['flag'], subday=subday)

    gts_dataset = get_dataset(real_data_args['data'], regenerate=False)
    seasonality = get_seasonality(gts_dataset.metadata.freq)
    if gts_dataset.metadata.freq == 'D':
        seasonality = 7
    print(seasonality)

    batch_train_dfs = []
    batch_pred_dfs = []
    model.eval()
    j = 0
    with torch.no_grad():
        # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        for i, batch in enumerate(test_dataloader):
            ids = batch["id"]
            batch_x = batch["x"]
            batch_y = batch["y"]

            batch_x_mark = batch["ts_x"]
            batch_y_mark = batch["ts_y"]
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            #######################

            # decoder input
            # dec_inp = torch.ones_like(
            #     batch_y[:, -pred_len:]).int()
            # dec_inp = torch.cat(
            #     [batch_y[:, :real_data_args['label_len']], dec_inp], dim=1).int().to(device)
            
            # x = {}
            # x['ts'] = batch_x_mark
            # x['history'] = batch_x.reshape(1,batch_x.size(1))
            # outputs = []
            # for pred_ind in range(0, pred_len):
                    
            #     x['target_dates'] = batch_y_mark[:, real_data_args['label_len'] + pred_ind].unsqueeze(1)
            #     x['task'] = dec_inp[:, real_data_args['label_len'] + pred_ind]
                
            #     output = model(x)
            #     if scaler == 'custom_robust':
            #         outputs.append((output['result'] * output['scale'][1].squeeze(-1)) + output['scale'][0].squeeze(-1))
            #     elif scaler == 'min_max':
            #         outputs.append((output['result'] * (output['scale'][0].squeeze(-1) - output['scale'][1].squeeze(-1))) + output['scale'][1].squeeze(-1))
            #     elif scaler == 'identity':
            #         outputs.append(output['result'])
                
            #     if real_data_args['auto_regressive']:
            #         x['history'] = torch.cat([x['history'], output['result']], dim=1)
            #         x['ts'] = torch.cat([x['ts'], x['target_dates']], dim=1)

            # f_dim = -1 if real_data_args['features'] == 'MS' else 0
            # outputs = torch.stack(outputs, dim=1).detach().cpu().numpy()
            if isinstance(model, SCMamba_Forecaster):
                mu_out, sigma2_out = multipoint_predict(
                    model, batch_x, batch_y, batch_x_mark, batch_y_mark, pred_len, scaler, device
                )
            else:
                # Legacy deterministic path — no uncertainty output
                if real_data_args['auto_regressive']:
                    mu_out = auto_regressive_predict(
                        model, batch_x, batch_y, batch_x_mark, batch_y_mark,
                        pred_len, real_data_args, scaler, device
                    )
                else:
                    mu_out = batch_predict(
                        model, batch_x, batch_y, batch_x_mark, batch_y_mark,
                        pred_len, real_data_args, scaler, device
                    )
                sigma2_out = None

            f_dim = -1 if real_data_args['features'] == 'MS' else 0

            if len(batch_y.shape) == 3:
                batch_y_np = batch_y[:, -pred_len:, f_dim:].detach().cpu().numpy()
            else:
                batch_y_np = batch_y[:, -pred_len:].detach().cpu().numpy()

            pred = mu_out.numpy().flatten()
            true = batch_y_np.squeeze().flatten()

            batch_train_dfs.append(pd.DataFrame({
                'id': ids.repeat_interleave(batch_x.size(1)).numpy(),
                'target': batch_x.flatten().detach().cpu().numpy()
            }))

            pred_row = {
                'id': ids.repeat_interleave(pred_len).numpy(),
                'pred': pred,
                'target': true,
            }
            if sigma2_out is not None:
                sigma2_flat = sigma2_out.numpy().flatten()
                sigma2_flat = np.clip(sigma2_flat, 1e-6, None)
                mu_t = torch.tensor(pred)
                y_t = torch.tensor(true)
                s2_t = torch.tensor(sigma2_flat)
                nll_vals = (0.5 * torch.log(torch.tensor(2 * np.pi) * s2_t)
                            + 0.5 * (y_t - mu_t) ** 2 / s2_t).numpy()
                pred_row['variance'] = sigma2_flat
                pred_row['nll'] = nll_vals

            batch_pred_dfs.append(pd.DataFrame(pred_row))

    train_df = pd.concat(batch_train_dfs)
    pred_df = pd.concat(batch_pred_dfs)

    mase_loss = mase(pred_df, ['pred'], seasonality, train_df, 'id', 'target')
    mae_loss = mae(pred_df, ['pred'], 'id', 'target')
    rmse_loss = rmse(pred_df, ['pred'], 'id', 'target')
    smape_loss = smape(pred_df, ['pred'], 'id', 'target')

    # Probabilistic metrics — unique to SC-Mamba vs deterministic Mamba4Cast baseline
    mean_nll = float(pred_df['nll'].mean()) if 'nll' in pred_df.columns else float('nan')

    # CRPS: closed-form for Gaussian predictive distribution
    # Lower is better; comparable across datasets (scale-independent after normalisation)
    if 'variance' in pred_df.columns:
        sigma_np = np.sqrt(np.clip(pred_df['variance'].values, 1e-6, None))
        mu_np = pred_df['pred'].values
        y_np = pred_df['target'].values
        z = (y_np - mu_np) / sigma_np
        crps_vals = sigma_np * (
            z * (2.0 * scipy_norm.cdf(z) - 1.0)
            + 2.0 * scipy_norm.pdf(z)
            - 1.0 / np.sqrt(np.pi)
        )
        mean_crps = float(crps_vals.mean())
    else:
        mean_crps = float('nan')

    # Drop INFs from MASE (e.g. constant series like some in car_parts)
    mase_vals = mase_loss['pred'].replace([float('inf'), float('-inf')], float('nan'))
    mase_mean = float(mase_vals.mean(skipna=True)) if mase_vals.notna().any() else float('nan')

    return (mase_mean, mae_loss['pred'].mean(),
            rmse_loss['pred'].mean(), smape_loss['pred'].mean(),
            mean_nll, mean_crps)


