import torch
import numpy as np
import pandas as pd
import yaml
import os
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.time_feature.seasonality import get_seasonality
from data.data_provider.data_factory import data_provider
from utilsforecast.losses import mase, mae, smape, rmse

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
    if scaler == 'custom_robust':
        output = (output['result'] * output['scale'][1].squeeze(-1)) + output['scale'][0].squeeze(-1)
    elif scaler == 'min_max':
        output = (output['result'] * (output['scale'][0].squeeze(-1) - output['scale'][1].squeeze(-1))) + output['scale'][1].squeeze(-1)
    elif scaler == 'identity':
        output = output['result']
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
    x = {}
    x['ts'] = batch_x_mark.to(device)
    x['history'] = batch_x.reshape(1,batch_x.size(1)).to(device)
    x['target_dates'] = batch_y_mark.to(device)
    x['task'] = torch.zeros(1,pred_len).int().to(device)
    if not isinstance(model, SCMamba_Forecaster):
        raise ValueError("Model must be an instance of SCMamba_Forecaster")
    output = model(x, pred_len)
    output = scale_data(output, scaler)
    output = output.detach().cpu().squeeze()
    return output


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
                outputs = multipoint_predict(model, batch_x, batch_y, batch_x_mark, batch_y_mark, pred_len, scaler, device)
            else:
                if real_data_args['auto_regressive']:
                    outputs  = auto_regressive_predict(model, batch_x, batch_y, batch_x_mark, batch_y_mark, pred_len, real_data_args, scaler, device)
                else:
                    outputs = batch_predict(model, batch_x, batch_y, batch_x_mark, batch_y_mark, pred_len, real_data_args, scaler, device)
            
            f_dim = -1 if real_data_args['features'] == 'MS' else 0

            if len(batch_y.shape) == 3:
                batch_y = batch_y[:, -pred_len:, f_dim:].detach().cpu().numpy()
            else:
                batch_y = batch_y[:, -pred_len:].detach().cpu().numpy()

            pred = outputs.squeeze()  # outputs.detach().cpu().numpy()  # .squeeze()
            true = batch_y.squeeze()  # batch_y.detach().cpu().numpy()  # .squeeze()

            # create dfs used to calculate the MASE such that the batch['x'] and pred are squeezed into 1 column and the ids are repeated 
            # for batch['x'].shape[1] times and pred_len times respectively
            batch_train_dfs.append(pd.DataFrame({
                'id': ids.repeat_interleave(batch_x.size(1)).numpy(),
                'target': batch_x.flatten().detach().cpu().numpy()
            }))
            
            batch_pred_dfs.append(pd.DataFrame({
                'id': ids.repeat_interleave(pred_len).numpy(),
                'pred': pred.flatten(),
                'target': true.flatten()
            }))

    train_df = pd.concat(batch_train_dfs)
    pred_df = pd.concat(batch_pred_dfs)

    mase_loss = mase(pred_df, ['pred'], seasonality, train_df, 'id', 'target')
    mae_loss = mae(pred_df, ['pred'], 'id', 'target')
    rmse_loss = rmse(pred_df, ['pred'], 'id', 'target')
    smape_loss = smape(pred_df, ['pred'], 'id', 'target')

    return mase_loss['pred'].mean(), mae_loss['pred'].mean(), rmse_loss['pred'].mean(), smape_loss['pred'].mean()


