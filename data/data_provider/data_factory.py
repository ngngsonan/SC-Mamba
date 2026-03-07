from argparse import Namespace
from os import sep
from benchmark.data_provider.data_loader import Dataset_GluonTS
from torch.utils.data import DataLoader

data_dict = {
    "nn5_daily_without_missing": Dataset_GluonTS,
    "nn5_weekly": Dataset_GluonTS,
    "covid_deaths": Dataset_GluonTS,
    "weather": Dataset_GluonTS,
    "hospital": Dataset_GluonTS, 
    "fred_md": Dataset_GluonTS,
    "car_parts_without_missing": Dataset_GluonTS,
    "traffic": Dataset_GluonTS,
    "traffic_nips": Dataset_GluonTS,
    "dominick": Dataset_GluonTS,
    "m3_monthly": Dataset_GluonTS,
    "ercot": Dataset_GluonTS,
    "m1_monthly": Dataset_GluonTS,
    "m1_quarterly": Dataset_GluonTS,
    "cif_2016": Dataset_GluonTS,
    "electricity_hourly": Dataset_GluonTS,
    "m4_daily": Dataset_GluonTS,
    "exchange_rate": Dataset_GluonTS,
    "m3_quarterly": Dataset_GluonTS,
    "m3_yearly": Dataset_GluonTS,
    "tourism_monthly": Dataset_GluonTS,
    "tourism_quarterly": Dataset_GluonTS,
    "temperature_rain_without_missing": Dataset_GluonTS,
    "nasdaq": Dataset_GluonTS, # SC-Mamba Special Addition
}


def data_provider(args, flag, subday=False):
    if isinstance(args, Namespace):
        data_args = vars(args)
    else:
        data_args = args
    Data = data_dict[data_args['data']]

    if flag == 'test':
        shuffle_flag = False
    else:
        shuffle_flag = True
    
    data_set = Data(
        data_path=data_args['data_path'],
        separate_dataset= data_args['separate_dataset'] if "separate_dataset" in data_args.keys() else False,
        flag=flag,
        size=[data_args['seq_len'], data_args['label_len'], data_args['pred_len']],
        features=data_args['features'],
        target=data_args['target'],
        scale=data_args['scale'],
        subday=subday,
    )
    
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=data_args['batch_size'],
        shuffle=shuffle_flag,
        num_workers=data_args['num_workers'],
        drop_last=True)
    return data_set, data_loader


