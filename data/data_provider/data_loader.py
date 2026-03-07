from re import sub
import sys
import os

# Get the directory of the current file
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_directory = os.path.dirname(current_file_directory)
grandparent_directory = os.path.dirname(parent_directory)

# Append the parent directory to sys.path
sys.path.append(parent_directory)
sys.path.append(grandparent_directory)

import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
import pickle

warnings.filterwarnings('ignore')


def compute_time_features_real(ts, subday=False):
    ts = pd.to_datetime(ts)
    if subday:
        features = np.stack([
            ts.year.values, 
            ts.month.values, 
            ts.day.values, 
            ts.day_of_week.values + 1, 
            ts.day_of_year.values,
            ts.hour.values,
            ts.minute.values
        ], axis=-1)
    else:
        features = np.stack([
                ts.year.values, 
                ts.month.values, 
                ts.day.values, 
                ts.day_of_week.values + 1, 
                ts.day_of_year.values
            ], axis=-1)
    return features

class Dataset_GluonTS(Dataset):
    def __init__(self, size, flag='train', separate_dataset=True, ## changed here
                 features='S', data_path='ETTh1.csv',
                 target='target', scale=False, 
                 scaler=StandardScaler(), train_budget=None, subday=False):
        
        self.seq_len = None if (size[0] == 0) else size[0] 
        self.label_len = size[1]
        self.pred_len = size[2]
        
        self.subday = subday
        
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.separate_dataset = separate_dataset ## changed here
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.scaler = scaler
        self.train_budget = train_budget

        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)

        save_path = f"{script_dir}/../../../data/real_val_datasets"
        file_path = os.path.join(save_path, self.data_path)
        print(f"reading data from {file_path}")
        with open(file_path, 'rb') as file:
            df_raw = pickle.load(file)

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        n_series = df_raw.index.get_level_values('Series').nunique()

 
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[2:]
            self.df_data = df_raw[cols_data]
        elif self.features == 'S':
            self.df_data = df_raw[[self.target]]

        border1 = 0
        border2 = n_series
        self.df_stamp = df_raw[['date']].loc[border1:border2]
        self.df_stamp['date'] = pd.to_datetime(self.df_stamp.date)
        if self.subday:
            time_feat_cols = ['year', 'month', 'day', 'day_of_week', 'day_of_year', 'hour', 'minute']
        else:
            time_feat_cols = ['year', 'month', 'day', 'day_of_week', 'day_of_year']
        self.df_stamp[time_feat_cols] = compute_time_features_real(self.df_stamp['date'].values, subday=self.subday)
        self.df_stamp.drop('date', axis=1, inplace=True)
        
        self.series_ids = df_raw.index.get_level_values('Series').unique().values

    def __getitem__(self, index):
        # each row is a different time series

        ts = torch.from_numpy(self.df_stamp.loc[index].values)
        y = torch.from_numpy(self.df_data.loc[index].values)
        s_begin = 0
        s_end = s_begin + (len(y) - self.pred_len)
        t_end = s_end + self.pred_len
        series_id = self.series_ids[index]

        return{"x": y[s_begin:s_end], "y": y[s_end:t_end], "ts_x": ts[s_begin:s_end], "ts_y": ts[s_end:t_end], "id": series_id}#, seq_x_original, seq_y_original

    def __len__(self):
        return self.df_data.index.get_level_values(0).nunique() # - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)