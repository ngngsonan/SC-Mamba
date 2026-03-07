import os
from re import sub
import sys
from pathlib import Path
from typing import Counter

# Get the directory of the current script
current_dir = Path(__file__).parent

# Get the parent directory (project_root)
parent_dir = current_dir.parent

# Add the synthetic_generation directory to the sys.path
sys.path.append(str(parent_dir))

import gpytorch
import numpy as np
import pandas as pd
import torch
import time
import argparse
import yaml
import torch
import pickle
import tqdm
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from synthetic_generation.constants import *
from joblib import Parallel, delayed
from pandas.tseries.frequencies import to_offset
from scipy.stats import beta
from functools import reduce
from synthetic_generation.utils import random_binary_map, create_kernel, extract_periodicities
import warnings
from synthetic_generation.generate_steps_n_spikes import generate_peak_spikes


warnings.filterwarnings('ignore')

def sample_gp_series(max_kernels, seq_len: int, likelihood_noise_level: float = 0.4, noise_level="random", subday: bool=False, return_list: bool=False, 
                     kernel_bank=None, gaussians_periodic=False, kernel_periods=None, peak_spike_ratio=None, max_period_ratio=0.6, subfreq_ratio=0.0,
                     periods_per_freq=0.0, gaussian_sampling_ratio=1.0,
                     **kernel_params):
    # Set up the model
    kernel_bank = KERNEL_BANK if kernel_bank is None else kernel_bank
    trend = np.random.choice([True, False])
    if trend:
        mean_module = gpytorch.means.LinearMean(input_size=1)
    else:
        mean_module = gpytorch.means.ConstantMean()
            
    yearly_sampling = ["yearly"] if seq_len <= 200 else []
    if subday:
        freq = freq_dict[np.random.choice(['minute','hourly','daily', 'weekly', 'monthly', 'quarterly'] + yearly_sampling)]['freq']
    else:
        freq = freq_dict[np.random.choice(['daily', 'weekly', 'monthly', 'quarterly'] + yearly_sampling)]['freq']
        
    subfreq = ""
    
    if np.random.rand() < subfreq_ratio:
        if freq == "min":
            subfreq = str(np.random.choice([5, 15, 30]))
        elif freq == "H":
            subfreq = str(np.random.choice([3, 6, 12]))
        elif freq == "D":
            subfreq = str(np.random.choice([2, 10]))
        elif freq == "W":
            subfreq = str(np.random.choice([2, 4]))
        elif freq == "MS":
            subfreq = str(np.random.choice([3, 4, 6]))
        # elif freq == "Y":
        #     subfreq = str(np.random.choice([2, 5]))
        
    if seq_len > 512:
        subfreq = ""
        
    exact_freqs = False
    if np.random.rand() < periods_per_freq:
        exact_freqs = True
        subfreq_int = int(subfreq) if subfreq != "" else 1 
        if freq == "min":
            kernel_periods = [5, 15, 30, 60, 120, 240, 360] 
        elif freq == "H":
            kernel_periods = [3, 6, 12, 24, 48, 72, 168]
        elif freq == "D":
            kernel_periods = [7, 14, 28, 30, 90, 180, 365]
        elif freq == "W":
            kernel_periods = [2, 4, 8, 12, 24, 52]
        elif freq == "MS":
            kernel_periods = [3, 4, 6, 12, 24, 36, 60]
        elif freq == "QS":
            kernel_periods = [4, 8, 12, 20, 40, 80]
        elif freq == "Y":
            kernel_periods = [2, 5, 10]
            
        kernel_periods = np.array(kernel_periods)
        kernel_periods = kernel_periods[kernel_periods >= subfreq_int]
        kernel_periods = list(kernel_periods // subfreq_int)
        
    # print(f"subfreq: {subfreq}")
    # print(f"freq: {freq}")
    # print(f"kernel_periods: {kernel_periods}")
    
    # Create the spaced points for the x values of the gp
    train_x = torch.linspace(0, 1, seq_len)
    train_y = None

    class GPModel(gpytorch.models.ExactGP):
        def __init__(self, mean_module, kernel, likelihood, train_x = train_x, train_y = train_y):
            super(GPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = mean_module
            self.covar_module = kernel
            self.train_x = train_x
    
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)  
        
    kernel_weights = np.array([i[1] for i in kernel_bank.values()])
    
    # Sample from the GP prior    
    
    # flag for not failed sampling
    fail_flag = True
    while fail_flag==True:
        try:
            kernel_ids = np.random.choice(list(kernel_bank.keys()), size=np.random.randint(1,max_kernels+1),
                                        replace=True, p=kernel_weights/np.sum(kernel_weights))
            kernel_names = [kernel_bank[i][0] for i in kernel_ids]
            kernel_counter = Counter(kernel_names)
            gaussian_sample = False
            if np.random.rand() < gaussian_sampling_ratio:
                gaussian_sample = True
            kernel = reduce(random_binary_map, [create_kernel(kernel_bank[i][0], seq_len=seq_len, max_period_length=int(max_period_ratio*seq_len), 
                                                              gaussians_periodic=gaussians_periodic, kernel_periods=kernel_periods, kernel_counter=kernel_counter,
                                                              freq=freq, exact_freqs=exact_freqs, gaussian_sample = gaussian_sample, subfreq=subfreq,
                                                              **kernel_params) 
                                                for i in kernel_ids])
            likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_covar=torch.diag(torch.full_like(train_x, likelihood_noise_level**2)))

            model = GPModel(mean_module, kernel, likelihood)
            
            if noise_level == "random":
                noise = np.random.choice([1e-1, 1e-2, 1e-3], p=[0.1, 0.2, 0.7])
            else:
                noise = 1e-1 if noise_level == "high" else 1e-2 if noise_level == "moderate" else 1e-3
                
            model.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_computations(), \
                            gpytorch.settings.cholesky_jitter(noise), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_sample = model(model.train_x).sample(sample_shape=torch.Size([1]))
            fail_flag = False
        except:
            continue
        
    if np.random.rand() < peak_spike_ratio:
        periodicitis = extract_periodicities(kernel, seq_len)
        if len(periodicitis) > 0:    
            p = np.round(max(periodicitis)).astype(int)
            spikes_type = np.random.choice(["regular", "patchy"], p=[0.3, 0.7])
            spikes = generate_peak_spikes(seq_len, p, spikes_type=spikes_type)
            spikes_shift = p - y_sample[:,:p].argmax()
            spikes = np.roll(spikes, -spikes_shift)
            if spikes.max() < 0:
                y_sample = y_sample + spikes + 1
            else:
                y_sample = y_sample * spikes

    start = pd.Timestamp(date.fromordinal(int((BASE_START - BASE_END)*beta.rvs(5,1)+BASE_START)))
    ts =  pd.date_range(start=start, periods=seq_len, freq=to_offset(subfreq+freq))

    if return_list:
        return np.stack([
                ts.year.values, 
                ts.month.values, 
                ts.day.values, 
                ts.day_of_week.values + 1, 
                ts.day_of_year.values,
                ts.hour.values,
                ts.minute.values], axis=-1), y_sample[0].numpy()
    else:
        return {
            'ts': ts.astype('int64'),
            'y': y_sample[0].numpy(),
            'noise': np.ones_like(y_sample[0])
        }

def generate_gp_time_series(num_samples: int, seq_len: int, max_kernels: int=5, likelihood_noise_level: float=0.1, sub_day: bool=False, 
                            kernel_bank = None, gaussians_periodic = False, **kernel_params):
    
    
    samples = Parallel(n_jobs=-2)(delayed(sample_gp_series)(max_kernels, seq_len=seq_len, likelihood_noise_level=likelihood_noise_level, subday=sub_day,
                                                            kernel_bank=kernel_bank, gaussians_periodic=gaussians_periodic, **kernel_params) 
                                     for _ in tqdm.tqdm(range(num_samples))
                                    )

    return samples    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to config file",
                        default="synthetic_generation/gp_prior_config.yaml")
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.load(config_file, yaml.loader.SafeLoader)
    
    start_time = time.time()
    samples = generate_gp_time_series(num_samples=config['num_samples'],
                                      seq_len=config['seq_len'],
                                      max_kernels=config['max_kernels'],
                                      noise_level=config['noise_level'],
                                      sub_day=config['sub_day'])
    print(f'time to generate samples: {time.time() - start_time}')

    print(samples[0]['ts'])
    
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    save_path = f"{script_dir}/{config['data_prefix']}/{config['version']}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(samples, f)


if __name__ == '__main__':
    main()