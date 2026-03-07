"""
Module to create train and test dfs
"""

from re import sub
import sys
import os

# Get the directory of the current file
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_directory = os.path.dirname(current_file_directory)

# Append the parent directory to sys.path
sys.path.append(parent_directory)


import torch
import torch.nn.functional as F
import os
import pickle
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, IterableDataset
from synthetic_generation.tf_generate_series import generate_single_sample
from synthetic_generation.generate_gp import sample_gp_series
from synthetic_generation.generate_steps_n_spikes import generate_damping, generate_spikes


MAX_LENGTH = 512

class GenerativeDatasetMultiPoints(IterableDataset):
    def __init__(self, config,  cpus_available, device, initial_epoch=0, mode='train', return_target_series=False):
        self.max_seq_len = config['max_seq_len']
        self.min_seq_len = config['min_seq_len']
        self.batch_size = config['batch_size']
        self.pred_len = config['pred_len']
        self.pred_len_min = config['pred_len_min']
        self.pred_len_sample = config['pred_len_sample'] # for a curriculum learning approach
        # 0 is gp 1 is fp
        self.prior_mix_frac = config["prior_config"]['prior_mix_frac']
        self.curriculum_learning = config["prior_config"]['curriculum_learning']

        self.gp_hypers = config["prior_config"]['gp_prior_config']
        self.fp_prior_options = config["prior_config"]['fp_options']
        self.subday = config['sub_day']
        if mode == 'train':
            self.batches_per_iter = int(np.ceil(config['training_rounds'] // cpus_available))
        else:
            self.batches_per_iter = int(np.ceil(config['validation_rounds'] // cpus_available))

        # when training is continued, we need to adjust the batch counter for the curriculum learning approach
        self.batch_counter = initial_epoch * self.batches_per_iter #defaults to 0 if not continuing training
        self.device = device
        self.return_target_series = return_target_series

        # curriculum learning set up - vary the fraction of gp samples over time
        if self.curriculum_learning:
            num_batches_per_worker = self.batches_per_iter * config['num_epochs']
            self.gp_fractions = np.linspace(0.2, self.prior_mix_frac, num_batches_per_worker)

        if self.gp_hypers["use_original_gp"]:
            self.kernel_bank = None
            self.gp_hypers["gaussians_periodic"] = False
        else:
            self.kernel_bank = {
                            0: ('matern_kernel', self.gp_hypers['kernel_bank']['matern_kernel']),
                            1: ('linear_kernel', self.gp_hypers['kernel_bank']['linear_kernel']),
                            2: ('periodic_kernel', self.gp_hypers['kernel_bank']['periodic_kernel']),
                            3: ('polynomial_kernel', self.gp_hypers['kernel_bank']['polynomial_kernel']),
                            4: ('spectral_mixture_kernel', self.gp_hypers['kernel_bank']['spectral_mixture_kernel'])
                        }
        ## for mixing up different time series
        self.mixup_prob = config["prior_config"]["mixup_prob"]
        self.mixup_max = config["prior_config"]["mixup_series"]
        
        self.damp_and_spike = config["prior_config"]["damp_and_spike"]
        self.damping_noise_ratio = config["prior_config"]["damping_noise_ratio"]
        self.spike_noise_ratio = config["prior_config"]["spike_noise_ratio"]
        
        self.spike_signal_ratio = config["prior_config"]["spike_signal_ratio"]
        self.spike_batch_ratio = config["prior_config"]["spike_batch_ratio"]
                
    def collate_fn(self, batch):
        return self.pin_memory(batch)

    def pin_memory(self, batch):
        if torch.cuda.device_count() < 2:
            return batch
        batch['ts'] = batch['ts'].pin_memory(device=self.device)
        batch['history'] = batch['history'].pin_memory(device=self.device)
        batch['target_dates'] = batch['target_dates'].pin_memory(device=self.device)
        batch['target_values'] = batch['target_values'].pin_memory(device=self.device)
        batch['task'] = batch['task'].pin_memory(device=self.device)
        return batch
        

    def worker_init_fn(self, worker_id):
        # Your custom worker initialization logic
        seed = (torch.initial_seed() | (int(worker_id) + np.random.randint(0, 1000))) % (2**32)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def __iter__(self):
        for _ in range(self.batches_per_iter):
            yield self._generate_data_batch()
    
    def _generate_data_batch(self):
        pred_len = self.pred_len
        if self.pred_len_sample:
            pred_len = np.random.randint(self.pred_len_min, self.pred_len + 1)
        seq_len = np.random.randint(self.min_seq_len, self.max_seq_len + 1) + pred_len

        if self.curriculum_learning:
            prior_fraction = self.gp_fractions[self.batch_counter]
        else:
            prior_fraction = self.prior_mix_frac

        num_gp_samples = int(self.batch_size * prior_fraction)
        num_fp_samples = self.batch_size - num_gp_samples
        
        gp_samples = np.zeros((num_gp_samples, seq_len, 8))
        fp_samples = np.zeros((num_fp_samples, seq_len, 8))

        for i in range(num_gp_samples):
            kernel_periods = None if "kernel_periods" not in self.gp_hypers else self.gp_hypers["kernel_periods"]
            gp_samples[i,:,:7], gp_samples[i,:,7] = sample_gp_series(self.gp_hypers['max_kernels'], seq_len,
                                                                    self.gp_hypers['likelihood_noise_level'], noise_level = self.gp_hypers['noise_level'],
                                                                    subday = self.subday, kernel_bank = self.kernel_bank,
                                                                    kernel_periods= kernel_periods, peak_spike_ratio = self.gp_hypers["peak_spike_ratio"],
                                                                    gaussians_periodic=self.gp_hypers["gaussians_periodic"], max_period_ratio=self.gp_hypers["max_period_ratio"], 
                                                                    subfreq_ratio=self.gp_hypers["subfreq_ratio"], periods_per_freq=self.gp_hypers["periods_per_freq"],
                                                                    gaussian_sampling_ratio=self.gp_hypers["gaussian_sampling_ratio"], return_list=True)

        for i in range(num_fp_samples):
            transition = np.random.choice([True, False], p=[self.fp_prior_options["transition_ratio"], 1-self.fp_prior_options["transition_ratio"]])
            yearly_sampling = ["yearly"] if seq_len <= 200 else []
            if self.subday:
                freq = np.random.choice(['minute','hourly','daily', 'weekly', 'monthly'] + yearly_sampling)
            else:
                freq = np.random.choice(['daily', 'weekly', 'monthly'] + yearly_sampling)
            fp_samples[i,:,:7], fp_samples[i,:,7] = generate_single_sample(size=seq_len, freq=freq, transition=transition,
                                                                           return_list=True, options=self.fp_prior_options)
        
        combined_samples = np.concatenate([gp_samples, fp_samples], axis=0)[np.random.permutation(self.batch_size),:,:]
        
        if np.random.rand() < self.mixup_prob:
            mixup_series = np.random.randint(2, self.mixup_max+1)
            mixup_indices = np.random.choice(self.batch_size, mixup_series, replace=False)
            original_vals = combined_samples[mixup_indices,:,-1].copy()
            for i in mixup_indices:
                mixup_weights = np.random.rand(mixup_series)
                mixup_weights /= np.sum(mixup_weights)
                combined_samples[i,:,-1] = np.sum(original_vals * mixup_weights[:, np.newaxis], axis=0)
            del original_vals

        if self.damp_and_spike:
            # choose random time series samples to add damping and spikes
            damping_ratio, spike_ratio = np.random.uniform(0, self.damping_noise_ratio), np.random.uniform(0, self.spike_noise_ratio)
            damping_indices = np.random.choice(self.batch_size, int(np.ceil(self.batch_size * damping_ratio)), replace=False)
            combined_samples[damping_indices, :,-1] = np.stack([x * generate_damping(x.shape[0]).numpy() for x in combined_samples[damping_indices, :,-1]], axis=0)
            spike_indices = np.random.choice(self.batch_size, int(np.ceil(self.batch_size * spike_ratio)), replace=False)
            spiked_samples = []
            for x in combined_samples[spike_indices, :,-1]:
                spike = generate_spikes(x.shape[0]).numpy()
                spiked_samples.append(x * spike if spike.max() < 0 else x + spike + 1)
            combined_samples[spike_indices, :,-1] = np.stack(spiked_samples, axis=0)
            
            if np.random.rand() < self.spike_signal_ratio:
                spikey_series_ratio = np.random.uniform(0, self.spike_batch_ratio)
                spike_replace_indices = np.random.choice(self.batch_size, int(np.ceil(self.batch_size * spikey_series_ratio)), replace=False)
                combined_samples[spike_replace_indices, :,-1] = np.stack([generate_spikes(x.shape[0]).numpy() for x in combined_samples[spike_replace_indices, :,-1]], axis=0)
            
        mean_task_idx = int(np.random.uniform(0.5, 0.9) * self.batch_size)
        task = np.zeros((self.batch_size, pred_len))
        task[mean_task_idx:] = np.ones((self.batch_size-mean_task_idx, pred_len))

        history_ts_y = combined_samples[:,:(seq_len-pred_len),:]
        target_ts = combined_samples[:,(seq_len-pred_len):,:]

        complete_target = target_ts

        target_ts_red = target_ts[:,:,:7][np.arange(self.batch_size), :, :]

        target_y_red = np.zeros((self.batch_size, pred_len))
        # processing for task=0
        target_y_red[:mean_task_idx] = target_ts[:mean_task_idx,:,7][np.arange(mean_task_idx), :]
        # processing for task=1
        target_y_red[mean_task_idx:] = np.cumsum(target_ts[mean_task_idx:,:,7], axis=-1)[np.arange(self.batch_size-mean_task_idx),:] / (np.arange(pred_len)[np.newaxis, :] + 1)

        batch = {'ts': torch.tensor(history_ts_y[:,:,:7]), 'history': torch.tensor(history_ts_y[:,:,7]),
                 'target_dates': torch.tensor(target_ts_red), 'target_values': torch.tensor(target_y_red), 'task': torch.tensor(task).int(),
                 "complete_target": torch.tensor(complete_target)}
        # useful for viz and evals later
        if self.return_target_series:
            batch['target_series'] = torch.tensor(target_ts)
        
        self.batch_counter += 1
        return batch

   

class GenerativeDataset(IterableDataset):
    def __init__(self, config, cpus_available, device, initial_epoch=0, mode='train', return_target_series=False):
        self.max_seq_len = config['max_seq_len']
        self.min_seq_len = config['min_seq_len']
        self.batch_size = config['batch_size']
        self.pred_len = config['pred_len']
        self.pred_len_min = config['pred_len_min']
        self.pred_len_sample = config['pred_len_sample']
        self.batch_counter = 0 # for a curriculum learning approach
        # 0 is gp 1 is fp
        self.prior_mix_frac = config["prior_config"]['prior_mix_frac']
        self.curriculum_learning = config["prior_config"]['curriculum_learning']

        self.gp_hypers = config["prior_config"]['gp_prior_config']
        self.fp_prior_options = config["prior_config"]['fp_options']
        self.subday = config['sub_day']
        if mode == 'train':
            self.batches_per_iter = int(np.ceil(config['training_rounds'] // cpus_available))
        else:
            self.batches_per_iter = int(np.ceil(config['validation_rounds'] // cpus_available))
        
        # when training is continued, we need to adjust the batch counter for the curriculum learning approach
        if config['continue_training']:
            self.batch_counter = initial_epoch * self.batches_per_iter
        else:
            self.batch_counter = 0
        self.device = device
        self.return_target_series = return_target_series
        
        # curriculum learning set up - vary the fraction of gp samples over time
        if self.curriculum_learning:
            num_batches_per_worker = self.batches_per_iter * config['num_epochs']
            self.gp_fractions = np.linspace(0.2, self.prior_mix_frac, num_batches_per_worker)

        if self.gp_hypers["use_original_gp"]:
            self.kernel_bank = None
            self.gp_hypers["gaussians_periodic"] = False
        else:
            self.kernel_bank = {
                            0: ('matern_kernel', self.gp_hypers['kernel_bank']['matern_kernel']),
                            1: ('linear_kernel', self.gp_hypers['kernel_bank']['linear_kernel']),
                            2: ('periodic_kernel', self.gp_hypers['kernel_bank']['periodic_kernel']),
                            3: ('polynomial_kernel', self.gp_hypers['kernel_bank']['polynomial_kernel']),
                            4: ('spectral_mixture_kernel', self.gp_hypers['kernel_bank']['spectral_mixture_kernel'])
                        }

                ## for mixing up different time series
        self.mixup_prob = config["prior_config"]["mixup_prob"]
        self.mixup_max = config["prior_config"]["mixup_series"]
        
        self.damping_noise_ratio = config["prior_config"]["damping_noise_ratio"]
        self.spike_noise_ratio = config["prior_config"]["spike_noise_ratio"]
        
        self.spike_signal_ratio = config["prior_config"]["spike_signal_ratio"]
        self.spike_batch_ratio = config["prior_config"]["spike_batch_ratio"]

    def collate_fn(self, batch):
        return self.pin_memory(batch)

    def pin_memory(self, batch):
        if torch.cuda.device_count() < 2:
            return batch
        batch['ts'] = batch['ts'].pin_memory(device=self.device)
        batch['history'] = batch['history'].pin_memory(device=self.device)
        batch['target_dates'] = batch['target_dates'].pin_memory(device=self.device)
        batch['target_values'] = batch['target_values'].pin_memory(device=self.device)
        batch['task'] = batch['task'].pin_memory(device=self.device)
        return batch
        

    def worker_init_fn(self, worker_id):
        # Your custom worker initialization logic
        seed = (torch.initial_seed() | (int(worker_id) + np.random.randint(0, 1000))) % (2**32)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def __iter__(self):
        for _ in range(self.batches_per_iter):
            yield self._generate_data_batch()
    
    def _generate_data_batch(self):
        pred_len = self.pred_len
        if self.pred_len_sample:
            pred_len = np.random.randint(self.pred_len_min, self.pred_len + 1)
        seq_len = np.random.randint(self.min_seq_len, self.max_seq_len + 1) + pred_len

        if self.curriculum_learning:
            prior_fraction = self.gp_fractions[self.batch_counter]
        else:
            prior_fraction = self.prior_mix_frac

        num_gp_samples = int(self.batch_size * prior_fraction)
        num_fp_samples = self.batch_size - num_gp_samples
        
        gp_samples = np.zeros((num_gp_samples, seq_len, 8))
        fp_samples = np.zeros((num_fp_samples, seq_len, 8))
        
        for i in range(num_gp_samples):
            kernel_periods = None if "kernel_periods" not in self.gp_hypers else self.gp_hypers["kernel_periods"]
            gp_samples[i,:,:7], gp_samples[i,:,7] = sample_gp_series(self.gp_hypers['max_kernels'], seq_len,
                                                                    self.gp_hypers['likelihood_noise_level'], noise_level = self.gp_hypers['noise_level'],
                                                                    subday = self.subday, kernel_bank = self.kernel_bank,
                                                                    kernel_periods= kernel_periods, peak_spike_ratio = self.gp_hypers["peak_spike_ratio"],
                                                                    gaussians_periodic=self.gp_hypers["gaussians_periodic"], return_list=True)
        
        for i in range(num_fp_samples):
            transition = np.random.choice([True, False], p=[self.fp_prior_options["transition_ratio"], 1-self.fp_prior_options["transition_ratio"]])
            yearly_sampling = ["yearly"] if seq_len <= 200 else []
            if self.subday:
                freq = np.random.choice(['minute','hourly','daily', 'weekly', 'monthly'] + yearly_sampling)
            else:
                freq = np.random.choice(['daily', 'weekly', 'monthly'] + yearly_sampling)
            fp_samples[i,:,:7], fp_samples[i,:,7] = generate_single_sample(size=seq_len, freq=freq, transition=transition,
                                                                           return_list=True, options=self.fp_prior_options)
        
        combined_samples = np.concatenate([gp_samples, fp_samples], axis=0)[np.random.permutation(self.batch_size),:,:]

        """to test here"""
        
        if np.random.rand() < self.mixup_prob:
            mixup_series = np.random.randint(2, self.mixup_max)
            mixup_indices = np.random.choice(self.batch_size, mixup_series, replace=False)
            for i in mixup_indices:
                mixup_weights = np.random.rand(mixup_series)
                mixup_weights /= np.sum(mixup_weights)
                combined_samples[i] = np.sum(combined_samples[i] * mixup_weights[:, np.newaxis, np.newaxis], axis=0)


        # choose random time series samples to add damping and spikes
        damping_ratio, spike_ratio = np.random.uniform(0, self.damping_noise_ratio), np.random.uniform(0, self.spike_noise_ratio)
        damping_indices = np.random.choice(self.batch_size, int(self.batch_size * damping_ratio), replace=False)
        combined_samples[damping_indices, :,-1] = torch.stack([x.unsqueeze(0) * generate_damping(x.unsqueeze(0)) for x in combined_samples[damping_indices, :,-1]], dim=0)
        # apply damping to each one in the following manner lambda x: x * damping(x)
        spike_indices = np.random.choice(self.batch_size, int(self.batch_size * spike_ratio), replace=False)
        combined_samples[spike_indices, :,-1] = torch.stack([x.unsqueeze(0) * generate_spikes(x.shape[0]) for x in combined_samples[spike_indices, :,-1]], dim=0)
        
        if np.random.rand() < self.spike_signal_ratio:
            spike_replace_indices = np.random.choice(self.batch_size, int(self.batch_size * self.spike_batch_ratio), replace=False)
            combined_samples[spike_replace_indices, :,-1] = torch.stack([generate_spikes(x.shape[0]).unsqueeze(0) for x in combined_samples[spike_replace_indices, :,-1]], dim=0)
        
        """end test"""
            
        mean_task_idx = int(np.random.uniform(0.5, 0.9) * self.batch_size)
        task = np.zeros((self.batch_size, 1))
        task[mean_task_idx:] = 1

        history_ts_y = combined_samples[:,:(seq_len-pred_len),:]
        target_ts = combined_samples[:,(seq_len-pred_len):,:]

        complete_target = target_ts

        target_idx = np.random.randint(0, pred_len, self.batch_size)
        target_ts_red = target_ts[:,:,:7][np.arange(self.batch_size), target_idx, :]

        target_y_red = np.zeros((self.batch_size, 1))
        # processing for task=0
        target_y_red[:mean_task_idx] = target_ts[:mean_task_idx,:,7][np.arange(mean_task_idx), target_idx[:mean_task_idx]][:, np.newaxis]
        # processing for task=1
        target_y_red[mean_task_idx:] = np.cumsum(target_ts[mean_task_idx:,:,7], axis=-1)[np.arange(self.batch_size-mean_task_idx),
                                                                                         target_idx[mean_task_idx:]][:, np.newaxis] / (target_idx[mean_task_idx:][:, np.newaxis] + 1)

        batch = {'ts': torch.tensor(history_ts_y[:,:,:7]), 'history': torch.tensor(history_ts_y[:,:,7]),
                 'target_dates': torch.tensor(target_ts_red).unsqueeze(1), 'target_values': torch.tensor(target_y_red), 'task': torch.tensor(task).int(),
                 "complete_target": torch.tensor(complete_target)}
        # useful for viz and evals later
        if self.return_target_series:
            batch['target_series'] = torch.tensor(target_ts)
        
        self.batch_counter += 1
        
        return batch

def create_train_test_batch_dl(config, device, cpus_available, multipoint=False, initial_epoch=0):
    print(f"multipoint data generation: {multipoint}")
    train_dataset = GenerativeDatasetMultiPoints(config, cpus_available=cpus_available, device=device, initial_epoch=0, mode='train') \
        if multipoint else GenerativeDataset(config, cpus_available=cpus_available, device=device, initial_epoch=0, mode='train')
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=None,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
        worker_init_fn=train_dataset.worker_init_fn,
        num_workers=cpus_available,
        prefetch_factor=15 if cpus_available > 0 else None,
        persistent_workers=cpus_available > 0,
    )
    val_dataset = GenerativeDatasetMultiPoints(config, cpus_available=cpus_available, device=device, initial_epoch=0, mode='val')\
        if multipoint else GenerativeDataset(config, cpus_available=cpus_available, device=device, initial_epoch=0, mode='val')
    val_data_loader = DataLoader(
        dataset=val_dataset,
        batch_size=None,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        worker_init_fn=val_dataset.worker_init_fn,
        num_workers=cpus_available,
        prefetch_factor=15 if cpus_available > 0 else None,
        persistent_workers=cpus_available > 0,
    )
    return train_data_loader, val_data_loader