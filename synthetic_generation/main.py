"""
Module to generate synthetic dataset for pre training
a time series forecasting model
"""
import os
import sys
from pathlib import Path
# Get the directory of the current script
current_dir = Path(__file__).parent

# Get the parent directory (project_root)
parent_dir = current_dir.parent

# Add the synthetic_generation directory to the sys.path
sys.path.append(str(parent_dir))

import yaml
import argparse
import pandas as pd
import pickle
from synthetic_generation.tf_generate_series import generate_time_series
from synthetic_generation.constants import freq_dict



def generate_and_save_datasets(prefix: str, version: str, subday: bool, options: dict, 
                               num_series: int = 10_000, size: int = 200, transition: bool = True):
    """
    Generate dataset and save as separate pickle files for each frequency
    """
    if subday:
        freqs = ['minute', 'hourly', 'daily', 'weekly', 'monthly', 'yearly']
    else:
        freqs = ['daily', 'weekly', 'monthly']

    for freq in freqs:
        print(f"Frequency: {freq}")
        dataset = generate_time_series(N=num_series, size=size, freq=freq, transition=transition, options=options)
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)

        save_path = f"{script_dir}/{prefix}/{version}_{freq}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(dataset, f)

        print(f"Dataset for {freq} saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to config file", default="synthetic_generation/config.example.yaml")
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.load(config_file, yaml.loader.SafeLoader)

    generate_and_save_datasets(config["prefix"], config["version"], config['subday'], config["options"],
                               config["num_series"], config["size"], config['transition'])

if __name__ == "__main__":
    main()
