"""
Module to convert process synthetic series using tensorflow
"""


import numpy as np
import pandas as pd
from datetime import date
from typing import List
from torch import rand
from synthetic_generation.generate_series import generate
from synthetic_generation.constants import *
from synthetic_generation.series_config import *
import tqdm
from joblib import Parallel, delayed


def generate_single_sample(size=CONTEXT_LENGTH, freq: str = None, transition: bool=True, start: pd.Timestamp = None,
                           return_list: bool=False, options: dict = {}):
    """
    Generate a single sample of time series data suitable for PyTorch models.
    """
    cfg, sample = generate(size, freq=freq, transition=transition, start=start, options=options, random_walk=False)
    id_ = str(cfg)

    if return_list:
        return np.stack([
                sample.index.year.values, 
                sample.index.month.values, 
                sample.index.day.values, 
                sample.index.day_of_week.values + 1, 
                sample.index.day_of_year.values,
                sample.index.hour.values,
                sample.index.minute.values], axis=-1), sample.series_values.values * sample.noise.values
    else:
        return {
            "id": id_,
            "ts": pd.to_datetime(sample.index).astype('int64'),
            "y": sample.series_values.values,
            "noise": sample.noise.values
        }

def generate_time_series(N=100, size=CONTEXT_LENGTH, freq: str = None, transition: bool=True,
                         start: pd.Timestamp = None, options: dict = {}, n_jobs: int = -2):
    """
    Generate time series data suitable for PyTorch models.
    """

    time_series = Parallel(n_jobs=n_jobs)(delayed(generate_single_sample)(size=size, freq=freq, transition=transition, start=start, options=options) 
                                     for _ in tqdm.tqdm(range(N))
                                    )
    return time_series

def generate_time_series_seq(N=100, size=CONTEXT_LENGTH, freqs: str | List[str] = None, transition: bool=True,
                         start: pd.Timestamp = None, options: dict = {}):
    """
    Generate time series data suitable for PyTorch models.
    """

    time_series = []
    for i in tqdm.tqdm(range(N)):
        if isinstance(freq, list):
            freq = freqs[i]
        else:
            freq = freqs
        time_series.append(generate_single_sample(size=size, freq=freq, transition=transition, start=start, options=options))

    return time_series
