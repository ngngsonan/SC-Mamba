"""
Module to generate trend and seasonal components of series
"""
import numpy as np
import pandas as pd
from synthetic_generation.constants import *
from synthetic_generation.series_config import SeriesConfig
from synthetic_generation.utils import shift_axis, weibull_noise, get_random_walk_series
from collections import defaultdict

def make_series_trend(series: SeriesConfig, dates: pd.DatetimeIndex):
    """
    Function to generate the trend(t) component of synthetic series
    :param series: series config for generating trend of synthetic series
    :param dates: dates for which data is present
    :return: trend component of synthetic series
    """
    values = np.full_like(dates, series.scale.base, dtype=np.float32)

    days = (dates - dates[0]).days
    if series.scale.linear is not None:
        values += shift_axis(days, series.offset.linear) * series.scale.linear
    if series.scale.exp is not None:
        values *= np.power(series.scale.exp, shift_axis(days, series.offset.exp))

    return values

def get_freq_component(dates_feature: pd.Index, n_harmonics: int, n_total: int, harmonic_scale: bool = True):
    """
    Method to get systematic movement of values across time
    :param dates_feature: the component of date to be used for generating
    the seasonal movement is different. For example, for monthly patterns
    in a year we will use months of a date, while for day-wise patterns in
    a month, we will use days as the feature
    :param n_harmonics: number of harmonics to include. For example,
    for monthly trend, we use 12/2 = 6 harmonics
    :param n_total: total cycle length
    :return: numpy array of shape dates_feature.shape containing
    sinusoidal value for a given point in time
    """
    harmonics = list(range(1, n_harmonics+1))

    # initialize sin and cosine coefficients with 0
    sin_coef = np.zeros(n_harmonics)
    cos_coef = np.zeros(n_harmonics)

    # choose coefficients inversely proportional to the harmonic
    for idx, harmonic in enumerate(harmonics):
        if not harmonic_scale:
            harmonic = 1
        sin_coef[idx] = np.random.normal(scale = 1 / harmonic)
        cos_coef[idx] = np.random.normal(scale = 1 / harmonic)

    # normalize the coefficients such that their sum of squares is 1
    coef_sq_sum = np.sqrt(np.sum(np.square(sin_coef)) + np.sum(np.square(cos_coef)))
    sin_coef /= coef_sq_sum
    cos_coef /= coef_sq_sum

    # construct the result for systematic movement which
    # comprises of patterns of varying frequency
    return_val = 0
    for idx, harmonic in enumerate(harmonics):
        return_val += sin_coef[idx] * np.sin(2 * np.pi * harmonic * dates_feature / n_total)
        return_val += cos_coef[idx] * np.cos(2 * np.pi * harmonic * dates_feature / n_total)

    return return_val


def make_series_seasonal(series: SeriesConfig, dates: pd.DatetimeIndex, options: dict):
    """
    Function to generate the seasonal(t) component of synthetic series.
    It represents the systematic pattern-based movement over time
    :param series: series config used for generating values
    :dates: dates on which the data needs to be calculated
    """
    seasonal = 1

    harmonic_scale = (np.random.rand() < options["harmonic_scale_ratio"]) if "harmonic_scale_ratio" in options else True
    harmonic_rate = options["harmonic_rate"] if "harmonic_rate" in options else 1.0
    period_factor = options["period_factor"] if "period_factor" in options else 1.0
    
    assert harmonic_rate > 0 and harmonic_rate < 3, "Harmonic rate should be greater than 0 and less than 3"
    
    seasonal_components = defaultdict(lambda: 1)
    if series.scale.minute is not None and series.scale.minute != 0:
        seasonal_components['minute'] = 1 + series.scale.minute * get_freq_component(dates.minute, int(np.ceil(10* harmonic_rate)), 60*period_factor, harmonic_scale=harmonic_scale)
        seasonal *= seasonal_components['minute']
    if series.scale.h is not None and series.scale.h != 0:
        seasonal_components['h'] = 1 + series.scale.h * get_freq_component(dates.hour, int(np.ceil(10* harmonic_rate)), 24*period_factor, harmonic_scale=harmonic_scale)
        seasonal *= seasonal_components['h']
    if series.scale.a is not None and series.scale.a != 0:
        seasonal_components['a'] = 1 + series.scale.a * get_freq_component(dates.month, int(np.ceil(6* harmonic_rate)), 12*period_factor, harmonic_scale=harmonic_scale)
        seasonal *= seasonal_components['a']
    if series.scale.m is not None and series.scale.m != 0:
        seasonal_components['m'] = 1 + series.scale.m * get_freq_component(dates.day, int(np.ceil(10* harmonic_rate)), 30.5*period_factor, harmonic_scale=harmonic_scale)
        seasonal *= seasonal_components['m']
    if series.scale.w is not None and series.scale.w != 0:
        seasonal_components['w'] = 1 + series.scale.w * get_freq_component(dates.dayofweek, int(np.ceil(4* harmonic_rate)), 7*period_factor, harmonic_scale=harmonic_scale)
        seasonal *= seasonal_components['w']

    seasonal_components['seasonal'] = seasonal
    return seasonal_components

def make_series(
    series: SeriesConfig,
    freq: pd.DateOffset,
    periods: int,
    start: pd.Timestamp,
    options: dict,
    random_walk: bool,
):
    """
    make series of the following form
    series(t) = trend(t) * seasonal(t)
    """
    start = freq.rollback(start)
    dates = pd.date_range(start=start, periods=periods, freq=freq)
    scaled_noise_term = 0

    if random_walk:
        values = get_random_walk_series(len(dates))
    if options["seasonal_only"]:
        values_seasonal = make_series_seasonal(series, dates, options)
        values = values_seasonal['seasonal']
    else:
        values_trend = make_series_trend(series, dates)
        values_seasonal = make_series_seasonal(series, dates, options)

        trend_additional = options["trend_additional"] if "trend_additional" in options else False
        
        if trend_additional:
            values = values_trend + values_seasonal['seasonal']
        else:
            values = values_trend * values_seasonal['seasonal']

        weibull_noise_term = weibull_noise(
            k=series.noise_config.k,
            median=series.noise_config.median,
            length=len(values)
        )

        # approximating estimated value from median
        noise_expected_val = series.noise_config.median

        # expected value of this term is 0
        # for no noise, scale is set to 0
        scaled_noise_term = series.noise_config.scale * (weibull_noise_term - noise_expected_val)

    dataframe_data = {
        **values_seasonal,
        'values': values,
        'noise': 1 + scaled_noise_term,
        'dates': dates
    }

    return dataframe_data
