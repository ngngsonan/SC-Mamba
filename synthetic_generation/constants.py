# Remove schema for the tf records decoding

"""
Module containing constants for synthetic data generation
"""

from datetime import date
# import tensorflow as tf

BASE_START = date.fromisoformat("1885-01-01").toordinal()
BASE_END = date.fromisoformat("2023-12-31").toordinal() + 1

CONTEXT_LENGTH = 630

KERNEL_BANK = {
    0: ('matern_kernel', 3),
    1: ('linear_kernel', 2),
    2: ('rbf_kernel', 2),
    3: ('periodic_kernel', 5),
    4: ('polynomial_kernel', 1),
    5: ('rational_quadratic_kernel', 1),
    6: ('spectral_mixture_kernel', 2)
}

freq_dict = {
    'minute': {'freq': "min", 'time_scale': 1/1440},
    'hourly': {'freq': "H", 'time_scale': 1/24},
    'daily': {'freq': "D", 'time_scale': 1},
    'weekly': {'freq': "W", 'time_scale': 7},
    'monthly': {'freq': "MS", 'time_scale': 30},
    'quarterly': {'freq': "QS", 'time_scale': 90},
    'yearly': {'freq': "Y", 'time_scale': 12}
}


