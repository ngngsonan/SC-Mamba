"""
Module containing different scaler functions
"""
import torch
import numpy as np

def identity_scaler(inputs, epsilon):
    return torch.ones_like(inputs[:,0:1]), inputs


def custom_scaler_robust(inputs, epsilon, q_low=0.25, q_high=0.75, clamp=None):
    # Define a helper function to compute median and IQR excluding NaNs
    def compute_median_iqr(tensor):
        # Median
        median = tensor.nanmedian(dim=1).values
        
        # IQR (Interquartile Range)
        q1 = torch.nanquantile(tensor, q_low, dim=1)
        q3 = torch.nanquantile(tensor, q_high, dim=1)
        iqr = q3 - q1
        
        return median, iqr


    medians, iqrs = compute_median_iqr(inputs)

    # Ensure IQR is not zero to avoid division by zero
    iqrs = iqrs + epsilon

    # Reshape medians and iqrs to match the input shape for broadcasting
    medians = medians.view(-1, 1, 1)
    iqrs = iqrs.view(-1, 1, 1)
    # print(medians.shape, iqrs.shape, inputs_with_nans.shape)
    # Perform robust scaling
    inputs_scaled = (inputs - medians) / iqrs

    # Replace NaNs back to 0s if needed
    inputs_scaled = torch.where(torch.isnan(inputs_scaled), torch.tensor(0.0), inputs_scaled)
    if clamp is not None:
        inputs_scaled = torch.clamp(inputs_scaled, -clamp, clamp)
    return (medians, iqrs), inputs_scaled.to(torch.float32)


def min_max_scaler(inputs, epsilon):
    scale = [torch.max(inputs, dim=1, keepdim=True)[0] + epsilon, torch.min(inputs, dim=1, keepdim=True)[0] - epsilon]
    scale = torch.stack(scale, dim=0)
    output = (inputs - scale[1]) / (scale[0] - scale[1])
    return scale, output.to(torch.float32)
