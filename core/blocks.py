from unittest import result
from numpy import block
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')
from training.utils import PositionExpansion, CustomScaling, SimpleRMSNorm
from training.constants import *
from mamba_ssm import Mamba, Mamba2
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SinPositionalEncoding(nn.Module):
    def __init__(self, d_model=36, max_len=5000, sin_pos_const=10000.0):
        super(SinPositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        positions = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(sin_pos_const) / d_model))
        self.encoding[:, 0::2] = torch.sin(positions * div_term)
        self.encoding[:, 1::2] = torch.cos(positions * div_term)
        self.encoding = self.encoding.unsqueeze(0).to(device)

    def forward(self, x):
        return self.encoding[:, :x.size(1)].detach()

class DilatedConv1dBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size=5,
                 max_dilation=3,
                 single_conv=False,
                 conv_gelu=True):
        super(DilatedConv1dBlock, self).__init__()
        self.conv_gelu = conv_gelu
        self.single_conv = single_conv
        if self.single_conv:
            padding = (kernel_size - 1) * 2**max_dilation
            self.conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=2**max_dilation,
                bias=True
            )
        else:
            self.conv = nn.ModuleList()
            conv_out_channels = out_channels // (max_dilation + 1)
            for dilation in range(max_dilation + 1):
                padding = (kernel_size - 1) * 2**dilation
                self.conv.append(nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=conv_out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=2**dilation,
                    bias=True
                ))
        
            self.inception_conv = nn.Conv1d(
                in_channels=out_channels,  # Total number of channels from all convolutions
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )

    def forward(self, x):
        # x is expected to be of shape (batch_size, in_channels, sequence_length)
        x = x.transpose(1, 2)
        seq_len = x.shape[-1]
        if self.single_conv:
            x = self.conv(x)
            if self.conv_gelu:
                x = F.gelu(x)
        else:
            if self.conv_gelu:
                x_list = [F.gelu(conv_layer(x))[:,:,:seq_len] for conv_layer in self.conv]
            else:
                x_list = [conv_layer(x)[:,:,:seq_len] for conv_layer in self.conv]
            x = torch.cat(x_list, dim=1)
            x = self.inception_conv(x)
        return x.transpose(1, 2)
            

class BiMambaEncoderBlock(nn.Module):
    def __init__(self, embed_dim, norm=True, norm_type='layernorm', residual=False, name='SSMEncoderBlock', mamba2=False,
                 enc_conv=False, enc_conv_kernel=5, enc_conv_dilation=0, d_state=128, block_expansion=2,**kwargs):
        super().__init__(**kwargs)
        self.enc_conv = enc_conv
        self.name = name
        self.norm = norm
        self.mamba_layer_forward = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=embed_dim, # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=block_expansion,    # Block expansion factor
        )
        self.mamba_layer_backward = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=embed_dim, # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=block_expansion,    # Block expansion factor
        )
        
        if mamba2:
            self.mamba_layer_forward = Mamba2(
            d_model=embed_dim, # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=block_expansion,    # Block expansion factor
            )
            self.mamba_layer_backward = Mamba2(
            d_model=embed_dim, # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=block_expansion,    # Block expansion factor
        )
            
        if self.enc_conv:
            self.stage_2_layer = DilatedConv1dBlock(embed_dim, embed_dim, enc_conv_kernel, 
                                                         enc_conv_dilation, single_conv=False) #single_conv=True)
        else:
            self.stage_2_layer = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())

        if self.norm:
            if norm_type == 'layernorm':
                self.norm_layer_1 = nn.LayerNorm(embed_dim)
                self.norm_layer_2 = nn.LayerNorm(embed_dim)
                self.norm_layer_3 = nn.LayerNorm(embed_dim)
            elif norm_type == 'rmsnorm':
                self.norm_layer_1 = SimpleRMSNorm(embed_dim)
                self.norm_layer_2 = SimpleRMSNorm(embed_dim)
                self.norm_layer_3 = SimpleRMSNorm(embed_dim)
        self.residual = residual

    def forward(self, x, pred_len=0):

        if self.norm:
            x_ssm = self.mamba_layer_forward(self.norm_layer_1(x))
            x_ssm = x_ssm + self.mamba_layer_backward(self.norm_layer_1(x.flip(dims=[1]))).flip(dims=[1])
        else:
            x_ssm = self.mamba_layer_forward(x)
            x_ssm = x_ssm + self.mamba_layer_backward(x.flip(dims=[1])).flip(dims=[1])

        if self.residual:
            x = x + x_ssm
        else:
            x = x_ssm

        if self.norm:
            x_out = self.stage_2_layer(self.norm_layer_2(x))
        else:
            x_out = self.stage_2_layer(x)
        
        if self.residual:
            x_out = x_out + x

        return x_out
    

class SSMEncoderBlock(nn.Module):
    def __init__(self, embed_dim, norm=True, norm_type='layernorm', residual=False, name='SSMEncoderBlock', mamba2=False,
                 enc_conv=False, enc_conv_kernel=5, enc_conv_dilation=0, d_state=128, block_expansion=2, **kwargs):
        super().__init__(**kwargs)
        
        self.enc_conv = enc_conv
        self.name = name
        self.norm = norm
        self.mamba_layer = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=embed_dim, # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor ## originally 16
            d_conv=4,    # Local convolution width
            expand=block_expansion,    # Block expansion factor
        )

        if mamba2:
            self.mamba_layer = Mamba2(
                d_model=embed_dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor ## originally 32
                d_conv=4,    # Local convolution width
                expand=block_expansion,    # Block expansion factor
            )
            
        if self.enc_conv:
            self.stage_2_layer = DilatedConv1dBlock(embed_dim, embed_dim, enc_conv_kernel, 
                                                         enc_conv_dilation, single_conv=False)
        else:
            self.stage_2_layer = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())

        if self.norm:
            if norm_type == 'layernorm':
                self.norm_layer_1 = nn.LayerNorm(embed_dim)
                self.norm_layer_2 = nn.LayerNorm(embed_dim)
                self.norm_layer_3 = nn.LayerNorm(embed_dim)
            elif norm_type == 'rmsnorm':
                self.norm_layer_1 = SimpleRMSNorm(embed_dim)
                self.norm_layer_2 = SimpleRMSNorm(embed_dim)
                self.norm_layer_3 = SimpleRMSNorm(embed_dim)
        self.residual = residual

    def forward(self, x):
        
        if self.norm:
            x_ssm = self.mamba_layer(self.norm_layer_1(x))
        else:
            x_ssm = self.mamba_layer(x)

        if self.residual:
            x = x + x_ssm
        else:
            x = x_ssm

        if self.norm:
            x_out = self.stage_2_layer(self.norm_layer_2(x))
        else:
            x_out = self.stage_2_layer(x) 
        
        if self.residual:
            x_out = x_out + x

        return x_out
    

class ConcatLayer(nn.Module):
    def __init__(self, dim=1, name=None):
        super().__init__()
        self.dim = dim
        self.name = name

    def forward(self, inputs):
        return torch.cat(inputs, dim=self.dim)