"""
Utility functions for training script
"""
import torch.nn.functional as F
from torch import mode, nn
import torch
import torchmetrics
import sys
sys.path.append('../')
from torch.optim.lr_scheduler import LRScheduler
from training.scalers import custom_scaler_robust, identity_scaler, min_max_scaler
import math
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def position_encoding(periods: int, freqs: int):
    return np.hstack([
        np.fromfunction(lambda i, j: np.sin(np.pi / periods * (2**j) * (i-1)), (periods + 1, freqs)),
        np.fromfunction(lambda i, j: np.cos(np.pi / periods * (2**j) * (i-1)), (periods + 1, freqs))
    ])

def generate_model_save_name(config):
    name = 'latest_'
    if config['model_type'] == 'ssm' and config['ssm_config']['bidirectional']:
        name += 'bi_'
    name += f'{config["version"][:2]}_'
    if config['model_type'] == 'ssm':
        model_dict = config['ssm_config']
    elif config['model_type'] == 'transformer':
        model_dict = config['transformer_config']
    else:
        raise ValueError('Model type not supported')
    name += f'{model_dict["num_encoder_layers"]}l_'
    name += f'{model_dict["token_embed_len"]}e_'
    name += 'res_' if model_dict["residual"] else 'nores_'
    
    name += f'{config["min_seq_len"]}-{config["max_seq_len"]}cl_'
    
    if config['model_type'] == 'ssm':
        if model_dict['mamba2']:
            name += 'm2_'
        name += f'gr{model_dict["linear_seq"]}_' if model_dict["global_residual"] else ""
        name += f'norm_' if model_dict["norm"] else ""
        if model_dict['enc_conv'] or model_dict['init_dil_conv']:
            name += f'dconv_'
            name += f'i{model_dict["init_conv_kernel"]}' if model_dict['init_dil_conv'] else ''
            name += f'e{model_dict["enc_conv_kernel"]}' if model_dict['enc_conv'] else ''
            name += '_'
        
    name += f'lr{config["learning_rate"]}_'
    name += f"mp{config['sample_multi_pred']}_" if config["multipoint"] else ""
    name += f"{config['loss']}_" if config['loss'] == 'mae' else ""
    if config["lr_scheduler"].startswith("cosine"):
        name += "initlr" + str(config["initial_lr"]) + "_"
        name += f't{config["t_max"]}' if config["t_max"] != -1 else f't{config["epochs"]}'
        name += f'r' if config["lr_scheduler"] == "cosine_warm_restarts" else ''

    if config['no_pos_enc']:
        name += f'_no-pos'
    elif config['sin_pos_enc']:
        name += f'_sin-pos' if config['sin_pos_const'] == 10000 else f'_sin-pos{config["sin_pos_const"]}'

    if config["prior_config"]['curriculum_learning']:
        name += f'_cu{config["prior_config"]["prior_mix_frac"]}'
    else:
        name += f'_pm{config["prior_config"]["prior_mix_frac"]}'

    if not config["prior_config"]["gp_prior_config"]['use_original_gp']:
        if config["prior_config"]["gp_prior_config"]["gaussians_periodic"]:
            name += "_nPer"
    
    name += f'_pl{config["pred_len_min"]}-{config["pred_len"]}' if config["pred_len_sample"] else f'_pl{config["pred_len"]}'
    
    name += '_subday' if config['sub_day'] else ''
    
    name += f'_subfreq{config["prior_config"]["gp_prior_config"]["subfreq_ratio"]}' if config["prior_config"]["gp_prior_config"]["subfreq_ratio"] != 0.0 else ''
    name += f'_perfreq{config["prior_config"]["gp_prior_config"]["periods_per_freq"]}' if config["prior_config"]["gp_prior_config"]["periods_per_freq"] != 0.0 else ''
        
    name += f'_d{config["prior_config"]["damping_noise_ratio"]}' if config["prior_config"]["damp_and_spike"] else ''
    name += f'_s{config["prior_config"]["spike_noise_ratio"]}' if config["prior_config"]["damp_and_spike"] else ''
    
    return name


def avoid_constant_inputs(inputs, outputs):
    idx_const_in = torch.nonzero(torch.all(inputs == inputs[:,0].unsqueeze(1), dim=1)).squeeze(1)
    if idx_const_in.size(0) > 0:
        inputs[idx_const_in, 0] += np.random.uniform(0.1, 1)


class CustomScaling(nn.Module):
    def __init__(self, name):
        super().__init__()
        if name == 'custom_robust':
            self.scaler = custom_scaler_robust
        elif name == 'min_max':
            self.scaler = min_max_scaler
        else:
            self.scaler = identity_scaler

    def forward(self, history_channels, epsilon):
        return self.scaler(history_channels, epsilon)
    

class PositionExpansion(nn.Module):
    def __init__(self, periods: int, freqs: int):
        super().__init__()
        # Channels could be ceiling(log_2(periods))
        self.periods = periods
        self.channels = freqs * 2
        self.embedding = torch.tensor(position_encoding(periods, freqs), device=device)

    def forward(self, tc: torch.Tensor):
        flat = tc.view(1, -1)
        embedded = self.embedding.index_select(0, flat.flatten().to(torch.long))
        out_shape = tc.shape
        return embedded.view(out_shape[0], out_shape[1], self.channels)
    

class SimpleRMSNorm(nn.Module):
    """
    SimpleRMSNorm

    Args:
        dim (int): dimension of the embedding

    Usage:
    We can use SimpleRMSNorm as a layer in a neural network as follows:
        >>> x = torch.randn(1, 10, 512)
        >>> simple_rms_norm = SimpleRMSNorm(dim=512)
        >>> simple_rms_norm(x).shape
        torch.Size([1, 10, 512])

    """

    def __init__(self, dim):
        super().__init__()
        self.scale = dim**-0.5

    def forward(self, x):
        """Forward method of SimpleRMSNorm"""
        return F.normalize(x, dim=-1) * self.scale

class SMAPEMetric(torchmetrics.Metric):
    def __init__(self, eps=1e-7):
        super().__init__(dist_sync_on_step=False)
        self.eps = eps
        self.add_state("total_smape", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and true labels.
        Args:
            preds (torch.Tensor): The predictions.
            target (torch.Tensor): Ground truth values.
        """
        preds = preds.float()
        target = target.float()
        diff = torch.abs((target - preds) / torch.clamp(target + preds, min=self.eps))
        smape = 200.0 * torch.mean(diff)  # Compute SMAPE for current batch
        self.total_smape += smape * target.numel()  # Multiply by batch size to prepare for mean
        self.total_count += target.numel()

    def compute(self):
        """
        Computes the mean of the accumulated SMAPE values.
        """
        return self.total_smape / self.total_count
    

class CosineAnnealDecayWarmRestarts(LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, max_lr_decay=0.9):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.max_lr_decay = max_lr_decay
        self.T_cur = last_epoch
        self.max_lrs = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur == 0 and self.last_epoch > 0:
            # Decay max_lr at each restart
            self.max_lrs = [lr * self.max_lr_decay for lr in self.max_lrs]
            return self.max_lrs
        else:
            return [self.eta_min + (max_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                    for max_lr in self.max_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = 0
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            
            
def save_figure_for_latex(fig, filename, dpi=500, bbox_inches='tight'):
    """
    Save a matplotlib or seaborn figure in a format suitable for LaTeX.
    
    Parameters:
    fig (matplotlib.figure.Figure): The figure to save
    filename (str): The filename to save the figure to (without extension)
    dpi (int): The resolution in dots per inch
    bbox_inches (str): The bounding box of the figure
    """
    # Ensure the filename ends with .pdf
    if not filename.lower().endswith('.pdf'):
        filename += '.pdf'
    
    # Save the figure as a PDF
    with PdfPages(filename) as pdf:
        pdf.savefig(fig, dpi=dpi, bbox_inches=bbox_inches)
    
    plt.close(fig)  # Close the figure to free up memory
    
    print(f"Figure saved as {filename}")