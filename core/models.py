import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.utils import PositionExpansion, CustomScaling
from core.constants import *
from core.blocks import *
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DIAG: set SC_MAMBA_DIAG=1 env var (or config['diag_prints']=True in train.py)
# to enable per-forward diagnostic prints. Checked at runtime, not import time.
def _is_diag():
    return os.environ.get('SC_MAMBA_DIAG', '0') == '1'
_diag_count = 0  # only log first N forwards to avoid spam

class SC_SSMModelBackbone(nn.Module):
    r"""
    Modified SSMModelNoPos from Mamba4Cast (Channel-Independent Backbone).
    Instead of projecting to a final 1D scalar (y_pred), it returns the dense 
    temporal embeddings for all assets Z \in R^{B \times N \times P_L \times D_model}.
    """
    def __init__(self,
            epsilon=1e-4,
            scaler='min_max',
            num_encoder_layers=3, # Usually 2 or 3
            embed_size=36,
            token_embed_len=1024, # D_model
            norm=True, 
            norm_type='layernorm',
            initial_gelu_flag=True,
            residual=False, 
            in_proj_norm=False,
            global_residual=False,
            linear_seq=15,
            mamba2=True,
            bidirectional=False,
            conv_d=4,
            d_state=128,
            block_expansion=2,
            chunk_size=256,  # Must match Mamba2 internal chunk_size (default=256); seq_len is padded to this multiple
            headdim=128,     # Must be >= max(BLOCK_SIZE_M)=128; headdim=64 (default) causes partial Triton blocks
            **kwargs
        ):
        super().__init__()
        self.epsilon = epsilon
        self.scaler = CustomScaling(scaler)
        self.embed_size = embed_size
        self.chunk_size = chunk_size  # stored for sequence padding in encode_temporal
        
        # Initial Expansion from 1D to D_model
        self.expand_target = nn.Linear(1, self.embed_size, bias=True) 
        self.target_marker = nn.Embedding(NUM_TASKS, self.embed_size)

        self.final_target_expansion = nn.Linear(self.embed_size, self.embed_size*2, bias=True)
        self.target_marker_expansion = nn.Linear(self.embed_size, self.embed_size*2, bias=True)
        
        self.num_encoder_layers = num_encoder_layers
        self.initial_gelu_flag = initial_gelu_flag
        self.concat_target = ConcatLayer(dim=1, name='AppendTarget')
        token_input_dim = self.embed_size * 2
        
        # Projection layer before Mamba blocks
        self.in_proj_layer = nn.Linear(token_input_dim, token_embed_len)
        self.global_residual = nn.Linear(token_input_dim*(linear_seq+1), token_embed_len) if global_residual else None
        self.linear_seq = linear_seq
        self.init_gelu = nn.GELU() if initial_gelu_flag else None
        self.norm = norm
        self.in_proj_norm = nn.LayerNorm(token_embed_len) if in_proj_norm else nn.Identity()
        
        if bidirectional:
            self.mamba_encoder_layers = nn.ModuleList([BiMambaEncoderBlock(token_embed_len, norm, norm_type, residual,
                                                                           d_state=d_state, block_expansion=block_expansion,
                                                                           mamba2=mamba2, conv_d=conv_d,
                                                                           chunk_size=chunk_size,
                                                                           headdim=headdim) for _ in range(self.num_encoder_layers)])
        else:
            self.mamba_encoder_layers = nn.ModuleList([SSMEncoderBlock(token_embed_len, norm, norm_type, residual,
                                                                       d_state=d_state, block_expansion=block_expansion,
                                                                       mamba2=mamba2, conv_d=conv_d,
                                                                       chunk_size=chunk_size,
                                                                       headdim=headdim) for _ in range(self.num_encoder_layers)])
        
        # STRIPPED: self.final_output = nn.Linear(token_embed_len, 1)

    def forward(self, x, prediction_length=None):
        ts, history, target_dates, task = x['ts'], x['history'], x['target_dates'], x['task']

        # Embed history
        history_channels = history.unsqueeze(-1)
        med_scale, scaled = self.scaler(history_channels, self.epsilon)
        
        embed_history = self.expand_target(scaled)
        embedded = self.final_target_expansion(embed_history)

        task_embed = self.target_marker(task) 
        target = self.target_marker_expansion(task_embed)

        # Forward through Temporal Blocks
        embedded_Z = self.encode_temporal(ts, embedded, target, prediction_length)

        return {'Z': embedded_Z, 'scale': med_scale}

    def encode_temporal(self, ts: torch.Tensor, embedded: torch.Tensor, target: torch.Tensor, prediction_length: int):
        # Concatenate encoded sequence and prediction targets
        x = self.concat_target([embedded, target])
        
        if self.global_residual:
            glob_res = x[:, -(self.linear_seq+1):, :].reshape(x.shape[0], -1)
            glob_res = self.global_residual(glob_res).unsqueeze(1).repeat(1, prediction_length, 1)
            
        x = self.in_proj_layer(x) 
        if self.init_gelu is not None:
            x = self.in_proj_norm(x) 
            x = self.init_gelu(x)
            
        # Pad sequence so Mamba2's SSD Triton kernel receives seq_len % chunk_size == 0.
        #
        # CRITICAL: nchunks = seq_len / chunk_size must be >= 2.
        # When nchunks=1, the _chunk_state_fwd_kernel autotuner hits an LLVM
        # compiler bug (IndexError: map::at) for certain BLOCK_SIZE_M tile configs.
        # Inputs can be as short as min_seq_len(30) + pred_len_min(10) = 40 tokens,
        # which when padded to 1 × chunk_size = 256 gives nchunks=1 — triggering the crash.
        # Guarantee: min padded length = 2 × chunk_size → nchunks ≥ 2.
        L = x.shape[1]
        min_padded = 2 * self.chunk_size                        # nchunks ≥ 2
        target_len = max(min_padded, math.ceil(L / self.chunk_size) * self.chunk_size)
        pad = target_len - L
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))  # zero-pad seq_len dim only

        # DIAG: log padding waste for first 3 forward passes
        global _diag_count
        if _is_diag() and _diag_count < 3:
            waste_pct = pad / target_len * 100 if target_len > 0 else 0
            print(f"  [DIAG:pad] seq_len={L} → padded={target_len} (pad={pad}, waste={waste_pct:.1f}%), chunk_size={self.chunk_size}")

        # Extract temporal dynamics via sequential Mamba blocks
        for encoder_layer in self.mamba_encoder_layers:
            x = encoder_layer(x)

        # Restore original sequence length before prediction window slicing
        if pad > 0:
            x = x[:, :L, :]
            
        # We extract only the prediction window temporal embeddings
        if self.global_residual:
            x = torch.concat([x[:, -prediction_length:, :], glob_res], dim=-1)
        else:
            x = x[:, -prediction_length:, :]
            
        return x # Shape: [Batch_N, Pred_Len, D_model]


class SpectralVariationalLayer(nn.Module):
    """
    Phase 1 & Phase 2 Cross-Asset Graph Interaction Layer.
    Executes a continuous-prior Causal Fourier filter across the Asset dimension.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # Variational inference MLPs mapping Complex spectrum to mu and sigma
        # We double the dim because RFFT produces complex numbers (Real, Imag)
        complex_dim = d_model * 2
        
        self.mu_net = nn.Sequential(
            nn.Linear(complex_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, complex_dim)
        )
        
        self.log_var_net = nn.Sequential(
            nn.Linear(complex_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, complex_dim)
        )
        
        # Learnable causal cut-off threshold τ — initialized at 0.1 so the filter
        # starts with moderate sparsity and learns the right threshold from data.
        self.tau = nn.Parameter(torch.tensor(0.1))

        # α controls the steepness of the sigmoid approximating a hard step-function.
        # DESIGN CHOICE: Made learnable (nn.Parameter) rather than hard-coded at 50.
        # Rationale: hard α=50 causes sigmoid saturation early in training → gradient≈0
        # through the mask → τ cannot update. Starting at α=1.0 keeps the mask soft
        # (smooth gradient landscape). The model naturally sharpens α as τ converges.
        # Clamped in forward() to [0.5, 50] to prevent degenerate solutions.
        self.log_alpha = nn.Parameter(torch.tensor(0.0))  # exp(0.0) = 1.0 initial alpha
        
    def forward(self, Z_real, N_assets):
        """
        Z_real shape: [Batch, N_assets, Pred_Len, D_model]
        Since Mamba4Cast flattens Batch x N_assets, we reshape it first.
        """
        B_N, P_L, D = Z_real.shape
        B = B_N // N_assets
        
        # Reshape to explicitly expose the Asset dimension (Dim 1)
        Z_spatial = Z_real.view(B, N_assets, P_L, D)
        
        # 1. Continuous Reparameterization in Fourier Domain (1D-FFT along Assets)
        # Output shape: [B, FLOOR(N/2)+1, P_L, D] (Complex Tensor)
        H_freq = torch.fft.rfft(Z_spatial, dim=1)
        
        # Decompose complex numbers into real mappings for Variational Inference
        H_real = H_freq.real
        H_imag = H_freq.imag
        H_concat = torch.cat([H_real, H_imag], dim=-1) # Shape: [B, Freq_Bins, P_L, 2*D]
        
        # Compute Prior parameters
        mu_F = self.mu_net(H_concat)
        log_var_F = self.log_var_net(H_concat)
        sigma_F = torch.exp(0.5 * log_var_F)
        
        # Reparameterization Trick (Sampling F)
        epsilon = torch.randn_like(sigma_F)
        F_sampled_concat = mu_F + sigma_F * epsilon if self.training else mu_F
        
        # Split back into complex components
        F_real, F_imag = torch.chunk(F_sampled_concat, 2, dim=-1)
        F_complex = torch.complex(F_real, F_imag)
        
        # 2. Causal Spectral Filtering (Learnable Hard-Thresholding)
        # Calculate amplitude |F_t(k)|
        amplitude = torch.abs(F_complex)

        # α is computed from log_alpha to keep it strictly positive; clamped to [0.5, 50]
        # so the sigmoid never fully saturates (gradient vanishing) during training.
        alpha = torch.clamp(torch.exp(self.log_alpha), min=0.5, max=50.0)

        # Filter mask M_t(k) using sigmoid to approximate step function.
        # Starts soft (α≈1) for stable gradient flow to τ; sharpens as model converges.
        mask = torch.sigmoid(alpha * (amplitude - self.tau))
        
        # Apply Causal Mask
        F_hat = mask * F_complex
        
        # 3. Message Passing (Pointwise multiplication in frequency)
        H_updated_freq = F_hat * H_freq
        
        # 4. Inverse Fourier Transform for Spatial Recovery
        Z_spatial_updated = torch.fft.irfft(H_updated_freq, n=N_assets, dim=1)
        
        # Flatten back matching Mamba4Cast baseline shapes
        Z_updated = Z_spatial_updated.view(B * N_assets, P_L, D)
        
        # Compute KL Divergence between Sampled F \sim N(mu, sigma) and Prior N(0, 1)
        kl_loss = -0.5 * torch.sum(1 + log_var_F - mu_F.pow(2) - log_var_F.exp())
        # Normalize KL across Batch and Asset Frequency Bins
        kl_loss = kl_loss / (B * (N_assets // 2 + 1) * P_L * D * 2)

        # DIAG: log spectral layer stats for first 3 forward passes
        global _diag_count
        if _is_diag() and _diag_count < 3:
            _diag_count += 1
            with torch.no_grad():
                fft_diff = (H_freq - torch.fft.rfft(Z_spatial, dim=1)).abs().mean().item()
                print(f"  [DIAG:spectral] N_assets={N_assets}, freq_bins={H_freq.shape[1]}, mask=[{mask.min().item():.4f}, {mask.max().item():.4f}], alpha={alpha.item():.2f}, tau={self.tau.item():.4f}, KL={kl_loss.item():.6f}, fft_identity_diff={fft_diff:.2e}")
        
        return Z_updated, kl_loss
        

class SCMamba_Forecaster(nn.Module):
    r"""
    Main Pipeline Wrapper combining Mamba4Cast backbone and SpectralVariationalLayer.
    Produces both Point Estimate (\mu) and Uncertainty (\sigma^2) for NLL loss.
    """
    def __init__(self, N_assets=1, ssm_config=None):
        super().__init__()
        self.N_assets = N_assets
        
        if ssm_config is None:
            ssm_config = {}
            
        # Inherit all temporal CI configurations
        self.backbone = SC_SSMModelBackbone(**ssm_config)
        
        token_embed_len = ssm_config.get('token_embed_len', 1024)
        if ssm_config.get('global_residual', False):
            token_embed_len = token_embed_len * 2
            
        self.spectral_layer = SpectralVariationalLayer(d_model=token_embed_len)
        
        # Stochastic Prediction Heads
        self.mu_head = nn.Linear(token_embed_len, 1)
        self.sigma_head = nn.Sequential(
            nn.Linear(token_embed_len, 1),
            nn.Softplus() # Enforces strictly positive variance
        )

    def forward(self, x, prediction_length=None):
        # 1. Temporal Encoding (Channel-Independent)
        backbone_out = self.backbone(x, prediction_length)
        Z = backbone_out['Z']
        med_scale = backbone_out['scale']
        
        # 2. Spectral Causal Message Passing
        Z_graph, kl_loss = self.spectral_layer(Z, self.N_assets)
        
        # 3. Distributional Decoding
        mu = self.mu_head(Z_graph).squeeze(-1)
        sigma2 = self.sigma_head(Z_graph).squeeze(-1) + 1e-6 # Add epsilon for numerical stability
        
        return {
            'mu': mu,
            'sigma2': sigma2,
            'scale': med_scale,
            'kl_loss': kl_loss
        }
