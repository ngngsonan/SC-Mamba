# COLAB 15/03/2026 Environments setup for mamba_ssm, causal_conv1d
# 1. Gỡ bỏ sạch sẽ các thư viện default của Colab có khả năng gây xung đột
!pip uninstall -y transformers sentence-transformers torch torchvision torchaudio

# 2. Cài đặt Torch 2.4.0 (chuẩn cu121) và Transformers 4.39.3 (tương thích Mamba)
!pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install transformers==4.39.3 packaging triton==3.0.0

# 3. Tải và tự động đổi tên file wheel của Causal-Conv1d
!wget -qO causal_conv1d-1.4.0-cp312-cp312-linux_x86_64.whl "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0%2Bcu122torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

# 4. Tải và tự động đổi tên file wheel của Mamba-SSM
!wget -qO mamba_ssm-2.2.4-cp312-cp312-linux_x86_64.whl "https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4%2Bcu12torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

# 5. Cài đặt trực tiếp (tốc độ cao, không tốn disk, không cần build)
!pip install causal_conv1d-1.4.0-cp312-cp312-linux_x86_64.whl
!pip install mamba_ssm-2.2.4-cp312-cp312-linux_x86_64.whl
!pip install yfinance gluonts tqdm utilsforecast pyyaml pandas numpy submitit torchmetrics gpytorch

from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/ngngsonan/SC-Mamba.git /content/SC-Mamba 2>/dev/null || echo 'Repo already cloned'
!cd /content/SC-Mamba && git pull


import os, sys
# ── Critical env vars for Triton 3.x on T4/A100 ─────────────────────────────
# TRITON_F32_DEFAULT=ieee prevents IndexError: map::at on Turing GPUs.
# Must be set BEFORE any mamba_ssm or triton import.
os.environ['TRITON_F32_DEFAULT'] = 'ieee'
os.environ['SC_MAMBA_DIAG'] = '1'

PROJECT_ROOT = '/content/SC-Mamba'
CKPT_DIR = '/content/drive/MyDrive/Colab Notebooks/SCMamba/sc_mamba_checkpoints'

os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
print(f"✅ CWD = {os.getcwd()}")

# Sanity-check key files
for path in ['core/train.py', 'core/models.py', 'core/real_data_val_pipeline.py', 'core/real_data_args.yaml']:
    status = "✅" if os.path.exists(path) else "❌ MISSING"
    print(f"  {status}  {path}")
