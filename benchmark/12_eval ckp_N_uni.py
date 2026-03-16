# @title 12_eval_ckp_N_uni.py 
import subprocess, yaml, os
import pandas as pd
from pathlib import Path

# =====================================================================
# SCMamba Evaluation Aggregation Script
# Tự động quét cache, chạy eval_real_dataset.py cho các dataset còn thiếu
# và tổng hợp kết quả (MASE, CRPS, NLL...) ra file CSV.
# =====================================================================

# --- Cấu hình Môi trường ---
os.environ["TRITON_F32_DEFAULT"] = "ieee"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Cập nhật theo config mới từ training script
CONFIG_NAME = 'config.v_config06_uni_17data.yaml' 
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'checkpoints', 'SCMamba_v_config06_uni_17data_best_mase.pth')

QUICK_VALIDATE = False
SELECTED_DATASETS = 'all'      # Hoặc truyền list: ['cif_2016', 'nn5_weekly']
PRED_STYLE = 'multipoint'      # SC-Mamba luôn dùng multipoint

# Dành cho Quick Validation
QUICK_KEYS = ['cif_2016', 'm1_quarterly', 'nn5_weekly', 'tourism_quarterly', 'm1_monthly', 'hospital']

# DATASET_CONFIGS: Nguồn chân lý (Tên dataset, chu kỳ dự đoán, tần suất, Tên hiển thị, Lĩnh vực)
DATASET_CONFIGS = [
    ('car_parts_without_missing',      12, 'M', 'Car Parts',        'Retail'),
    ('cif_2016',                       12, 'M', 'CIF 2016',         'Finance/Competition'),
    ('covid_deaths',                   30, 'D', 'Covid Deaths',     'Epidemiology'),
    ('ercot',                          24, 'H', 'ERCOT Load',       'Energy'),
    ('exchange_rate',                  30, 'B', 'Exchange Rate',    'Forex Finance'),
    ('fred_md',                        12, 'M', 'FRED-MD',          'Macroeconomics'),
    ('hospital',                       12, 'M', 'Hospital',         'Healthcare'),
    ('m1_monthly',                     18, 'M', 'M1 Monthly',       'Economics'),
    ('m1_quarterly',                    8, 'Q', 'M1 Quarterly',     'Economics'),
    ('m3_monthly',                     18, 'M', 'M3 Monthly',       'Economics'),
    ('m3_quarterly',                    8, 'Q', 'M3 Quarterly',     'Economics'),
    ('nn5_daily_without_missing',      56, 'D', 'NN5 Daily',        'Banking'),
    ('nn5_weekly',                      8, 'W', 'NN5 Weekly',       'Banking'),
    ('tourism_monthly',                24, 'M', 'Tourism Monthly',  'Tourism'),
    ('tourism_quarterly',               8, 'Q', 'Tourism Quarterly','Tourism'),
    ('traffic',                        24, 'H', 'Traffic',          'Transportation'),
    ('weather',                        30, 'D', 'Weather',          'Meteorology'),
]

# --- Thiết lập Đường dẫn & Tên Cache ---
# Lấy chính xác tên checkpoint làm tên thư mục Cache (Giữ nguyên cả chữ _best_mase)
EVAL_MODEL_NAME = os.path.basename(CHECKPOINT_PATH).replace('.pth', '')
EVAL_DIR = Path(PROJECT_ROOT) / 'data' / 'real_data_evals' / EVAL_MODEL_NAME / PRED_STYLE
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Lọc dataset cần evaluate
if QUICK_VALIDATE:
    selected = [d for d in DATASET_CONFIGS if d[0] in QUICK_KEYS]
elif SELECTED_DATASETS == 'all':
    selected = DATASET_CONFIGS
else:
    selected = [d for d in DATASET_CONFIGS if d[0] in SELECTED_DATASETS]

# --- Kiểm tra Cache hiện tại ---
cached, missing = [], []
for gluonts_key, pred_len, freq, name, domain in selected:
    # 512 là MAX_LENGTH cố định chặn trên contextual string trong source
    yml_path = EVAL_DIR / f'{gluonts_key}_512.yml'
    if yml_path.exists():
        cached.append(name)
    else:
        missing.append(name)

print(f"{'='*60}")
print(f"📊 BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH")
print(f"{'='*60}")
print(f'Model name : {EVAL_MODEL_NAME}')
print(f'Checkpoint : {CHECKPOINT_PATH}')
print(f'Config YAML: {CONFIG_NAME}')
print(f'Eval dir   : {EVAL_DIR}')
print(f'Trạng thái : ✅ Cached: {len(cached)} | ❌ Missing: {len(missing)}')

# --- Gọi Subprocess chạy Evaluation cho dataset chưa có ---
if missing:
    print(f'▶ Sẽ chạy Evaluation cho {len(missing)} dataset mới: {missing}\n')
    cmd = [
        'python', f'{PROJECT_ROOT}/core/eval_real_dataset.py',
        '-c', CHECKPOINT_PATH,
        '-o', EVAL_MODEL_NAME,  # Ép thư mục output bằng chính tên Checkpoint
        '-cfg', os.path.join(PROJECT_ROOT, 'core', CONFIG_NAME) # Để tự lấy tag sub_day=True
    ]

    print(f'💻 Executing: {" ".join(cmd)}')
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f'\n⚠️ LỖI: eval_real_dataset.py bị crash (Mã lỗi {result.returncode})')
        print(f'--- STDERR ---\n{result.stderr}')
    else:
        print(f'\n✅ Evaluation Script chạy xong không gặp lỗi.')
else:
    print('\n✅ Tất cả datasets yêu cầu đều đã có Cache. Bỏ qua bước chạy mô hình.')

# --- Đọc Load Kết quả vào Pandas DataFrame ---
results = []
for gluonts_key, pred_len, freq, name, domain in selected:
    yml_path = EVAL_DIR / f'{gluonts_key}_512.yml'

    if not yml_path.exists():
        results.append({'dataset': name, 'key': gluonts_key, 'domain': domain,
                        'freq': freq, 'pred_len': pred_len, 'status': 'MISSING', 'MASE': None})
        continue

    with open(yml_path) as f:
        raw = yaml.safe_load(f)

    if not raw:
        results.append({'dataset': name, 'key': gluonts_key, 'domain': domain,
                        'freq': freq, 'pred_len': pred_len, 'status': 'EMPTY', 'MASE': None})
        continue

    metrics = next(iter(raw.values()), {})
    if metrics.get('mase') is None:
        results.append({'dataset': name, 'key': gluonts_key, 'domain': domain,
                        'freq': freq, 'pred_len': pred_len, 'status': 'PLACEHOLDER', 'MASE': None})
        continue

    mase_val = metrics.get("mase")
    mcrps_val = metrics.get("mcrps", metrics.get("crps_scaled"))
    
    mase_str = f"{mase_val:.4f}" if isinstance(mase_val, (int, float)) else "?"
    mcrps_str = f"{mcrps_val:.4f}" if isinstance(mcrps_val, (int, float)) else "?"

    results.append({
        'dataset': name, 'key': gluonts_key, 'domain': domain,
        'freq': freq, 'pred_len': pred_len, 'status': 'OK',
        'MASE': mase_val, 'MAE': metrics.get('mae'),
        'RMSE': metrics.get('rmse'), 'sMAPE': metrics.get('smape'),
        'NLL': metrics.get('nll'), 'CRPS': metrics.get('crps'),
        'mCRPS': mcrps_val, 
    })
    print(f'✅ [{domain[:15]:15s}] {name[:20]:20s} MASE={mase_str}  mCRPS={mcrps_str}')

# Lập bảng tổng kết
df = pd.DataFrame(results)
csv_name = f'{EVAL_MODEL_NAME}_quick_results.csv' if QUICK_VALIDATE else f'{EVAL_MODEL_NAME}_all_results.csv'
csv_path = os.path.join(PROJECT_ROOT, 'results', csv_name)
os.makedirs(os.path.join(PROJECT_ROOT, 'results'), exist_ok=True)
df.to_csv(csv_path, index=False)

# --- In Tổng kết ---
print(f'\n{"="*60}')
print(f'📊 TỔNG KẾT — {EVAL_MODEL_NAME}')
print(f'{"="*60}')

ok = df[df['status'] == 'OK']
if not ok.empty:
    cols = ['dataset', 'domain', 'freq', 'pred_len']
    for m in ['MASE', 'sMAPE', 'mCRPS', 'CRPS', 'NLL']:
        if m in ok.columns: cols.append(m)

    # Render table standard python (No Jupyter requirement)
    print(ok[cols].round(4).to_string(index=False))

    print(f'\n💎 Avg MASE: {ok["MASE"].mean():.4f}')
    if 'mCRPS' in ok.columns:
        print(f'💎 Avg mCRPS: {ok["mCRPS"].mean():.4f}')

err = df[df['status'] != 'OK']
if not err.empty:
    print(f'\n⚠️ Phát hiện {len(err)} datasets LỖI (Chưa eval được):')
    for _, r in err.iterrows():
        print(f'   [{r["status"]:12s}] {r["dataset"]}')

print(f'\n💾 Đã lưu full bảng biểu tại: {csv_path}')
