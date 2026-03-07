import os
import pickle
import numpy as np
from gluonts.dataset.util import to_pandas
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.repository import dataset_names
import pandas as pd
from tqdm import tqdm
import yfinance as yf
import datetime

MAX_LENGTH = 512

# The ones in the paper
REAL_DATASETS = {
    "nn5_daily_without_missing": 56,
    "nn5_weekly": 8,
    "covid_deaths": 30,
    "weather": 30,
    "hospital": 12, 
    "fred_md": 12,
    "car_parts_without_missing": 12,
    "traffic": 24,
    "m3_monthly": 18,
    "ercot": 24,
    "m1_monthly": 18,
    "m1_quarterly": 8,
    "cif_2016": 12,
    "exchange_rate": 30,
    "m3_quarterly": 8,
    "tourism_monthly": 24,
    "tourism_quarterly": 8,
    "nasdaq": 30, # SC-Mamba Special Addition 
}

def create_real_val_datasets(pad: bool=False):

    for dataset, pred_len in REAL_DATASETS.items():
        
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)

        save_path = f"{script_dir}/../../data/real_val_datasets"
        os.makedirs(save_path, exist_ok = True)
        padded = 'pad' if pad else 'nopad'

        if os.path.exists(f"{save_path}/{dataset}_{padded}_{MAX_LENGTH}.pkl"):
            print(f"Dataset {dataset} already exists. Skipping...")
            continue

        print(f"Processing {dataset}")
        
        test_dfs = [] # Standard format expected by the subsequent logic
        dataframes = [] # Structure to hold the processed Dataframes
        
        if dataset == "nasdaq":
            print("Fetching NASDAQ 100 via yfinance...")
            # We fetch a predefined list of top NASDAQ components (for stability and reproducibility)
            top_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'AVGO', 'PEP', 'COST', 
                           'CSCO', 'TMUS', 'ADBE', 'TXN', 'QCOM', 'HON', 'AMGN', 'INTC', 'SBUX', 'GILD'] # Limited to 20 for computational speed in parsing, can be expanded to 100
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=5*365) # 5 years historical data
            
            # Fetch explicitly as single threaded to avoid yfinance rate limits
            for ticker in tqdm(top_tickers, desc="Downloading Tickers"):
                try:
                    df_ticker = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    if not df_ticker.empty:
                        # We only need the Close price. Ensure it's a 1D Series indexed by Date
                        target_series = df_ticker['Close']
                        # Re-index to ensure continuous business days, forward fill gaps
                        all_bds = pd.date_range(start=target_series.index.min(), end=target_series.index.max(), freq='B')
                        target_series = target_series.reindex(all_bds).ffill().bfill()
                        test_dfs.append(target_series)
                except Exception as e:
                    print(f"Failed to fetch {ticker}: {e}")
            
            # Instead of reading data.test, we explicitly mock sizes
            sizes = [len(series) for series in test_dfs]
        
        else:
            if dataset == "ercot":
                data = get_dataset(dataset, regenerate=True)
            else:
                data = get_dataset(dataset, regenerate=True, prediction_length=pred_len)
            
            from concurrent.futures import ThreadPoolExecutor  
            with ThreadPoolExecutor(max_workers=5) as executor:
                test_dfs = list(executor.map(to_pandas, list(data.test)))
                
            sizes = []
            for series in list(data.test):
                sizes.append(series["target"].shape[0])
            
        max_series_len = min(max(sizes), MAX_LENGTH + pred_len) # changed this to account for pred len
        
        padded_series = []
        for series in tqdm(test_dfs):
            current_length = len(series)

            if current_length < max_series_len:
                if pad:
                    # Create an array of zeros with the necessary padding length
                    padding = np.zeros(max_series_len - current_length)
                    # Combine the current series values with padding
                    padded_array = np.concatenate([padding, series.values])
                    # Handle the index
                    padded_index = [pd.NaT] * (max_series_len - current_length) + series.index.tolist()
                    # Create a new Series
                    new_series = pd.Series(padded_array, index=padded_index)
                else:
                    new_series = series # Do not pad
            else:
                new_series = series.iloc[-max_series_len:]
            padded_series.append(new_series)


        for i, series in enumerate(padded_series):
            df = series.reset_index()
            df.rename(columns={'index': 'date', 0: "target"}, inplace=True)
            df['Series'] = i  # Add the series identifier
            dataframes.append(df)
            
    # Concatenate all the individual dataframes
        stacked_df = pd.concat(dataframes, ignore_index=False)
        stacked_df.set_index(['Series', stacked_df.index], inplace=True)
        stacked_df.date = stacked_df.date.apply(lambda x: x.to_timestamp())
        
        with open(f"{save_path}/{dataset}_{padded}_{MAX_LENGTH}.pkl", "wb") as f:
            pickle.dump(stacked_df, f)

if __name__ == "__main__":
    create_real_val_datasets(pad=False)
    print("Done")