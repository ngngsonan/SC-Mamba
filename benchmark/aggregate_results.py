import os
import pandas as pd
import yaml
import glob
import argparse

def aggregate_results(models_to_evaluate, base_dir="../../data/real_data_evals"):
    """
    Reads evaluation dictionaries for specified baseline and ablation models, 
    and outputs a compiled CSV table representing "Table 3" from Mamba4cast.
    """
    all_results = []
    
    for model_name in models_to_evaluate:
        # Assuming multipoint evaluation mode for SC-Mamba
        eval_path = os.path.join(base_dir, model_name, "multipoint")
        if not os.path.exists(eval_path):
            print(f"[Warning] Evaluations for {model_name} not found in {eval_path}.")
            continue
            
        yaml_files = glob.glob(os.path.join(eval_path, "*.yml"))
        
        for yml_file in yaml_files:
            dataset_name = os.path.basename(yml_file).split("_512.yml")[0]
            with open(yml_file, "r") as f:
                res = yaml.load(f, Loader=yaml.SafeLoader)
                
            # Usually res has keys like '512_30' mapping to the dict
            for key, metrics in res.items():
                row = {
                    "Model": model_name,
                    "Dataset": dataset_name,
                    "Context_Len": key.split('_')[0],
                    "Pred_Len": key.split('_')[1],
                    "MASE": f"{metrics.get('mase', 0):.3f}",
                    "sMAPE": f"{metrics.get('smape', 0):.3f}",
                    "CRPS_Proxy_NLL": f"{metrics.get('nll', 0):.3f}" # Using Expected NLL as pseudo-proxy
                }
                all_results.append(row)
                
    if not all_results:
        print("No valid results to aggregate.")
        return
        
    df = pd.DataFrame(all_results)
    
    # Pivot table to present Datasets as Rows and Models as Columns for MASE
    try:
        pivot_mase = df.pivot(index='Dataset', columns='Model', values='MASE')
        print("\n=== Table 3 (MASE) Equivalent ===")
        print(pivot_mase.to_string())
        pivot_mase.to_csv("table3_mase_benchmark.csv")
        print("\nSaved Table 3 MASE results to: benchmark/table3_mase_benchmark.csv")
    except Exception as e:
        print(f"Error pivoting MASE: {e}")
        
    # Pivot table to present Datasets as Rows and Models as Columns for NLL (Uncertainty)
    try:
        pivot_nll = df.pivot(index='Dataset', columns='Model', values='CRPS_Proxy_NLL')
        print("\n=== Uncertainty Benchmark (NLL) ===")
        print(pivot_nll.to_string())
        pivot_nll.to_csv("table4_nll_benchmark.csv")
        print("Saved Uncertainty results to: benchmark/table4_nll_benchmark.csv")
    except Exception as e:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs='+', default=["sc_mamba_main_v1", "sc_mamba_ablation_no_filter", "sc_mamba_ablation_ci_only"], help="List of models to aggregate")
    args = parser.parse_args()
    
    aggregate_results(args.models)
