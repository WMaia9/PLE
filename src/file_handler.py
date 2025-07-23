import pandas as pd
import yaml
import os
from datetime import datetime
from typing import Dict, Any

def load_and_clean_eem_csv(filepath: str) -> pd.DataFrame:
    try:
        df_raw = pd.read_csv(filepath, header=None)
        df_clean = df_raw.drop(index=[0, 1]).reset_index(drop=True)
        return df_clean.astype(float)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file {filepath} was not found.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while processing {filepath}: {e}")

def save_run_config(run_dir: str, params: Dict[str, Any], file_paths: list):
    config_to_save = {
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": params,
        "input_files": [os.path.basename(f) for f in file_paths]
    }
    
    yaml_path = os.path.join(run_dir, "run_config.yaml")
    with open(yaml_path, "w", encoding='utf-8') as f:
        yaml.dump(config_to_save, f, default_flow_style=False, sort_keys=False)
    
    print(f"Run configuration saved to {yaml_path}")