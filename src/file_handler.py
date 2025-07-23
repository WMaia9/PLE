import pandas as pd
import yaml
from typing import Dict, Any

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_and_clean_eem_csv(filepath: str) -> pd.DataFrame:
    """
    Loads a spectroscopy CSV file, removes header rows,
    and converts data types to float.

    Args:
        filepath: The path to the input CSV file.

    Returns:
        A cleaned pandas DataFrame.
    """
    try:
        df_raw = pd.read_csv(filepath, header=None)
        # Drop the first two metadata rows
        df_clean = df_raw.drop(index=[0, 1]).reset_index(drop=True)
        # Convert all columns to float
        return df_clean.astype(float)
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        raise
    except Exception as e:
        print(f"An error occurred while processing {filepath}: {e}")
        raise