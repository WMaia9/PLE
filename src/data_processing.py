# src/data_processing.py

import numpy as np
import pandas as pd
import os
from typing import Dict, Any, List, Tuple
from .file_handler import load_and_clean_eem_csv

def _clean_label(p: str) -> str:
    """Helper function to group par/perp files."""
    return os.path.basename(p).lower().replace("perp", "").replace("par", "").replace(".csv", "").replace("_", "").strip()

def get_file_pairs(file_paths: List[str]) -> Tuple[Dict, Dict]:
    """Separates the dye pair from the sample pairs."""
    paired = {}
    for path in file_paths:
        label = _clean_label(path)
        if label not in paired:
            paired[label] = {}
        if "par" in path.lower():
            paired[label]['parallel'] = path
        elif "perp" in path.lower():
            paired[label]['perpendicular'] = path

    dye_label = next((label for label in paired if "dye" in label), None)
    if not dye_label:
        raise ValueError("No reference 'dye' files found.")
    
    dye_pair = {dye_label: paired.pop(dye_label)}
    sample_pairs = paired
    return dye_pair, sample_pairs

def _calculate_average_intensities(par_path: str, perp_path: str, params: Dict[str, Any]) -> Dict:
    """Processes a single par/perp file pair and returns a dictionary of results."""
    par_df = load_and_clean_eem_csv(par_path)
    perp_df = load_and_clean_eem_csv(perp_path)
    
    lambda_em, par_signal, lamp = par_df.iloc[:-1, 0].values, par_df.iloc[:-1, 1:].values, par_df.iloc[-1, 1:].values
    _, perp_signal, _ = perp_df.iloc[:-1, 0].values, perp_df.iloc[:-1, 1:].values, perp_df.iloc[-1, 1:].values
    
    num_excitation_points = par_signal.shape[1]
    lambda_ex = np.arange(450, 450 + num_excitation_points)

    background = params['background']
    par_bg = par_signal - background
    perp_bg = perp_signal - background

    par_lamp_corr = par_bg / lamp
    perp_lamp_corr = perp_bg / lamp

    lambda_0_key = next((p for p in params['lambda_0_dict'] if os.path.basename(p) == os.path.basename(par_path)), par_path)
    lambda_0 = params['lambda_0_dict'][lambda_0_key]
    
    window_size = params['window_size']
    half_window = window_size // 2
    
    i_star = np.argmin(np.abs(lambda_em - lambda_0))
    start_idx = max(i_star - half_window, 0)
    end_idx = min(i_star + half_window + 1, len(lambda_em))

    if end_idx - start_idx < window_size:
        if start_idx == 0:
            end_idx = start_idx + window_size
        elif end_idx == len(lambda_em):
            start_idx = end_idx - window_size

    par_avg_lamp_corr = par_lamp_corr[start_idx:end_idx, :].mean(axis=0)
    perp_avg_lamp_corr = perp_lamp_corr[start_idx:end_idx, :].mean(axis=0)
    par_avg_raw = par_bg[start_idx:end_idx, :].mean(axis=0)
    
    return {
        "lambda_ex": lambda_ex, "lambda_0": lambda_0, "raw_lamp_vector": lamp,
        "par_avg_raw": par_avg_raw, "par_avg_lamp_corr": par_avg_lamp_corr,
        "perp_avg_lamp_corr": perp_avg_lamp_corr,
    }

def calculate_g_factor_from_dye(dye_pair: Dict, params: Dict[str, Any]) -> pd.DataFrame:
    """Calculates the G-Factor from the dye files and returns a DataFrame."""
    label, pair_paths = list(dye_pair.items())[0]
    print(f"\n-> Calculating G-Factor (Cj) using '{label}'...")
    
    dye_results = _calculate_average_intensities(pair_paths['parallel'], pair_paths['perpendicular'], params)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        Cj = dye_results["par_avg_lamp_corr"] / dye_results["perp_avg_lamp_corr"]
    Cj = np.nan_to_num(Cj, nan=1.0)

    g_factor_df = pd.DataFrame({
        "Excitation Wavelength (nm)": dye_results["lambda_ex"],
        "Par Avg": dye_results["par_avg_lamp_corr"],
        "Perp Avg": dye_results["perp_avg_lamp_corr"],
        "Cj (Par / Perp)": Cj,
    })
    print("\n--- G-Factor Data Table (Dye) ---")
    print(g_factor_df.head(20))

    g_factor_df["perp_avg_g_corr"] = dye_results["perp_avg_lamp_corr"] * Cj
    g_factor_df["raw_lamp_vector"] = dye_results["raw_lamp_vector"]
    g_factor_df["par_avg_raw"] = dye_results["par_avg_raw"]
    return g_factor_df

def process_single_sample(sample_label: str, sample_pair: Dict, g_factor_df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Processes a single sample using the calculated G-Factor and returns a DataFrame."""
    print(f"-> Processing sample: '{sample_label}'")
    sample_results = _calculate_average_intensities(sample_pair['parallel'], sample_pair['perpendicular'], params)
    
    Cj = g_factor_df["Cj (Par / Perp)"].values
    perp_corrected = sample_results["perp_avg_lamp_corr"] * Cj
    par_avg = sample_results["par_avg_lamp_corr"]
    anisotropy = (par_avg - perp_corrected) / (par_avg + 2 * perp_corrected)
    
    final_df = pd.DataFrame({
        "Excitation Wavelength (nm)": sample_results["lambda_ex"],
        "Par Avg": par_avg,
        "Perp Avg": sample_results["perp_avg_lamp_corr"],
        "Correction Factor (Cj)": Cj,
        "Perp Corrected": perp_corrected,
        "Anisotropy": anisotropy,
        "Energy Relative to Lambda_0 (eV)": (1240 / sample_results["lambda_ex"]) - (1240 / sample_results["lambda_0"]),
        
        "lambda_0": sample_results["lambda_0"],
        "raw_lamp_vector": sample_results["raw_lamp_vector"],
        "par_avg_raw": sample_results["par_avg_raw"],
    })
    print(f"\n--- Results Table for '{sample_label}' ---")
    print(final_df.head(10))
    return final_df