import numpy as np
import pandas as pd
import os
from typing import Dict, Any, List, Tuple
from .file_handler import load_and_clean_eem_csv

def _clean_label(p: str) -> str:
    return os.path.basename(p).lower().replace("perp", "").replace("par", "").replace("_", "").strip()

def _pair_files(file_paths: List[str]) -> Dict[str, Dict[str, str]]:
    paired = {}
    for path in file_paths:
        label = _clean_label(path)
        if label not in paired:
            paired[label] = {}
        if "par" in path.lower():
            paired[label]['parallel'] = path
        elif "perp" in path.lower():
            paired[label]['perpendicular'] = path
    return paired

def _get_avg_intensities(par_path: str, perp_path: str, params: Dict[str, Any]) -> Tuple:
    df_par = load_and_clean_eem_csv(par_path)
    lambda_em, par_signal, lamp = df_par.iloc[:-1, 0].values, df_par.iloc[:-1, 1:].values, df_par.iloc[-1, 1:].values
    lambda_ex = np.array(df_par.columns[1:], dtype=float)

    df_perp = load_and_clean_eem_csv(perp_path)
    perp_signal = df_perp.iloc[:-1, 1:].values
    
    par_corr = (par_signal - params['background']) / lamp
    perp_corr = (perp_signal - params['background']) / lamp
    
    lambda_0_key = next((p for p in params['lambda_0_dict'] if os.path.basename(p) == os.path.basename(par_path)), par_path)
    lambda_0 = params['lambda_0_dict'][lambda_0_key]
    i_star = np.argmin(np.abs(lambda_em - lambda_0))
    start_idx = max(i_star - params['half_window_pts'], 0)
    end_idx = min(i_star + params['half_window_pts'] + 1, len(lambda_em))

    par_avg = par_corr[start_idx:end_idx, :].mean(axis=0)
    perp_avg = perp_corr[start_idx:end_idx, :].mean(axis=0)
    
    return par_avg, perp_avg, lambda_ex, lambda_0

def calculate_anisotropy(par_avg: np.ndarray, perp_avg: np.ndarray, g_factor: np.ndarray) -> np.ndarray:
    perp_corrected = perp_avg * g_factor
    numerator = par_avg - perp_corrected
    denominator = par_avg + 2 * perp_corrected
    return np.divide(numerator, denominator, out=np.zeros(numerator.shape, dtype=float), where=denominator!=0)

def process_anisotropy_run(file_paths: List[str], params: Dict[str, Any]) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    paired_files = _pair_files(file_paths)
    results = {}
    
    dye_label = next((label for label in paired_files if "dye" in label), None)
    if not dye_label:
        raise ValueError("No reference 'dye' files found to calculate G-factor.")

    print(f"\n[1/3] Calculating G-Factor using '{dye_label}'...")
    dye_pair = paired_files.pop(dye_label)
    dye_par_avg, dye_perp_avg, lambda_ex, _ = _get_avg_intensities(dye_pair['parallel'], dye_pair['perpendicular'], params)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        c_vector = dye_par_avg / dye_perp_avg
    
    if params['correction_mode'] == 'scalar':
        c_scalar = np.nanmean(c_vector)
        g_factor_to_apply = np.full_like(c_vector, c_scalar)
        print(f"  - Using SCALAR G-Factor: {c_scalar:.4f}")
    else:
        g_factor_to_apply = np.nan_to_num(c_vector, nan=1.0)
        print("  - Using VECTOR G-Factor.")
        
    g_factor_df = pd.DataFrame({'Excitation Wavelength (nm)': lambda_ex, 'G_Factor': g_factor_to_apply})

    print("\n[2/3] Calculating Anisotropy for samples...")
    for label, pair in paired_files.items():
        if 'parallel' not in pair or 'perpendicular' not in pair:
            print(f"  - Skipping '{label}': missing a par/perp file.")
            continue
        
        print(f"  - Processing sample: '{label}'")
        par_avg, perp_avg, lambda_ex, lambda_0 = _get_avg_intensities(pair['parallel'], pair['perpendicular'], params)
        anisotropy = calculate_anisotropy(par_avg, perp_avg, g_factor_to_apply)
        
        df = pd.DataFrame({
            "Excitation Wavelength (nm)": lambda_ex,
            "Relative Energy (eV)": (1240 / lambda_ex) - (1240 / lambda_0),
            "Anisotropy": anisotropy,
            "I_parallel_avg": par_avg,
            "I_perpendicular_avg": perp_avg,
            "I_perp_corrected": perp_avg * g_factor_to_apply,
        })
        results[label] = df
        
    print("\n[3/3] Analysis complete.")
    return results, g_factor_df