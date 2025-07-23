import numpy as np
import pandas as pd
from typing import Callable

def split_eem_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splits the DataFrame into emission wavelengths, signal, and lamp."""
    lambda_em = df.iloc[:-1, 0].values
    signal = df.iloc[:-1, 1:].values  # [emission x excitation] matrix
    lamp = df.iloc[-1, 1:].values     # [excitation] vector
    return lambda_em, signal, lamp

def process_raw_to_avg_intensity(
    par_filepath: str,
    perp_filepath: str,
    bg_level: float,
    lambda_0: float,
    window_size: int,
    file_loader_func: Callable[[str], pd.DataFrame]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full pipeline to load, clean, correct, and calculate the average intensity
    for parallel and perpendicular signals.
    """
    # Load and process parallel data
    df_par = file_loader_func(par_filepath)
    lambda_em, par_signal, lamp = split_eem_data(df_par)
    par_bg_sub = par_signal - bg_level
    par_corr = par_bg_sub / lamp

    # Load and process perpendicular data
    df_perp = file_loader_func(perp_filepath)
    _, perp_signal, _ = split_eem_data(df_perp)
    perp_bg_sub = perp_signal - bg_level
    perp_corr = perp_bg_sub / lamp

    # Find integration window around the emission peak
    i_star = np.argmin(np.abs(lambda_em - lambda_0))
    half_window = window_size // 2
    start_idx = max(i_star - half_window, 0)
    end_idx = min(i_star + half_window + 1, len(lambda_em))

    # Calculate average intensities within the window
    par_avg = par_corr[start_idx:end_idx, :].mean(axis=0)
    perp_avg = perp_corr[start_idx:end_idx, :].mean(axis=0)
    
    # Generate the excitation axis (assuming 1nm steps starting from 450nm)
    n_excitation = par_avg.shape[0]
    lambda_ex = np.arange(450, 450 + n_excitation) 

    return par_avg, perp_avg, lambda_ex, lamp

def calculate_g_factor(par_avg: np.ndarray, perp_avg: np.ndarray) -> np.ndarray:
    """Calculates the G-factor correction as the ratio I_par / I_perp."""
    return par_avg / perp_avg

def calculate_anisotropy(
    par_avg: np.ndarray,
    perp_avg: np.ndarray,
    g_factor: np.ndarray
) -> np.ndarray:
    """Calculates the fluorescence anisotropy (r)."""
    perp_corrected = perp_avg * g_factor
    numerator = par_avg - perp_corrected
    denominator = par_avg + 2 * perp_corrected
    # Avoid division by zero, return 0 where denominator is 0
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)

def calculate_relative_energy(lambda_ex: np.ndarray, lambda_0: float) -> np.ndarray:
    """Calculates the excitation energy relative to the emission peak in eV."""
    # Using the relation E = hc/λ, where hc ≈ 1240 eV·nm
    return (1240 / lambda_ex) - (1240 / lambda_0)