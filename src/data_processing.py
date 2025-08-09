# src/data_processing.py

from __future__ import annotations

import os
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd


# =========================
# File pairing utilities
# =========================

def _clean_label(file_object_or_path) -> str:
    """
    Make a friendly label from either a filesystem path or a Streamlit UploadedFile.
    Example: 'sample1_par.csv' -> 'sample1'
    """
    filename = os.path.basename(
        file_object_or_path.name if hasattr(file_object_or_path, "name") else file_object_or_path
    )
    return (
        filename.lower()
        .replace("perp", "")
        .replace("parallel", "")
        .replace("par", "")
        .replace(".csv", "")
        .replace("_", "")
        .strip()
    )


def get_file_pairs(file_list: List) -> Tuple[Dict, Dict]:
    """
    Group uploaded files into:
      - dye_pair: {dye_label: {"parallel": UploadedFile, "perpendicular": UploadedFile}}
      - sample_pairs: {sample_label: {"parallel": UploadedFile, "perpendicular": UploadedFile}}

    Works with either filenames (str) OR Streamlit UploadedFile objects.
    """
    paired: Dict[str, Dict[str, Any]] = {}
    for file_obj in file_list:
        label = _clean_label(file_obj)
        if label not in paired:
            paired[label] = {}

        filename = file_obj.name if hasattr(file_obj, "name") else os.path.basename(file_obj)
        low = filename.lower()
        if "par" in low and "parallel" not in paired[label]:
            paired[label]["parallel"] = file_obj
        if "perp" in low and "perpendicular" not in paired[label]:
            paired[label]["perpendicular"] = file_obj

    dye_label = next((lab for lab in paired if "dye" in lab), None)
    if not dye_label:
        raise ValueError("No reference 'dye' files found (expect filenames containing 'dye').")

    dye_pair = {dye_label: paired.pop(dye_label)}
    sample_pairs = paired
    return dye_pair, sample_pairs


# =========================
# Page-1 core helpers
# (Keeps your original behavior)
# =========================

def _calculate_average_intensities(par_file, perp_file, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single par/perp file pair and return averaged signals around λ0.

    CSV format expected (your existing convention):
      - We drop the first two rows (headers).
      - Column 0 (after dropping) = emission wavelength (nm)
      - Columns 1..N (rows except last) = intensities for each excitation (one column per λ_ex)
      - Last row = lamp vs excitation (across columns 1..N)
    """
    par_df = pd.read_csv(par_file, header=None).drop(index=[0, 1]).reset_index(drop=True).astype(float)
    perp_df = pd.read_csv(perp_file, header=None).drop(index=[0, 1]).reset_index(drop=True).astype(float)

    # Emission axis is the first column except the last lamp row
    lambda_em = par_df.iloc[:-1, 0].values.astype(float)

    # Signals are rows except the last (lamp) and columns from 1 onward
    par_signal = par_df.iloc[:-1, 1:].values.astype(float)
    perp_signal = perp_df.iloc[:-1, 1:].values.astype(float)

    # Lamp vector is the last row, columns from 1 onward
    lamp = par_df.iloc[-1, 1:].values.astype(float)

    # Build a dummy excitation axis of matching length (keeps your original behavior)
    num_excitation_points = par_signal.shape[1]
    lambda_ex = np.arange(450, 450 + num_excitation_points, dtype=float)

    # Background subtraction (scalar)
    background = float(params["background"])
    par_bg = par_signal - background
    perp_bg = perp_signal - background

    # Lamp correction (per excitation)
    par_lamp_corr = par_bg / lamp
    perp_lamp_corr = perp_bg / lamp

    # λ0 lookup by filename (keep parity with your sidebar mapping)
    par_filename = par_file.name if hasattr(par_file, "name") else os.path.basename(par_file)
    lambda_0_key = next((p for p in params["lambda_0_dict"] if os.path.basename(p) == par_filename), par_filename)
    lambda_0 = float(params["lambda_0_dict"][lambda_0_key])

    # Emission window around λ0 (in points)
    window_size = int(params["window_size"])
    half_window = window_size // 2

    i_star = int(np.argmin(np.abs(lambda_em - lambda_0)))
    start_idx = max(i_star - half_window, 0)
    end_idx = min(i_star + half_window + 1, len(lambda_em))

    # Pad window if clipped
    if end_idx - start_idx < window_size:
        if start_idx == 0:
            end_idx = min(len(lambda_em), start_idx + window_size)
        elif end_idx == len(lambda_em):
            start_idx = max(0, end_idx - window_size)

    # Average within emission window
    par_avg_lamp_corr = par_lamp_corr[start_idx:end_idx, :].mean(axis=0)
    perp_avg_lamp_corr = perp_lamp_corr[start_idx:end_idx, :].mean(axis=0)
    par_avg_raw = (par_signal[start_idx:end_idx, :] - 0).mean(axis=0)  # raw avg (pre-lamp, pre-bg if you want to compare)

    return {
        "lambda_ex": lambda_ex,
        "lambda_0": lambda_0,
        "raw_lamp_vector": lamp,
        "par_avg_raw": par_avg_raw,
        "par_avg_lamp_corr": par_avg_lamp_corr,
        "perp_avg_lamp_corr": perp_avg_lamp_corr,
    }


def calculate_g_factor_from_dye(dye_pair: Dict, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute dye G-factor (Cj = Par/Perp) vs excitation, keeping your original structure.
    """
    label, file_objects = list(dye_pair.items())[0]
    print(f"\n-> Calculating G-Factor (Cj) using '{label}'...")

    dye_results = _calculate_average_intensities(file_objects["parallel"], file_objects["perpendicular"], params)

    with np.errstate(divide="ignore", invalid="ignore"):
        Cj = dye_results["par_avg_lamp_corr"] / dye_results["perp_avg_lamp_corr"]
    Cj = np.nan_to_num(Cj, nan=1.0, posinf=1.0, neginf=1.0)

    g_factor_df = pd.DataFrame(
        {
            "Excitation Wavelength (nm)": dye_results["lambda_ex"],
            "Par Avg": dye_results["par_avg_lamp_corr"],
            "Perp Avg": dye_results["perp_avg_lamp_corr"],
            "Cj (Par / Perp)": Cj,
        }
    )
    print("\n--- G-Factor Data Table (Dye) ---")
    print(g_factor_df.head(20))

    # Keep extra columns you were storing
    g_factor_df["perp_avg_g_corr"] = dye_results["perp_avg_lamp_corr"] * Cj
    g_factor_df["raw_lamp_vector"] = dye_results["raw_lamp_vector"]
    g_factor_df["par_avg_raw"] = dye_results["par_avg_raw"]
    return g_factor_df


def process_single_sample(
    sample_label: str,
    sample_pair: Dict[str, Any],
    g_factor_df: pd.DataFrame,
    params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Process one sample using previously computed dye G-factor.
    """
    print(f"-> Processing sample: '{sample_label}'")
    sample_results = _calculate_average_intensities(
        sample_pair["parallel"], sample_pair["perpendicular"], params
    )

    Cj = g_factor_df["Cj (Par / Perp)"].to_numpy()
    perp_corrected = sample_results["perp_avg_lamp_corr"] * Cj
    par_avg = sample_results["par_avg_lamp_corr"]

    with np.errstate(divide="ignore", invalid="ignore"):
        anisotropy = (par_avg - perp_corrected) / (par_avg + 2.0 * perp_corrected)
    anisotropy = np.nan_to_num(anisotropy, nan=0.0)

    final_df = pd.DataFrame(
        {
            "Excitation Wavelength (nm)": sample_results["lambda_ex"],
            "Par Avg": par_avg,
            "Perp Avg": sample_results["perp_avg_lamp_corr"],
            "Correction Factor (Cj)": Cj,
            "Perp Corrected": perp_corrected,
            "Anisotropy": anisotropy,
            "Energy Relative to Lambda_0 (eV)": (1240.0 / sample_results["lambda_ex"])
            - (1240.0 / sample_results["lambda_0"]),
            "lambda_0": sample_results["lambda_0"],
            "raw_lamp_vector": sample_results["raw_lamp_vector"],
            "par_avg_raw": sample_results["par_avg_raw"],
        }
    )
    print(f"\n--- Results Table for '{sample_label}' ---")
    print(final_df.head(10))
    return final_df


# =========================
# Page-2 helpers (NEW, fixed)
# =========================

def _read_eem_as_columns(file_like, header_rows: int = 2):
    """
    Read EEM CSV where:
      - Column 0            = emission wavelength (nm)
      - Columns 1..N (rows except last) = intensity for each excitation (one column per λ_ex)
      - LAST ROW            = lamp vs excitation (across columns 1..N)

    Returns:
      lambda_em: (n_em,) emission axis (excludes lamp row)
      intens:    (n_em, n_ex) intensities (par or perp), rows=emission, cols=excitation
      lamp:      (n_ex,) lamp vector by excitation
    """
    try:
        file_like.seek(0)
    except Exception:
        pass

    df = pd.read_csv(file_like, header=None)
    if header_rows > 0 and len(df) > header_rows:
        df = df.drop(index=list(range(header_rows))).reset_index(drop=True)

    df = df.apply(pd.to_numeric, errors="coerce")

    lambda_em = df.iloc[:-1, 0].to_numpy(dtype=float)
    intens = df.iloc[:-1, 1:].to_numpy(dtype=float)
    lamp = df.iloc[-1, 1:].to_numpy(dtype=float)

    intens = np.nan_to_num(intens, nan=0.0)
    lamp = np.nan_to_num(lamp, nan=1.0)

    return lambda_em, intens, lamp


def emission_sliced_anisotropy_at_fixed_exc(
    sample_par_path: str,
    sample_perp_path: str,
    dye_par_path: str,   # kept for signature compatibility
    dye_perp_path: str,  # kept for signature compatibility
    lambda_em,
    lambda_ex,
    lambda_0: float,
    lambda_ex_star: float,
    background: float,
    Cj_vector,
    slice_points: int = 15,
    num_slices: int = 15,
) -> pd.DataFrame:
    """
    Slice anisotropy across EMISSION for a fixed excitation λ_ex*.
    CSV layout:
      col0 = λ_em, cols1.. = intensities per excitation, LAST ROW = lamp per excitation.
    """
    # --- normalize all inputs ---
    lambda_ex = np.asarray(lambda_ex, dtype=float)
    Cj_vector = np.asarray(Cj_vector, dtype=float)
    lambda_em = np.asarray(lambda_em, dtype=float)
    lambda_ex_star = float(lambda_ex_star)
    lambda_0 = float(lambda_0)
    slice_points = int(slice_points)
    num_slices = int(num_slices)

    # Load EEMs in "columns" form
    lam_em_par, PAR, L_par = _read_eem_as_columns(sample_par_path)
    lam_em_perp, PERP, L_perp = _read_eem_as_columns(sample_perp_path)

    # Sanity
    if not np.array_equal(lam_em_par, lam_em_perp):
        raise ValueError("Emission axes differ between PAR and PERP files.")
    if lambda_em.size == 0:
        lambda_em = lam_em_par
    if PAR.shape != PERP.shape:
        raise ValueError(f"PAR/PERP shape mismatch: {PAR.shape} vs {PERP.shape}")
    if L_par.shape[0] != PAR.shape[1] or L_perp.shape[0] != PAR.shape[1]:
        raise ValueError("Lamp vectors length must match number of excitation columns.")

    # Choose excitation column j* (INT!)
    j_star = int(np.argmin(np.abs(lambda_ex - lambda_ex_star)))

    # Take emission column at j*
    v_par_raw  = np.asarray(PAR[:, j_star], dtype=float)
    v_perp_raw = np.asarray(PERP[:, j_star], dtype=float)

    # Lamp at that excitation (average par/perp lamp if both valid)
    L_candidates = []
    for Lvec in (L_par, L_perp):
        try:
            L_candidates.append(float(Lvec[int(j_star)]))
        except Exception:
            pass
    if not L_candidates or not np.isfinite(np.mean(L_candidates)) or np.mean(L_candidates) == 0:
        raise ValueError("Invalid lamp value at selected excitation.")
    L_star = float(np.mean(L_candidates))

    # Emission window (points) centered at λ0
    total_pts = slice_points * num_slices
    i0_center = int(np.argmin(np.abs(lambda_em - lambda_0)))
    half_span = total_pts // 2
    i0 = max(0, i0_center - half_span)
    i1 = min(len(lambda_em), i0_center + half_span)

    v_par_win  = v_par_raw[i0:i1]
    v_perp_win = v_perp_raw[i0:i1]
    lamb_win   = lambda_em[i0:i1]

    usable = (len(lamb_win) // slice_points) * slice_points
    if usable == 0:
        raise ValueError("Window too small for the chosen slice width/number of slices.")
    v_par_win  = v_par_win[:usable]
    v_perp_win = v_perp_win[:usable]
    lamb_win   = lamb_win[:usable]
    num_slices_eff = min(num_slices, usable // slice_points)

    # Interpolate C*(λ_ex*) to avoid off-by-one artifacts
    if Cj_vector.shape[0] != lambda_ex.shape[0]:
        raise ValueError("Cj_vector and lambda_ex must have the same length.")
    C_star = float(np.interp(lambda_ex_star, lambda_ex, Cj_vector))

    # Compute anisotropy per slice
    r_list, lamb_center = [], []
    w = slice_points
    for i in range(num_slices_eff):
        a, b = i * w, (i + 1) * w
        par_mean  = float(np.mean(v_par_win[a:b]))
        perp_mean = float(np.mean(v_perp_win[a:b]))

        Ipar_corr  = (par_mean  - float(background)) / L_star
        Iperp_corr = (perp_mean - float(background)) / L_star
        Iperp_corr_star = C_star * Iperp_corr

        denom = Ipar_corr + 2.0 * Iperp_corr_star
        r = np.nan if denom == 0 else (Ipar_corr - Iperp_corr_star) / denom
        r_list.append(r)

        lamb_center.append(float(np.mean(lamb_win[a:b])))

    E_emit = 1240.0 / np.array(lamb_center, dtype=float)

    return pd.DataFrame({
        "Emission Center (nm)": lamb_center,
        "Emission Energy (eV)": E_emit,
        "Anisotropy (slice)": r_list,
        "lambda_ex_star": float(lambda_ex_star),
        "lambda_0": float(lambda_0),
    })
