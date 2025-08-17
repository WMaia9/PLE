# src/plotting.py

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from typing import Dict, List

def plot_lamp_functions_all(all_data: dict, save_dir: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, data_dict in all_data.items():
        # Convert to numpy array if it's a list from a dict conversion
        lambda_ex = np.array(data_dict["Excitation Wavelength (nm)"])
        raw_lamp = np.array(data_dict["raw_lamp_vector"])
        ax.plot(lambda_ex, raw_lamp, label=label, alpha=0.8)
    ax.set_title("Lamp Functions (Raw)")
    ax.set_xlabel("Excitation Wavelength (nm)")
    ax.set_ylabel("Lamp Intensity (a.u.)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, '1_lamp_functions_all.png'))
    return fig, ax

def plot_raw_vs_lamp_corrected_all(all_data: dict, save_dir: str = None):
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_data)))
    for i, (label, data_dict) in enumerate(all_data.items()):
        lambda_ex = np.array(data_dict["Excitation Wavelength (nm)"])
        par_avg_raw = np.array(data_dict["par_avg_raw"])
        par_avg_lamp_corr = np.array(data_dict["Par Avg"])
        ax.plot(lambda_ex, par_avg_raw, label=f"{label} (Raw Avg)", linestyle='--', color=colors[i])
        ax.plot(lambda_ex, par_avg_lamp_corr, label=f"{label} (Lamp-Corrected)", linestyle='-', color=colors[i])
    ax.set_title("Raw vs. Lamp-Corrected Average Intensities (All Files)")
    ax.set_xlabel("Excitation Wavelength (nm)")
    ax.set_ylabel("Average Intensity (a.u.)")
    ax.legend(fontsize='small', ncol=2)
    ax.grid(True)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, '2_raw_vs_lamp_corrected.png'))
    return fig, ax

def plot_sample_corrected_only(samples_data: dict, save_dir: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, df in samples_data.items():
        ax.plot(df["Excitation Wavelength (nm)"], df["Par Avg"], label=label, alpha=0.8)
    ax.set_title("Lamp-Corrected Average Intensities (Samples Only)")
    ax.set_xlabel("Excitation Wavelength (nm)")
    ax.set_ylabel("Average Intensity (a.u.)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, '3_sample_corrected_only.png'))
    return fig, ax

def plot_correction_factor(dye_data: pd.DataFrame, save_dir: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dye_data["Excitation Wavelength (nm)"], dye_data["Cj (Par / Perp)"], linewidth=2)
    ax.set_title("Correction Factor (G-Factor or Cj)")
    ax.set_xlabel("Excitation Wavelength (nm)")
    ax.set_ylabel("G-Factor (I∥ / I⊥)")
    ax.grid(True)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, '4_correction_factor.png'))
    return fig, ax
    
def plot_dye_intensity_comparison(dye_data: pd.DataFrame, save_dir: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dye_data["Excitation Wavelength (nm)"], dye_data["Par Avg"], label="Dye Parallel I∥ (Lamp-Corrected)")
    ax.plot(dye_data["Excitation Wavelength (nm)"], dye_data["Perp Avg"], label="Dye Perpendicular I⊥ (Lamp-Corrected)", linestyle='--')
    ax.plot(dye_data["Excitation Wavelength (nm)"], dye_data["perp_avg_g_corr"], label="Dye Perpendicular I⊥ (G-Factor Corrected)", linestyle='-.')
    ax.set_title("Dye Intensity Comparison")
    ax.set_xlabel("Excitation Wavelength (nm)")
    ax.set_ylabel("Average Intensity (a.u.)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, '5_dye_intensity_comparison.png'))
    return fig, ax

def plot_corrected_intensities_all_samples(samples_data: dict, save_dir: str = None):
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(samples_data)))
    for i, (label, df) in enumerate(samples_data.items()):
        ax.plot(df["Excitation Wavelength (nm)"], df["Par Avg"], label=f"{label} (Parallel I∥)", linestyle='-')
        ax.plot(df["Excitation Wavelength (nm)"], df["Perp Avg"], label=f"{label} (Perpendicular I⊥)", linestyle='--')
        ax.plot(df["Excitation Wavelength (nm)"], df["Perp Corrected"], label=f"{label} (Corrected I⊥)", linestyle='--')
    ax.set_title("Corrected Intensities (All Samples)")
    ax.set_xlabel("Excitation Wavelength (nm)")
    ax.set_ylabel("Average Intensity (a.u.)")
    ax.legend(fontsize='small', ncol=2)
    ax.grid(True)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, '6_corrected_intensities_all_samples.png'))
    return fig, ax

def plot_anisotropy_all_samples(samples_data: dict, save_dir: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, df in samples_data.items():
        ax.plot(df["Energy Relative to Lambda_0 (eV)"], df["Anisotropy"], label=label, marker='o', linestyle='-', markersize=4)
    ax.set_title("Anisotropy vs. Relative Energy (All Samples)")
    ax.set_xlabel("Relative Energy (ΔE) (eV)")
    ax.set_ylabel("Anisotropy (r)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, '7_anisotropy_all_samples.png'))
    return fig, ax

def plot_anisotropy_individual(samples_data: dict, save_dir: str = None):
    figs = []
    for label, df in samples_data.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df["Energy Relative to Lambda_0 (eV)"], df["Anisotropy"], linestyle='-', label=f"Anisotropy — {label}")
        ax.set_title(f"Anisotropy vs. Relative Energy - {label}")
        ax.set_xlabel("Relative Energy (ΔE) (eV)")
        ax.set_ylabel("Anisotropy (r)")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'8_anisotropy_individual_{label}.png'))
        figs.append(fig)
    return figs

def plot_emission_sliced_anisotropy(df: pd.DataFrame, save_dir: str | None = None, sample_label: str = "", xaxis: str = "lambda"):
    if xaxis == "energy":
        x = df["Emission Energy (eV)"]
        xlabel = "Emission Energy (eV)"
        suffix = "E"
    else:
        x = df["Emission Center (nm)"]
        xlabel = "Emission Wavelength (nm)"
        suffix = "nm"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, df["Anisotropy (slice)"], marker="o", linestyle="-")
    lam_ex = df["lambda_ex_star"].iloc[0]
    ax.set_title(f"Emission-Sliced Anisotropy at λ_ex = {lam_ex:.0f} nm — {sample_label}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Anisotropy (r)")
    ax.grid(True)
    fig.tight_layout()

    if save_dir:
        fname = f"emission_sliced_anisotropy_{sample_label}_{int(lam_ex)}nm_{suffix}.png"
        fig.savefig(os.path.join(save_dir, fname))

    return fig, ax

def plot_emission_overlay_scatter(
    per_ex_results: dict,                 # {lam_ex_star_nm: df_slice}
    sample_label: str = "",
    xaxis: str = "lambda",               # "lambda" or "energy"
    xlim_nm: tuple | None = (565, 600),  # tighten to your band; set None to auto
    ylim: tuple = (-0.025, 0.025),       # match reference scale
):
    fig, ax = plt.subplots(figsize=(8, 5))
    # colors & markers like the reference (filled/hollow alternating)
    colors  = ["tab:red", "tab:green", "tab:blue", "tab:orange", "tab:purple", "tab:brown"]
    markers = ["s", "o", "*", "D", "P", "X", "^", "v", "<", ">"]

    for i, lam_ex_star in enumerate(sorted(per_ex_results.keys())):
        df = per_ex_results[lam_ex_star]
        if xaxis == "energy":
            x = df["Emission Energy (eV)"]
            xlabel = "emission energy (eV)"
            xlim = None
        else:
            x = df["Emission Center (nm)"]
            xlabel = "emission wavelength (nm)"
            xlim = xlim_nm

        y = df["Anisotropy (slice)"]

        c = colors[i % len(colors)]
        m = markers[i % len(markers)]

        # plot filled + hollow to mimic reference variety
        ax.scatter(x, y, s=28, c=c, marker=m, linewidths=0, alpha=0.9, label=f"{int(lam_ex_star)} nm")
        ax.scatter(x, y, s=28, facecolors="none", edgecolors=c, marker=m, linewidths=1.2, alpha=0.9)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("anisotropy (r)")
    if ylim: ax.set_ylim(*ylim)
    if xlim: ax.set_xlim(*xlim)
    ax.grid(True, alpha=0.25)
    ax.legend(title="λ_ex", ncols=3, frameon=False, loc="upper right")
    ax.set_title(f"Emission-sliced anisotropy — {sample_label}")
    fig.tight_layout()
    return fig, ax

def plot_correction_factor_smoothed(df: pd.DataFrame, save_dir: str = None):
    """
    Plots original and smoothed Cj (correction factor) curves.

    Parameters:
        df (pd.DataFrame): DataFrame with 'Cj (Par / Perp)' and 'Cj Smoothed' columns
        save_dir (str, optional): Path to save PNG image (if provided)

    Returns:
        fig, ax: Matplotlib figure and axis
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["Excitation Wavelength (nm)"], df["Cj (Par / Perp)"], linestyle="--", label="Original Cj", alpha=0.6)
    ax.plot(df["Excitation Wavelength (nm)"], df["Cj Smoothed"], linewidth=2, label="Smoothed Cj")
    ax.set_title("Smoothed Correction Factor (Savitzky-Golay)")
    ax.set_xlabel("Excitation Wavelength (nm)")
    ax.set_ylabel("G-Factor (Cj)")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, "correction_factor_smoothed.png"))

    return fig, ax