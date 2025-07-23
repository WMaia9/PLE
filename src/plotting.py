# src/plotting.py

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def plot_lamp_functions_all(all_data: dict, save_dir: str):
    plt.figure(figsize=(10, 6))
    for label, data in all_data.items():
        plt.plot(data["Excitation Wavelength (nm)"], data["raw_lamp_vector"], label=label, alpha=0.8)
    plt.title("Lamp Functions (Raw)")
    plt.xlabel("Excitation Wavelength (nm)")
    plt.ylabel("Lamp Intensity (a.u.)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '1_lamp_functions_all.png'))
    plt.show()

def plot_raw_vs_lamp_corrected_all(all_data: dict, save_dir: str):
    plt.figure(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_data)))
    for i, (label, data) in enumerate(all_data.items()):
        plt.plot(data["Excitation Wavelength (nm)"], data["par_avg_raw"], label=f"{label} (Raw Avg)", linestyle='--', color=colors[i])
        plt.plot(data["Excitation Wavelength (nm)"], data["Par Avg"], label=f"{label} (Lamp-Corrected)", linestyle='-', color=colors[i])
    plt.title("Raw vs. Lamp-Corrected Average Intensities (All Files)")
    plt.xlabel("Excitation Wavelength (nm)")
    plt.ylabel("Average Intensity (a.u.)")
    plt.legend(fontsize='small', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '2_raw_vs_lamp_corrected.png'))
    plt.show()

def plot_sample_corrected_only(samples_data: dict, save_dir: str):
    plt.figure(figsize=(10, 6))
    for label, data in samples_data.items():
        plt.plot(data["Excitation Wavelength (nm)"], data["Par Avg"], label=label, alpha=0.8)
    plt.title("Lamp-Corrected Average Intensities (Samples Only)")
    plt.xlabel("Excitation Wavelength (nm)")
    plt.ylabel("Average Intensity (a.u.)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '3_sample_corrected_only.png'))
    plt.show()

def plot_correction_factor(dye_data: pd.DataFrame, save_dir: str):
    plt.figure(figsize=(10, 6))
    plt.plot(dye_data["Excitation Wavelength (nm)"], dye_data["Cj (Par / Perp)"], linewidth=2)
    plt.title("Correction Factor (G-Factor or Cj)")
    plt.xlabel("Excitation Wavelength (nm)")
    plt.ylabel("G-Factor (I∥ / I⊥)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '4_correction_factor.png'))
    plt.show()
    
def plot_dye_intensity_comparison(dye_data: pd.DataFrame, save_dir: str):
    plt.figure(figsize=(10, 6))
    plt.plot(dye_data["Excitation Wavelength (nm)"], dye_data["Par Avg"], label="Dye Parallel I∥ (Lamp-Corrected)")
    plt.plot(dye_data["Excitation Wavelength (nm)"], dye_data["Perp Avg"], label="Dye Perpendicular I⊥ (Lamp-Corrected)", linestyle='--')
    plt.plot(dye_data["Excitation Wavelength (nm)"], dye_data["perp_avg_g_corr"], label="Dye Perpendicular I⊥ (G-Factor Corrected)", linestyle='-.')
    plt.title("Dye Intensity Comparison")
    plt.xlabel("Excitation Wavelength (nm)")
    plt.ylabel("Average Intensity (a.u.)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '5_dye_intensity_comparison.png'))
    plt.show()

def plot_corrected_intensities_all_samples(samples_data: dict, save_dir: str):
    plt.figure(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(samples_data)))
    for i, (label, data) in enumerate(samples_data.items()):
        plt.plot(data["Excitation Wavelength (nm)"], data["Par Avg"], label=f"{label} (Parallel I∥)", linestyle='-')
        plt.plot(data["Excitation Wavelength (nm)"], data["Perp Avg"], label=f"{label} (Perpendicular I⊥)", linestyle='--')
        plt.plot(data["Excitation Wavelength (nm)"], data["Perp Corrected"], label=f"{label} (Corrected I⊥)", linestyle='--')
    plt.title("Corrected Intensities (All Samples)")
    plt.xlabel("Excitation Wavelength (nm)")
    plt.ylabel("Average Intensity (a.u.)")
    plt.legend(fontsize='small', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '6_corrected_intensities_all_samples.png'))
    plt.show()

def plot_anisotropy_all_samples(samples_data: dict, save_dir: str):
    plt.figure(figsize=(10, 6))
    for label, data in samples_data.items():
        plt.plot(data["Energy Relative to Lambda_0 (eV)"], data["Anisotropy"], label=label, marker='o', linestyle='-', markersize=4)
    plt.title("Anisotropy vs. Relative Energy (All Samples)")
    plt.xlabel("Relative Energy (ΔE) (eV)")
    plt.ylabel("Anisotropy (r)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '7_anisotropy_all_samples.png'))
    plt.show()

def plot_anisotropy_individual(samples_data: dict, save_dir: str):
    for label, data in samples_data.items():
        plt.figure(figsize=(8, 5))
        plt.plot(data["Energy Relative to Lambda_0 (eV)"], data["Anisotropy"], linestyle='-', label=f"Anisotropy — {label}")
        plt.title(f"Anisotropy vs. Relative Energy - {label}")
        plt.xlabel("Relative Energy (ΔE) (eV)")
        plt.ylabel("Anisotropy (r)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'8_anisotropy_individual_{label}.png'))
        plt.show()