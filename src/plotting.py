import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_all_anisotropy(df: pd.DataFrame, save_dir: str):
    plt.figure(figsize=(10, 6))
    for sample, group in df.groupby('Sample'):
        plt.plot(group['Relative Energy (eV)'], group['Anisotropy'], label=sample, marker='o', linestyle='-', markersize=4)
    plt.title('Anisotropy vs. Relative Energy for All Samples')
    plt.xlabel('Relative Energy ($E_{ex} - E_{em}$) (eV)')
    plt.ylabel('Anisotropy (r)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'anisotropy_all_samples.png'))
    plt.close()

def plot_corrected_intensities(df: pd.DataFrame, sample_name: str, save_dir: str):
    plt.figure(figsize=(10, 6))
    plt.plot(df["Excitation Wavelength (nm)"], df["I_parallel_avg"], label="Parallel $I_\\parallel$")
    plt.plot(df["Excitation Wavelength (nm)"], df["I_perpendicular_avg"], label="Perpendicular $I_\\perp$", linestyle='--')
    plt.plot(df["Excitation Wavelength (nm)"], df["I_perp_corrected"], label="$G \\times I_\\perp$ (Corrected)", linestyle='-.')
    plt.title(f'Corrected Intensities - {sample_name}')
    plt.xlabel('Excitation Wavelength (nm)')
    plt.ylabel('Average Intensity (a.u.)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'intensities_{sample_name}.png'))
    plt.close()

def plot_g_factor(df: pd.DataFrame, save_dir: str):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Excitation Wavelength (nm)'], df['G_Factor'], label='G-Factor')
    plt.title('Instrumental G-Factor vs. Excitation Wavelength')
    plt.xlabel('Excitation Wavelength (nm)')
    plt.ylabel('G-Factor ($I_\\parallel / I_\\perp$)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'g_factor_plot.png'))
    plt.close()