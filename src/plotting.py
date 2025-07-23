import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_and_save(df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str, save_path: str):
    """Generic function to create and save a line plot."""
    plt.figure(figsize=(10, 6))
    plt.plot(df[x_col], df[y_col], linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    # Ensure the target directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_corrected_intensities(df: pd.DataFrame, save_path: str):
    """Plots the raw and corrected emission intensities."""
    plt.figure(figsize=(10, 6))
    plt.plot(df["Excitation Wavelength (nm)"], df["Par Avg"], label="Parallel $I_\\parallel$")
    plt.plot(df["Excitation Wavelength (nm)"], df["Perp Avg"], label="Perpendicular $I_\\perp$", linestyle='--')
    plt.plot(df["Excitation Wavelength (nm)"], df["Perp Corrected"], label="$G \\times I_\\perp$ (Corrected)", linestyle='-.')
    plt.title("Average Emission Intensity vs. Excitation Wavelength")
    plt.xlabel("Excitation Wavelength (nm)")
    plt.ylabel("Average Intensity (a.u.)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()