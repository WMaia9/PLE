# src/plotting.py
"""
This module contains all plotting functions for the PLE Anisotropy Streamlit app.

It is divided into two main sections:
1. Matplotlib Functions: Used for generating static PNG images for the downloadable
   ZIP archive. These functions typically accept a `save_dir` argument.
2. Plotly Functions: Used for creating interactive visualizations displayed
   directly in the Streamlit web application. These functions return Plotly
   Figure objects.
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from typing import Dict, List, Any, Tuple
import plotly.graph_objects as go
import plotly.express as px

# ==============================================================================
# Matplotlib Functions (for Static Image Generation)
# ==============================================================================

def plot_lamp_functions_all(all_data: Dict[str, Dict[str, Any]], save_dir: str = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the raw lamp intensity vs. excitation wavelength for all files.

    Args:
        all_data: A dictionary where keys are sample labels and values are
                  dictionaries containing the processed data, including
                  'Excitation Wavelength (nm)' and 'raw_lamp_vector'.
        save_dir: The directory path to save the plot image. If None,
                  the plot is not saved to a file.

    Returns:
        A tuple containing the Matplotlib figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, data_dict in all_data.items():
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

def plot_raw_vs_lamp_corrected_all(all_data: Dict[str, Dict[str, Any]], save_dir: str = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Compares raw vs. lamp-corrected average intensities for all files.

    Args:
        all_data: A dictionary containing processed data for all samples,
                  including 'Excitation Wavelength (nm)', 'par_avg_raw',
                  and 'Par Avg'.
        save_dir: The directory path to save the plot image.

    Returns:
        A tuple containing the Matplotlib figure and axes objects.
    """
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

def plot_sample_corrected_only(samples_data: Dict[str, pd.DataFrame], save_dir: str = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the final lamp-corrected average intensities for only the samples.

    Args:
        samples_data: A dictionary where keys are sample labels and values
                      are DataFrames with the final results.
        save_dir: The directory path to save the plot image.

    Returns:
        A tuple containing the Matplotlib figure and axes objects.
    """
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

def plot_correction_factor(dye_data: pd.DataFrame, save_dir: str = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the G-Factor (Cj) correction curve from the dye data.

    Args:
        dye_data: DataFrame containing the G-factor calculation results,
                  including 'Excitation Wavelength (nm)' and 'Cj (Par / Perp)'.
        save_dir: The directory path to save the plot image.

    Returns:
        A tuple containing the Matplotlib figure and axes objects.
    """
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

def plot_dye_intensity_comparison(dye_data: pd.DataFrame, save_dir: str = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Compares parallel, perpendicular, and G-factor corrected perpendicular
    intensities for the dye.

    Args:
        dye_data: DataFrame with the dye's processed intensity data.
        save_dir: The directory path to save the plot image.

    Returns:
        A tuple containing the Matplotlib figure and axes objects.
    """
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

def plot_corrected_intensities_all_samples(samples_data: Dict[str, pd.DataFrame], save_dir: str = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the final corrected parallel and perpendicular intensities for all samples.

    Args:
        samples_data: Dictionary of result DataFrames for each sample.
        save_dir: The directory path to save the plot image.

    Returns:
        A tuple containing the Matplotlib figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    for label, df in samples_data.items():
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

def plot_anisotropy_all_samples(samples_data: Dict[str, pd.DataFrame], save_dir: str = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the calculated anisotropy vs. relative energy for all samples on one graph.

    Args:
        samples_data: Dictionary of result DataFrames for each sample.
        save_dir: The directory path to save the plot image.

    Returns:
        A tuple containing the Matplotlib figure and axes objects.
    """
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

def plot_anisotropy_individual(samples_data: Dict[str, pd.DataFrame], save_dir: str = None) -> List[plt.Figure]:
    """
    Creates a separate anisotropy vs. relative energy plot for each sample.

    Args:
        samples_data: Dictionary of result DataFrames for each sample.
        save_dir: The directory path to save the plot images.

    Returns:
        A list of Matplotlib figure objects, one for each sample.
    """
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

def plot_correction_factor_smoothed(df: pd.DataFrame, save_dir: str = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots original and smoothed Cj (correction factor) curves.

    Args:
        df: DataFrame with 'Cj (Par / Perp)' and 'Cj Smoothed' columns.
        save_dir: The directory path to save the plot image.

    Returns:
        A tuple containing the Matplotlib figure and axes objects.
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

def plot_emission_sliced_anisotropy(df: pd.DataFrame, save_dir: str = None, sample_label: str = "", xaxis: str = "lambda") -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots anisotropy vs. emission for a single excitation wavelength slice.

    Args:
        df: DataFrame containing the anisotropy slice data.
        save_dir: The directory path to save the plot image.
        sample_label: The name of the sample being plotted.
        xaxis: The unit for the x-axis, either 'lambda' or 'energy'.

    Returns:
        A tuple containing the Matplotlib figure and axes objects.
    """
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

def plot_emission_overlay_scatter(per_ex_results: Dict[float, pd.DataFrame], sample_label: str = "", xaxis: str = "lambda", xlim_nm: Tuple[float, float] = None, ylim: Tuple[float, float] = (-0.025, 0.025)) -> Tuple[plt.Figure, plt.Axes]:
    """
    Creates an overlay scatter plot of anisotropy vs. emission for multiple
    excitation wavelengths.

    Args:
        per_ex_results: A dictionary where keys are excitation wavelengths
                        and values are the corresponding slice DataFrames.
        sample_label: The name of the sample being plotted.
        xaxis: The unit for the x-axis, either 'lambda' or 'energy'.
        xlim_nm: A tuple defining the x-axis limits in nanometers.
        ylim: A tuple defining the y-axis limits.

    Returns:
        A tuple containing the Matplotlib figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
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


# ==============================================================================
# Plotly Functions (for Interactive Streamlit Display)
# ==============================================================================

def plot_lamp_functions_all_plotly(all_data: Dict[str, Dict[str, Any]]) -> go.Figure:
    """
    Creates an interactive plot of raw lamp intensity vs. excitation wavelength.

    Args:
        all_data: A dictionary containing the processed data for all samples.

    Returns:
        A Plotly Figure object for interactive display.
    """
    fig = go.Figure()
    for label, data_dict in all_data.items():
        fig.add_trace(go.Scatter(
            x=data_dict["Excitation Wavelength (nm)"],
            y=data_dict["raw_lamp_vector"],
            mode='lines',
            marker=dict(size=4),
            name=label
        ))
    fig.update_layout(
        title="Lamp Functions (Raw)",
        xaxis_title="Excitation Wavelength (nm)",
        yaxis_title="Lamp Intensity (a.u.)",
        legend_title="Samples",
        width=800
    )
    return fig

def plot_raw_vs_lamp_corrected_all_plotly(all_data: Dict[str, Dict[str, Any]]) -> go.Figure:
    """
    Creates an interactive plot comparing raw vs. lamp-corrected intensities.

    Args:
        all_data: A dictionary containing the processed data for all samples.

    Returns:
        A Plotly Figure object for interactive display.
    """
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, (label, data_dict) in enumerate(all_data.items()):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=data_dict["Excitation Wavelength (nm)"],
            y=data_dict["par_avg_raw"],
            mode='lines',
            marker=dict(size=4, color=color),
            name=f"{label} (Raw Avg)",
            line=dict(dash='dash', color=color)
        ))
        fig.add_trace(go.Scatter(
            x=data_dict["Excitation Wavelength (nm)"],
            y=data_dict["Par Avg"],
            mode='lines',
            marker=dict(size=4, color=color),
            name=f"{label} (Lamp-Corrected)",
            line=dict(color=color)
        ))
    fig.update_layout(
        title="Raw vs. Lamp-Corrected Average Intensities (All Files)",
        xaxis_title="Excitation Wavelength (nm)",
        yaxis_title="Average Intensity (a.u.)",
        legend_title="Files",
        width=800
    )
    return fig

def plot_sample_corrected_only_plotly(samples_data: Dict[str, pd.DataFrame]) -> go.Figure:
    """
    Creates an interactive plot of lamp-corrected intensities for samples only.

    Args:
        samples_data: A dictionary of result DataFrames for each sample.

    Returns:
        A Plotly Figure object for interactive display.
    """
    fig = go.Figure()
    for label, df in samples_data.items():
        fig.add_trace(go.Scatter(
            x=df["Excitation Wavelength (nm)"],
            y=df["Par Avg"],
            mode='lines',
            marker=dict(size=4),
            name=label
        ))
    fig.update_layout(
        title="Lamp-Corrected Average Intensities (Samples Only)",
        xaxis_title="Excitation Wavelength (nm)",
        yaxis_title="Average Intensity (a.u.)",
        legend_title="Samples",
        width=800
    )
    return fig

def plot_correction_factor_plotly(dye_data: pd.DataFrame) -> go.Figure:
    """
    Creates an interactive plot of the G-Factor (Cj) correction curve.

    Args:
        dye_data: DataFrame containing the G-factor calculation results.

    Returns:
        A Plotly Figure object for interactive display.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dye_data["Excitation Wavelength (nm)"],
        y=dye_data["Cj (Par / Perp)"],
        mode='lines',
        marker=dict(size=4),
        name='G-Factor'
    ))
    fig.update_layout(
        title="Correction Factor (G-Factor or Cj)",
        xaxis_title="Excitation Wavelength (nm)",
        yaxis_title="G-Factor (I∥ / I⊥)",
        width=800
    )
    return fig

def plot_dye_intensity_comparison_plotly(dye_data: pd.DataFrame) -> go.Figure:
    """
    Creates an interactive plot comparing intensities for the dye.

    Args:
        dye_data: DataFrame with the dye's processed intensity data.

    Returns:
        A Plotly Figure object for interactive display.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dye_data["Excitation Wavelength (nm)"],
        y=dye_data["Par Avg"],
        mode='lines',
        marker=dict(size=4),
        name="Dye Parallel I∥ (Lamp-Corrected)"
    ))
    fig.add_trace(go.Scatter(
        x=dye_data["Excitation Wavelength (nm)"],
        y=dye_data["Perp Avg"],
        mode='lines',
        marker=dict(size=4),
        name="Dye Perpendicular I⊥ (Lamp-Corrected)",
        line=dict(dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=dye_data["Excitation Wavelength (nm)"],
        y=dye_data["perp_avg_g_corr"],
        mode='lines',
        marker=dict(size=4),
        name="Dye Perpendicular I⊥ (G-Factor Corrected)",
        line=dict(dash='dot')
    ))
    fig.update_layout(
        title="Dye Intensity Comparison",
        xaxis_title="Excitation Wavelength (nm)",
        yaxis_title="Average Intensity (a.u.)",
        legend_title="Intensity",
        width=800
    )
    return fig

def plot_corrected_intensities_all_samples_plotly(samples_data: Dict[str, pd.DataFrame]) -> go.Figure:
    """
    Creates an interactive plot of corrected intensities for all samples.

    Args:
        samples_data: Dictionary of result DataFrames for each sample.

    Returns:
        A Plotly Figure object for interactive display.
    """
    fig = go.Figure()
    for label, df in samples_data.items():
        fig.add_trace(go.Scatter(
            x=df["Excitation Wavelength (nm)"],
            y=df["Par Avg"],
            mode='lines',
            marker=dict(size=4),
            name=f"{label} (Parallel I∥)"
        ))
        fig.add_trace(go.Scatter(
            x=df["Excitation Wavelength (nm)"],
            y=df["Perp Avg"],
            mode='lines',
            marker=dict(size=4),
            name=f"{label} (Perpendicular I⊥)",
            line=dict(dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=df["Excitation Wavelength (nm)"],
            y=df["Perp Corrected"],
            mode='lines',
            marker=dict(size=4),
            name=f"{label} (Corrected I⊥)",
            line=dict(dash='dot')
        ))
    fig.update_layout(
        title="Corrected Intensities (All Samples)",
        xaxis_title="Excitation Wavelength (nm)",
        yaxis_title="Average Intensity (a.u.)",
        legend_title="Samples",
        width=800
    )
    return fig

def plot_anisotropy_all_samples_plotly(samples_data: Dict[str, pd.DataFrame]) -> go.Figure:
    """
    Creates an interactive plot of anisotropy for all samples.

    Args:
        samples_data: Dictionary of result DataFrames for each sample.

    Returns:
        A Plotly Figure object for interactive display.
    """
    fig = go.Figure()
    for label, df in samples_data.items():
        fig.add_trace(go.Scatter(
            x=df["Energy Relative to Lambda_0 (eV)"],
            y=df["Anisotropy"],
            mode='lines',
            marker=dict(size=4),
            name=label
        ))
    fig.update_layout(
        title="Anisotropy vs. Relative Energy (All Samples)",
        xaxis_title="Relative Energy (ΔE) (eV)",
        yaxis_title="Anisotropy (r)",
        legend_title="Samples",
        width=800
    )
    return fig

def plot_anisotropy_individual_plotly(samples_data: Dict[str, pd.DataFrame]) -> List[go.Figure]:
    """
    Creates a list of individual interactive anisotropy plots for each sample.

    Args:
        samples_data: Dictionary of result DataFrames for each sample.

    Returns:
        A list of Plotly Figure objects.
    """
    figs = []
    for label, df in samples_data.items():
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Energy Relative to Lambda_0 (eV)"],
            y=df["Anisotropy"],
            mode='lines',
            marker=dict(size=4),
            name=f"Anisotropy — {label}"
        ))
        fig.update_layout(
            title=f"Anisotropy vs. Relative Energy - {label}",
            xaxis_title="Relative Energy (ΔE) (eV)",
            yaxis_title="Anisotropy (r)",
            width=800
        )
        figs.append(fig)
    return figs

def plot_correction_factor_smoothed_plotly(df: pd.DataFrame) -> go.Figure:
    """
    Creates an interactive plot comparing original and smoothed Cj curves.

    Args:
        df: DataFrame with 'Cj (Par / Perp)' and 'Cj Smoothed' columns.

    Returns:
        A Plotly Figure object for interactive display.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Excitation Wavelength (nm)"],
        y=df["Cj (Par / Perp)"],
        mode='lines',
        marker=dict(size=4),
        name="Original Cj",
        line=dict(dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=df["Excitation Wavelength (nm)"],
        y=df["Cj Smoothed"],
        mode='lines',
        marker=dict(size=4),
        name="Smoothed Cj"
    ))
    fig.update_layout(
        title="Smoothed Correction Factor (Savitzky-Golay)",
        xaxis_title="Excitation Wavelength (nm)",
        yaxis_title="G-Factor (Cj)",
        legend_title="Correction Factor",
        width=800
    )
    return fig

def plot_emission_sliced_anisotropy_plotly(df: pd.DataFrame, sample_label: str = "", xaxis: str = "lambda") -> go.Figure:
    """
    Creates an interactive plot of anisotropy vs. emission for a single slice.

    Args:
        df: DataFrame containing the anisotropy slice data.
        sample_label: The name of the sample being plotted.
        xaxis: The unit for the x-axis, either 'lambda' or 'energy'.

    Returns:
        A Plotly Figure object for interactive display.
    """
    if xaxis == "energy":
        x_col, xlabel = "Emission Energy (eV)", "Emission Energy (eV)"
    else:
        x_col, xlabel = "Emission Center (nm)", "Emission Wavelength (nm)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df["Anisotropy (slice)"],
        mode="lines",
        marker=dict(size=5),
        name="Anisotropy"
    ))
    lam_ex = df["lambda_ex_star"].iloc[0]
    fig.update_layout(
        title=f"Emission-Sliced Anisotropy at λ_ex = {lam_ex:.0f} nm — {sample_label}",
        xaxis_title=xlabel,
        yaxis_title="Anisotropy (r)",
        width=800
    )
    return fig

def plot_emission_overlay_scatter_plotly(per_ex_results: Dict[float, pd.DataFrame], sample_label: str = "", xaxis: str = "lambda", xlim_nm: Tuple[float, float] = None, ylim: Tuple[float, float] = (-0.025, 0.025)) -> go.Figure:
    """
    Creates an interactive overlay plot of anisotropy vs. emission for
    multiple excitation wavelengths.

    Args:
        per_ex_results: A dictionary where keys are excitation wavelengths
                        and values are the corresponding slice DataFrames.
        sample_label: The name of the sample being plotted.
        xaxis: The unit for the x-axis, either 'lambda' or 'energy'.
        xlim_nm: A tuple defining the x-axis limits in nanometers.
        ylim: A tuple defining the y-axis limits.

    Returns:
        A Plotly Figure object for interactive display.
    """
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    if xaxis == "energy":
        x_col, xlabel, x_range = "Emission Energy (eV)", "Emission Energy (eV)", None
    else:
        x_col, xlabel, x_range = "Emission Center (nm)", "Emission Wavelength (nm)", xlim_nm

    for i, lam_ex_star in enumerate(sorted(per_ex_results.keys())):
        df = per_ex_results[lam_ex_star]
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df["Anisotropy (slice)"],
            mode="lines",
            marker=dict(size=5, color=color),
            name=f"{int(lam_ex_star)} nm"
        ))
    fig.update_layout(
        title=f"Emission-sliced anisotropy — {sample_label}",
        xaxis_title=xlabel,
        yaxis_title="Anisotropy (r)",
        legend_title="λ_ex",
        width=800,
        xaxis_range=x_range,
        yaxis_range=ylim
    )
    return fig