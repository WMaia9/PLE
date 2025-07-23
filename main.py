# main.py

import tkinter as tk
from tkinter import filedialog
import os
from datetime import datetime
import pandas as pd

from src.gui import launch_parameter_form
from src.data_processing import get_file_pairs, calculate_g_factor_from_dye, process_single_sample
from src.file_handler import save_run_config
from src.plotting import (
    plot_lamp_functions_all,
    plot_raw_vs_lamp_corrected_all,
    plot_sample_corrected_only,
    plot_correction_factor,
    plot_dye_intensity_comparison,
    plot_corrected_intensities_all_samples,
    plot_anisotropy_all_samples,
    plot_anisotropy_individual,
)

def main():
    """Main application workflow."""
    root = tk.Tk()
    root.withdraw()

    # --- 1. Select files and get parameters via GUI ---
    print("--- Starting Anisotropy Analysis ---")
    file_paths = filedialog.askopenfilenames(
        title="Select ALL CSV files (par and perp, including the dye)",
        filetypes=[("CSV Files", "*.csv")]
    )
    if not file_paths:
        print("No files selected. Exiting.")
        return
    
    params = launch_parameter_form(root, list(file_paths))
    root.destroy()
    
    if not params:
        print("Parameter window closed. Exiting.")
        return
    
    # --- 2. Setup run folder ---
    run_dir, plots_dir, data_dir = setup_run_folders()
    if not run_dir:
        return
    save_run_config(run_dir, params, list(file_paths))

    print("\n--- Starting Processing Pipeline ---")
    dye_pair, sample_pairs = get_file_pairs(list(file_paths))
    
    dye_results = calculate_g_factor_from_dye(dye_pair, params)
    
    samples_results_dict = {}
    for label, pair in sample_pairs.items():
        if 'parallel' not in pair or 'perpendicular' not in pair:
            continue
        
        sample_df = process_single_sample(label, pair, dye_results, params)
        samples_results_dict[label] = sample_df

    # Combine all data dictionaries for the plots that need them
    dye_label = list(dye_pair.keys())[0]
    all_files_results = {dye_label: dye_results, **samples_results_dict}
    
    # --- Save Data Files ---
    dye_results.to_csv(os.path.join(data_dir, "g_factor_data.csv"), index=False)
    for label, df in samples_results_dict.items():
        df.to_csv(os.path.join(data_dir, f"{label}_result.csv"), index=False)
        
    print("\n--- Generating and Displaying Diagnostic Plots ---")
    
    print("\n-> Plot 1: Lamp Functions. Close plot window to continue...")
    plot_lamp_functions_all(all_files_results, plots_dir)
    
    print("-> Plot 2: Raw vs. Lamp-Corrected. Close plot window to continue...")
    plot_raw_vs_lamp_corrected_all(all_files_results, plots_dir)

    print("-> Plot 3: Samples Lamp-Corrected. Close plot window to continue...")
    plot_sample_corrected_only(samples_results_dict, plots_dir)

    print("-> Plot 4: G-Factor from Dye. Close plot window to continue...")
    plot_correction_factor(dye_results, plots_dir)

    print("-> Plot 5: Dye Intensity Comparison. Close plot window to continue...")
    plot_dye_intensity_comparison(dye_results, plots_dir)

    print("-> Plot 6: All Samples Corrected Intensities. Close plot window to continue...")
    plot_corrected_intensities_all_samples(samples_results_dict, plots_dir)

    print("-> Plot 7: All Samples Anisotropy. Close plot window to continue...")
    plot_anisotropy_all_samples(samples_results_dict, plots_dir)

    print("-> Plot 8: Individual Sample Anisotropy Plots. Close each window to continue...")
    plot_anisotropy_individual(samples_results_dict, plots_dir)

    print(f"\n✅ Analysis complete! All data and plots have been saved in {run_dir}")

def setup_run_folders():
    root = tk.Tk()
    root.withdraw()
    base_dir = filedialog.askdirectory(parent=root, title="Select a folder to save the run results")
    root.destroy()
    
    if not base_dir:
        return None, None, None
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"anisotropy_run_{timestamp}")
    plots_dir = os.path.join(run_dir, "plots")
    data_dir = os.path.join(run_dir, "processed_data")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    print(f"Results will be saved in: {run_dir}")
    return run_dir, plots_dir, data_dir
    
if __name__ == "__main__":
    main()