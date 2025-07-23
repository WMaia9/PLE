import tkinter as tk
from tkinter import filedialog
import os
from datetime import datetime
import pandas as pd

from src.gui import launch_parameter_form
from src.data_processing import process_anisotropy_run
from src.file_handler import save_run_config
from src.plotting import plot_all_anisotropy, plot_corrected_intensities, plot_g_factor

def main():
    """Main application workflow."""
    root = tk.Tk()
    root.withdraw()

    # --- 1. Select Input Files ---
    file_paths = filedialog.askopenfilenames(
        title="Select ALL Parallel and Perpendicular CSV Files (including dye)",
        filetypes=[("CSV Files", "*.csv")]
    )
    if not file_paths:
        print("No files selected. Exiting.")
        return

    # --- 2. Get Parameters from GUI ---
    params = launch_parameter_form(list(file_paths))
    if not params:
        print("Parameter window closed. Exiting.")
        return
    
    # --- 3. Create a Run Folder ---
    base_dir = filedialog.askdirectory(title="Select a Folder to Save the Run Output")
    if not base_dir:
        print("No output directory selected. Exiting.")
        return
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"anisotropy_run_{timestamp}")
    plots_dir = os.path.join(run_dir, "plots")
    data_dir = os.path.join(run_dir, "processed_data")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    print(f"Output will be saved in: {run_dir}")

    # --- 4. Save Configuration ---
    save_run_config(run_dir, params, list(file_paths))
    
    # --- 5. Run the Main Processing Pipeline ---
    try:
        results_dict, g_factor_df = process_anisotropy_run(list(file_paths), params)
    except Exception as e:
        print(f"\n--- An error occurred during processing ---")
        print(f"Error: {e}")
        return

    # --- 6. Save and Plot Results ---
    all_results_df = pd.DataFrame()
    for sample_name, df in results_dict.items():
        csv_path = os.path.join(data_dir, f"{sample_name}_result.csv")
        df.to_csv(csv_path, index=False)
        
        df_to_append = df[["Relative Energy (eV)", "Anisotropy"]].copy()
        df_to_append['Sample'] = sample_name
        all_results_df = pd.concat([all_results_df, df_to_append])
        
        plot_corrected_intensities(df, sample_name, plots_dir)

    combined_csv_path = os.path.join(data_dir, "ALL_SAMPLES_anisotropy.csv")
    all_results_df.to_csv(combined_csv_path, index=False)
    
    plot_all_anisotropy(all_results_df, plots_dir)
    plot_g_factor(g_factor_df, plots_dir)
    
    print(f"\n✅ Analysis complete! All data and plots saved successfully to {run_dir}")
    
if __name__ == "__main__":
    main()