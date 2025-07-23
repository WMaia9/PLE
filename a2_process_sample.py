import pandas as pd
import os
from src.file_handler import load_config, load_and_clean_eem_csv
from src.data_processing import (
    process_raw_to_avg_intensity, 
    calculate_anisotropy, 
    calculate_relative_energy
)
from src.plotting import plot_and_save, plot_corrected_intensities

def run_sample_analysis():
    """
    Executes the pipeline to calculate the anisotropy of a sample,
    using the pre-calculated G-Factor.
    """
    # 1. Load configuration
    config = load_config()
    params = config['processing_parameters']
    sample = config['sample_files']
    paths = config['output_paths']

    # 2. Load the pre-calculated G-Factor
    g_factor_path = os.path.join(paths['processed_data'], paths['g_factor_filename'])
    try:
        df_g_factor = pd.read_csv(g_factor_path)
        g_factor = df_g_factor["G_Factor"].values
    except FileNotFoundError:
        print(f"Error: G-Factor file not found at {g_factor_path}")
        print("Please run '1_calculate_g_factor.py' first.")
        return

    # 3. Process sample data to get average intensities
    par_avg, perp_avg, lambda_ex, _ = process_raw_to_avg_intensity(
        par_filepath=sample['parallel'],
        perp_filepath=sample['perpendicular'],
        bg_level=params['background_level'],
        lambda_0=params['peak_emission_wavelength'],
        window_size=params['integration_window_size'],
        file_loader_func=load_and_clean_eem_csv
    )

    # 4. Calculate anisotropy and relative energy
    anisotropy = calculate_anisotropy(par_avg, perp_avg, g_factor)
    relative_energy = calculate_relative_energy(lambda_ex, params['peak_emission_wavelength'])

    # 5. Assemble final DataFrame and save
    df_final = pd.DataFrame({
        "Excitation Wavelength (nm)": lambda_ex,
        "Par Avg": par_avg,
        "Perp Avg": perp_avg,
        "Perp Corrected": perp_avg * g_factor,
        "Anisotropy": anisotropy,
        "Relative Energy (eV)": relative_energy
    })
    
    output_dir = paths['processed_data']
    save_path_csv = os.path.join(output_dir, paths['anisotropy_filename'])
    df_final.to_csv(save_path_csv, index=False)
    print(f"Anisotropy results saved to: {save_path_csv}")

    # 6. Generate and save plots
    plots_dir = paths['plots']
    plot_and_save(
        df=df_final,
        x_col="Relative Energy (eV)",
        y_col="Anisotropy",
        title=f"Anisotropy vs. Relative Energy - Sample: {sample['name']}",
        xlabel="Relative Energy ($E_{ex} - E_{em}$) (eV)",
        ylabel="Anisotropy (r)",
        save_path=os.path.join(plots_dir, f"anisotropy_vs_energy_{sample['name']}.png")
    )
    plot_corrected_intensities(
        df=df_final,
        save_path=os.path.join(plots_dir, f"corrected_intensities_{sample['name']}.png")
    )
    print(f"Sample plots saved to: {plots_dir}")

if __name__ == "__main__":
    run_sample_analysis()