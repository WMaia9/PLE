import pandas as pd
import os
from src.file_handler import load_config, load_and_clean_eem_csv
from src.data_processing import process_raw_to_avg_intensity, calculate_g_factor
from src.plotting import plot_and_save

def run_g_factor_calculation():
    """
    Executes the pipeline to calculate the instrumental G-factor
    from the reference dye measurements.
    """
    # 1. Load configuration
    config = load_config()
    params = config['processing_parameters']
    files = config['g_factor_files']
    paths = config['output_paths']

    # 2. Process raw data to get average intensities
    par_avg, perp_avg, lambda_ex, _ = process_raw_to_avg_intensity(
        par_filepath=files['parallel'],
        perp_filepath=files['perpendicular'],
        bg_level=params['background_level'],
        lambda_0=params['peak_emission_wavelength'],
        window_size=params['integration_window_size'],
        file_loader_func=load_and_clean_eem_csv
    )

    # 3. Calculate the G-Factor
    g_factor = calculate_g_factor(par_avg, perp_avg)

    # 4. Assemble and save the results DataFrame
    df_g_factor = pd.DataFrame({
        "Excitation Wavelength (nm)": lambda_ex,
        "G_Factor": g_factor
    })
    
    output_dir = paths['processed_data']
    os.makedirs(output_dir, exist_ok=True)
    save_path_csv = os.path.join(output_dir, paths['g_factor_filename'])
    df_g_factor.to_csv(save_path_csv, index=False)
    print(f"G-factor data saved to: {save_path_csv}")

    # 5. Generate and save the plot
    plot_path = os.path.join(paths['plots'], "g_factor_plot.png")
    plot_and_save(
        df=df_g_factor,
        x_col="Excitation Wavelength (nm)",
        y_col="G_Factor",
        title="Instrumental Correction (G-Factor)",
        xlabel="Excitation Wavelength (nm)",
        ylabel="G-Factor ($I_\\parallel / I_\\perp$)",
        save_path=plot_path
    )
    print(f"G-factor plot saved to: {plot_path}")

# This part allows the script to be run by itself OR imported by another script.
if __name__ == "__main__":
    run_g_factor_calculation()