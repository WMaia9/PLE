# Interactive Photoluminescence Anisotropy Analysis

This is a Python-based application designed to calculate fluorescence anisotropy from Photoluminescence Excitation (PLE) spectra. It provides an interactive Graphical User Interface (GUI) for setting experimental parameters, processes raw spectral data, applies all necessary corrections, and generates a comprehensive set of diagnostic plots and data files for scientific analysis.

## Features

-   **Interactive GUI:** A user-friendly `tkinter` interface allows for setting all parameters at runtime, eliminating the need to edit code.
-   **Multi-File Processing:** Analyze a reference dye and multiple samples in a single, organized run.
-   **Automated G-Factor Calculation:** Automatically calculates the dynamic (vector) G-Factor from a specified reference dye.
-   **Comprehensive Outputs:** Generates 8 detailed diagnostic plots to visualize every step of the analysis, from raw data to the final anisotropy results.
-   **Organized Runs:** Each analysis is saved into a unique, timestamped folder, containing the configuration, processed data, and plots for perfect reproducibility.
-   **Full Data Export:** Exports all intermediate and final data to CSV files for further analysis in other software.

## Methodology

The application follows a standard scientific workflow for anisotropy calculation:
1.  **Data Loading:** Raw parallel (`par`) and perpendicular (`perp`) CSV files are loaded and cleaned.
2.  **Background Subtraction:** A user-defined background level is subtracted from all intensity data.
3.  **Lamp Function Correction:** The background-subtracted data is normalized by the lamp intensity vector to correct for variations in the excitation source.
4.  **Emission Window Averaging:** For each excitation wavelength, intensities are averaged across a user-defined window (e.g., 15 points) centered on a specified emission peak (`λ₀`).
5.  **G-Factor Calculation:** A dynamic G-Factor vector (`Cj` or `G`) is calculated from the reference dye data using the formula: $G(\lambda_{ex}) = I_{\parallel,dye}(\lambda_{ex}) / I_{\perp,dye}(\lambda_{ex})$
6.  **Anisotropy Calculation:** The final anisotropy (`r`) is calculated for each sample using the fully corrected intensities:
    $$
    r(\lambda_{ex}) = \frac{I_{\parallel,sample} - G \cdot I_{\perp,sample}}{I_{\parallel,sample} + 2 \cdot G \cdot I_{\perp,sample}}
    $$

## Project Structure

```
anisotropy_project/
├── data/
│   └── (Your CSV files here)
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── file_handler.py
│   ├── gui.py
│   └── plotting.py
├── .gitignore
├── requirements.txt
└── main.py
```

## Installation

1.  Clone or download the project repository.
2.  It is highly recommended to create a Python virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

The entire application is launched from a single entry point.

1.  Run the main script from your terminal:
    ```bash
    python main.py
    ```
2.  **Follow the prompts:**
    * A file dialog will open. Select all the `par` and `perp` CSV files for your experiment.
    * The parameter GUI will appear. Adjust the background, window size, and peak emission wavelengths (`λ₀`) for your dye and samples.
    * A final dialog will ask you to select an output directory where the results will be saved.
3.  **Interactive Plotting:** The application will then run the analysis, pausing after each step to display a diagnostic plot. **Close each plot window to proceed to the next step.**

## Output

For each run, a new timestamped folder (e.g., `anisotropy_run_20250723_160530`) will be created containing:
-   `run_config.yaml`: A file with the exact parameters used for the analysis.
-   **`processed_data/`** subfolder:
    -   `g_factor_data.csv`: Intermediate data from the dye calculation.
    -   `[sample_name]_result.csv`: The final, complete data table for each sample.
-   **`plots/`** subfolder:
    -   8 different PNG files visualizing every stage of the analysis.