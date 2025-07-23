# Polarization Anisotropy Analysis from PLE Spectra

This project provides a complete Python pipeline to calculate polarization anisotropy from Photoluminescence Excitation (PLE) spectra. It processes raw spectral data, applies necessary corrections—including background subtraction, lamp function normalization, and instrumental G-factor correction—and computes the final anisotropy values for analysis.

## Methodology

The analysis pipeline is based on a standard methodology for correcting and analyzing polarized fluorescence data.

1.  [cite_start]**Data Acquisition**: The process starts with two Photoluminescence Excitation (PLE) matrices for each sample: one with the analyzer polarized parallel to the incident light ($M^{(||)}$) and one with perpendicular polarization ($M^{(\perp)}$). [cite: 1, 3, 4] [cite_start]Each matrix element represents photoluminescence intensity as a function of both emission and excitation wavelengths. [cite: 5]

2.  **Data Structure**:
    * [cite_start]**Excitation Wavelengths ($\lambda^{ex}$)**: Sampled from 450 nm to 600 nm with a 1 nm increment. [cite: 15, 17, 19]
    * [cite_start]**Emission Wavelengths ($\lambda^{em}$)**: A discrete set of 1024 points from approximately 499.349 nm to 629.315 nm. [cite: 20, 21]
    * [cite_start]**Lamp Function ($L_j$)**: The intensity profile of the excitation lamp is appended to each data matrix to be used for spectral correction. [cite: 24, 25, 26]

3.  **Processing Steps**:
    * [cite_start]**Emission Slice Selection**: To reduce noise, a symmetric window of 15 emission data points is selected around a central peak emission wavelength ($\lambda_0$). [cite: 48]
    * [cite_start]**Averaging & Background Subtraction**: The intensity across this 15-point window is averaged for each excitation wavelength. [cite: 54, 55] [cite_start]A constant background intensity is then subtracted. [cite: 58, 60]
    * [cite_start]**Lamp Correction**: The averaged, background-subtracted intensity is divided by the lamp function ($L_j$) to correct for non-uniform illumination from the excitation source. [cite: 65, 67]
    * [cite_start]**G-Factor Correction**: A dimensionless correction factor ($C_j$, or "G-factor") is calculated from a reference dye sample ($C_j = I_{dye}^{(||,corr)} / I_{dye}^{(\perp,corr)}$). [cite: 75] [cite_start]This factor corrects for the instrument's wavelength-dependent sensitivity to different polarizations and is applied to the sample's perpendicular intensity data. [cite: 70, 78]
    * [cite_start]**Anisotropy Calculation**: The final polarization anisotropy ($r$) is calculated using the fully corrected parallel and perpendicular intensities. [cite: 86, 87]
    * [cite_start]**Energy Conversion**: For plotting and analysis, the excitation wavelength axis is converted to photon energy (eV) relative to the emission peak. [cite: 81, 83]

## Project Structure

```
anisotropy_project/
├── data/
│   ├── par_dye.csv
│   └── ...
├── results/
│   ├── plots/
│   └── processed_data/
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── file_handler.py
│   └── plotting.py
├── config.yaml
├── requirements.txt
├── step1_calculate_g_factor.py
├── a2_process_sample.py
└── run_full_pipeline.py
```

## Installation

1.  **Clone the repository.**
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can run the analysis in two ways: step-by-step or all at once.

#### 1. Step-by-Step Execution

First, run the calibration script, then the analysis script.

```bash
# Step 1: Calculate the G-factor from the dye sample
python step1_calculate_g_factor.py

# Step 2: Analyze your sample using the generated G-factor
python a2_process_sample.py
```

#### 2. Full Pipeline Execution

To run the entire process (calibration and analysis) with a single command, use the master script.

```bash
python run_full_pipeline.py
```

<details>
<summary><b>Click to see the required code for the pipeline scripts</b></summary>

**`step1_calculate_g_factor.py`**
This file must contain the `run_g_factor_calculation()` function.

```python
# In step1_calculate_g_factor.py

def run_g_factor_calculation():
    """The main logic for the G-factor calculation."""
    # ... all the code that was previously in main() goes here ...
    print("G-factor calculation complete.")

if __name__ == "__main__":
    run_g_factor_calculation()
```

**`a2_process_sample.py`**
This file must contain the `run_sample_analysis()` function.

```python
# In a2_process_sample.py

def run_sample_analysis():
    """The main logic for the sample analysis."""
    # ... all the code that was previously in main() goes here ...
    print("Sample analysis complete.")

if __name__ == "__main__":
    run_sample_analysis()
```

**`run_full_pipeline.py`**
This master script correctly imports from your named files.

```python
# In run_full_pipeline.py

import time
from step1_calculate_g_factor import run_g_factor_calculation
from a2_process_sample import run_sample_analysis

def main():
    """Runs the full analysis pipeline."""
    print("🚀 Starting full analysis pipeline...")
    
    # Step 1: Calibration
    print("\n--- Step 1: Calculating G-Factor ---")
    start_time = time.time()
    run_g_factor_calculation()
    print(f"--- Step 1 finished in {time.time() - start_time:.2f} seconds ---\n")
    
    # Step 2: Sample Analysis
    print("--- Step 2: Analyzing Sample ---")
    start_time = time.time()
    run_sample_analysis()
    print(f"--- Step 2 finished in {time.time() - start_time:.2f} seconds ---\n")
    
    print("✅ Full pipeline finished successfully!")

if __name__ == "__main__":
    main()
```

</details>

## Configuration

All experimental parameters can be adjusted in the **`config.yaml`** file. This includes:
* File paths for input data.
* Background level (`background_level`).
* Peak emission wavelength (`peak_emission_wavelength`).
* Integration window size (`integration_window_size`).
* Output filenames and paths.

## Outputs

The pipeline generates two main types of output in the `results/` directory:
* **Processed Data (`/processed_data`):** CSV files containing the calculated G-factor and the final anisotropy results for each sample.
* **Plots (`/plots`):** PNG images visualizing the G-factor, corrected intensities, and the final plot of anisotropy vs. relative energy.