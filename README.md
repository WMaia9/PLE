# Interactive Photoluminescence Anisotropy (Streamlit)

An interactive **Streamlit** app to compute fluorescence anisotropy from Photoluminescence Excitation (PLE) spectra. It loads parallel/perpendicular CSVs, applies background and lamp corrections, supports **two correction methods** (Dye vector `Cj` and No-dye scalar `C*`), and produces publication-ready tables and plots.

---

## Highlights

- **Multi-page Streamlit UI**
  - **Home** — session status and a single **Download ALL** (Excitation + Emission).
  - **1) Anisotropy vs. Excitation** — emission-averaged `r(λ_ex)`.
  - **2) Anisotropy vs. Emission** — fixed-excitation, **sliced** `r(λ_em)`.

- **Two correction methods**
  - **Use dye (vector `Cj`)** — computes `Cj(λ_ex) = I∥/I⊥` from a dye pair; optional smoothing (Moving Average, Savitzky–Golay, Gaussian).
  - **No dye (scalar `C*`)** — you **pick a reference excitation** on a plot; app sets a **single** `C* = (I∥/I⊥)@λ_ref` and applies it globally.

- **Upload all files at once** — automatic pairing of `*_par.csv` & `*_perp.csv` (also supports `par_*` / `perp_*`). Dye files are detected when the filename contains **`dye`**.

- **Session-aware** — re-open the No-dye picker without re-uploading; results persist across pages.

- **Full export** — per-page downloads plus a **Home** button that zips **everything** (Excitation + Emission), including plots.

---

## File format & naming

**CSV layout (PAR and PERP files):**

- First **two** rows: headers (ignored).
- Column 0 (after dropping headers): **emission wavelength (nm)**.
- Columns 1..N (all rows except the last): intensities for each **excitation** (one column per `λ_ex`).
- **Last row**: **lamp vector vs. excitation** (length N).

> The app lamp-corrects **each channel with its own lamp row** (PAR uses the last row of the PAR CSV; PERP uses the last row of the PERP CSV).

**Filenames (pairing rules):**

- Each sample needs **two** files: one **parallel** and one **perpendicular**.
  - Suffix style: `sample1_par.csv` and `sample1_perp.csv`, or
  - Prefix style: `par_sample1.csv` and `perp_sample1.csv`.
- **Dye method**: include the dye pair; filenames must contain the word **`dye`**, e.g. `dye_par.csv`, `dye_perp.csv`.

---

## Methodology (what the app does)

1. **Background subtraction**  
   `I' = I − background` (scalar background, default ~990).

2. **Lamp correction** (per channel)  
   `I∥,corr = I'∥ / L∥`, `I⊥,corr = I'⊥ / L⊥`.

3. **Emission window averaging** (Page 1)  
   Around your chosen `λ₀`, average over `window_size` points for each excitation column → get emission-averaged `I∥(λ_ex)` and `I⊥(λ_ex)`.

4. **Correction factor**
   - **Dye**: `Cj(λ_ex) = I∥,dye / I⊥,dye` (optional smoothing).
   - **No-dye**: pick `λ_ref`; set a **constant** `C* = (I∥/I⊥)@λ_ref`.

5. **Anisotropy** (both methods)  
   \[
   r = \frac{I_\parallel - C\,I_\perp}{I_\parallel + 2\,C\,I_\perp}
   \]
   where `C = Cj(λ_ex)` (dye) or **`C*`** (no-dye).

6. **Page 2 (Emission-sliced)**  
   For any fixed `λ_ex*`, compute slice means across the emission band near `λ₀` and apply the same correction:
   - **Dye**: use `Cj(λ_ex*)`.
   - **No-dye**: reuse the **same constant `C*`** chosen on Page 1.

---

## Project Structure

```
PLE/
├── streamlit_app/
│ ├── Home.py
│ └── pages/
│ ├── 1_Anisotropy_vs_Excitation.py
│ └── 2_Anisotropy_vs_Emission.py
├── src/
│ ├── init.py
│ ├── data_processing.py
│ └── plotting.py
├── requirements.txt
├── .streamlit/
│ └── config.toml # optional: server/headless config
└── README.md
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

streamlit run streamlit_app/Home.py

## Output

processed_data/
  Excitation/
    g_factor_data.csv                      # dye mode only
    <sample>_excitation_result.csv
    no_dye_summary.csv                     # no-dye only (λ_ref and C*)
  Emission/
    <sample>/<EX_nm>nm_slices.csv

plots/
  Excitation/
    1_lamp_functions_all.png
    2_raw_vs_lamp_corrected.png
    3_sample_corrected_only.png
    4_correction_factor.png                # dye only
    5_dye_intensity_comparison.png         # dye only
    6_corrected_intensities_all_samples.png
    7_anisotropy_all_samples.png
    8_anisotropy_individual_<sample>.png
  Emission/
    <sample>_overlay.png

run_config.yaml