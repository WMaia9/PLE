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

## Workflow (pages)

### Home
- Overview + session status (files, last method, saved No-dye references).
- Quick actions:
  - **Re-open No-dye reference picker** (Page 1) without re-uploading.
  - **Clear session** to start fresh.
- **⬇️ Download ALL results (Excitation + Emission)** — one ZIP with all data + plots.

---

### 1) Anisotropy vs. Excitation (emission-averaged)
1. **Choose correction method** in the sidebar:
   - **Use dye (vector `Cj`)**
   - **No dye (scalar `C*`)**
2. **Upload all CSVs at once** (all samples, and the dye pair if using Dye).
3. **Set `λ₀`** (peak emission) per pair.
4. Click **Run Analysis**.

**Dye path**
- Computes `Cj(λ_ex) = I∥/I⊥` from the dye pair.
- Optional smoothing: Moving Average, Savitzky–Golay, or Gaussian.
- Computes sample anisotropy using `Cj(λ_ex)`.
- Shows dye tables/plots + shared diagnostics.

**No-dye path**
- After **Run Analysis**, a plot appears to **pick `λ_ref`** (by index/marker on the excitation axis).
- Click **Save ref for this sample** → the app computes results for **all samples** using the single constant `C* = (I∥/I⊥)@λ_ref`.
- You can re-open the picker later from **Home** or Page 1 (no re-upload).

**Outputs on Page 1**
- Per-sample excitation tables (and dye table if applicable).
- Diagnostic plots: lamp functions, raw vs. corrected, corrected intensities, anisotropy (all + individual), and dye plots when in Dye mode.

---

### 2) Anisotropy vs. Emission (fixed excitation, sliced)
1. Enter **fixed excitation(s)** in nm (comma-separated).
2. Set **Emission slice width (points)** and **Number of slices**.
3. Click **Compute emission-sliced anisotropy**.

**How it computes**
- Uses the same background & lamp corrections.
- **Dye:** applies `Cj(λ_ex*)` at the chosen excitation.
- **No-dye:** reuses the **same constant `C*`** chosen on Page 1.
- Slices the emission band around `λ₀` and computes `r` per slice.

**Outputs on Page 2**
- One overlay plot per sample (default y-axis: **−0.020 … +0.020**).
- One CSV **per excitation** with slice centers and anisotropy.

## Outputs

When you finish Page 1 and/or Page 2, you can download everything from **Home → “Download ALL results (Excitation + Emission)”**.  
The ZIP contains CSVs and plots organized by page.

### What each file contains

**`processed_data/Excitation/<sample>_excitation_result.csv`**
- `Excitation Wavelength (nm)`
- `Par Avg`, `Perp Avg`
- `Correction Factor (Cj)`  
  - **Dye:** vector `Cj(λ_ex)`  
  - **No-dye:** constant `C*` repeated for all rows
- `Perp Corrected` = `C · Perp Avg`
- `Anisotropy` = `(Par − C·Perp) / (Par + 2·C·Perp)`
- `Energy Relative to Lambda_0 (eV)`
- `lambda_0`
- `raw_lamp_vector`, `par_avg_raw`
- *(if your build writes them)* `lambda_ref_nm_used`, `C_star_used` (No-dye)

**`processed_data/Excitation/g_factor_data.csv`** *(Dye mode only)*
- `Excitation Wavelength (nm)`, `Par Avg`, `Perp Avg`, `Cj (Par / Perp)`
- May include smoothed `Cj` if you enabled smoothing
- Also: `perp_avg_g_corr`, `raw_lamp_vector`, `par_avg_raw`

**`processed_data/Excitation/no_dye_summary.csv`** *(No-dye only)*
- `Sample`, `λ_ref used (nm)`, `C* used`

**`processed_data/Emssion/<sample>/<EX_nm>nm_slices.csv`** *(Page 2)*
- `Emission Center (nm)`, `Emission Energy (eV)`
- `Anisotropy (slice)`
- `lambda_ex_star`, `lambda_0`

**Plots under `plots/Excitation/`**
- Same diagnostics as Page 1:
  - lamp functions, raw vs lamp-corrected, corrected intensities, anisotropy (all + per-sample)
  - and, when **Dye** is used: correction factor and dye intensities

**`plots/Emssion/<sample>_overlay.png`**
- Page 2 overlay of slice anisotropy vs emission (tight x-limits as configured; y-range −0.020 … +0.020 by default)

**`run_config.yaml`**
- Timestamp, parameters used, list of input filenames, and flags indicating which pages are present in the ZIP.

### Notes
- Folder names in the ZIP are intentionally **`Excitation/`** and **`Emssion/`**.
- In **No-dye**, Page 2 reuses the **same constant `C*`** determined on Page 1.
- If you don’t see dye-specific files/plots, it means No-dye was selected (that’s expected).

## Troubleshooting

- **“No reference ‘dye’ files found” (Dye mode)**  
  Include a dye pair. The filenames must contain **`dye`** (e.g., `dye_par.csv`, `dye_perp.csv`).

- **“Expected suffix _par/_perp …”**  
  Each sample needs both `_par.csv` and `_perp.csv` (or prefix style `par_*` / `perp_*`) exported from the same instrument run.

- **“No columns to parse from file”**  
  The CSV is empty/corrupted or the pointer wasn’t rewound. Re-export the CSVs; re-upload if needed. The app attempts to `seek(0)` internally.

- **“Lamp vectors differ”**  
  The last row (lamp vector) in PAR and PERP must have the same number of excitation columns. Ensure both files are from the same acquisition and not truncated.

- **Page 2 plot is ~0 everywhere (No-dye)**  
  You must reuse the **same constant `C*`** from Page 1. Make sure you are on the build that passes `C*` from Page 1 to Page 2 (the Home ZIP + Page-2 code provided here already do this).

- **Can’t pick a new `λ_ref` without re-uploading (No-dye)**  
  Use **Home → “Re-open No-dye reference picker”** (or the same button on Page 1). This returns you to the picker using the files already in session.

- **Streamlit tries to open a browser (gio … Operation not supported)**  
  Run Streamlit in headless mode or set a config:
  ```toml
  # .streamlit/config.toml
  [server]
  headless = true

## Credits

- **Intellectual Property:** Fernanda Hlousek  
- **Lead Developer:** Wesley Maia  
- **Affiliation:** University of California, Merced  
- © 2025 Fernanda Hlousek. All Rights Reserved.
  
