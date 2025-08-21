# streamlit_app/pages/2_Anisotropy_vs_Emission.py

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

# Make project root importable: .../PLE/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data_processing import (
    get_file_pairs,
    emission_sliced_anisotropy_at_fixed_exc,   # DYE method (existing)
    emission_sliced_anisotropy_no_dye,         # NO DYE method
    _calculate_average_intensities,             # utility for C* calculation
)
from src.plotting import (
    plot_emission_overlay_scatter,  # scatter overlay that matches the reference style
)

st.set_page_config(page_title="Anisotropy vs. Emission", layout="wide")
st.title("🌈 Photoluminescence Anisotropy — Anisotropy vs. Emission (fixed excitation, sliced)")

# ---------------- Guards ----------------
if not st.session_state.get("results_generated", False):
    st.warning("Please run **Anisotropy vs. Excitation** first.")
    st.stop()

uploaded_files = st.session_state.get("uploaded_files")
if not uploaded_files:
    st.warning("Uploaded files not found in the session. Please re-run Page 1.")
    st.stop()

params_p1 = st.session_state.get("params", {})
correction_method = params_p1.get("correction_method", "Use dye (vector Cj)")

# Pull excitation axis and Cj from Page 1; if missing, derive lambda_ex as fallback when needed
lambda_ex = st.session_state.get("lambda_ex")
Cj_vector = st.session_state.get("Cj_vector")
g_factor_df = st.session_state.get("g_factor_df")

if correction_method == "Use dye (vector Cj)":
    if lambda_ex is None and g_factor_df is not None:
        for col in ["Excitation Wavelength (nm)", "lambda_ex_nm"]:
            if col in g_factor_df.columns:
                lambda_ex = g_factor_df[col].to_numpy()
                break

    if Cj_vector is None and g_factor_df is not None:
        if "Cj Smoothed" in g_factor_df.columns:
            Cj_vector = g_factor_df["Cj Smoothed"].to_numpy()
            st.info("ℹ️ Using smoothed Cj vector for emission-based anisotropy calculations.")
        else:
            cj_col = next((c for c in ["Cj (Par / Perp)", "Cj", "G_factor", "G (Par/Perp)"] if c in g_factor_df.columns), None)
            if cj_col:
                Cj_vector = g_factor_df[cj_col].to_numpy()

    missing = [k for k, v in dict(lambda_ex=lambda_ex, Cj_vector=Cj_vector).items() if v is None]
    if missing:
        st.error(f"Missing required data from Page 1: {missing}. Please re-run Page 1 in dye mode.")
        st.stop()

# ---------------- UI ----------------
st.sidebar.header("Slicing Parameters")
lambda_ex_list_str = st.sidebar.text_input("Fixed excitation(s) in nm (comma-separated)", value="532, 561")

# points-based window (matches Page 1 convention)
slice_points = int(st.sidebar.number_input("Emission slice width (points)", min_value=1, max_value=101, value=15, step=1))
num_slices   = int(st.sidebar.number_input("Number of slices",            min_value=5, max_value=100, value=15, step=1))

# Overlay axes (optional, keep your previous look)
#xlim_min, xlim_max = 568.0, 598.0
#ylim_min, ylim_max = -0.025, 0.025

# ---------------- About (sidebar) ----------------
st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.markdown(
    """
    This application was developed as a project by students from the 
    **University of California, Merced**.

    - **Intellectual Property:** Fernanda Hlousek
    - **Lead Developer:** Wesley Maia

    *© 2025 Fernanda Hlousek. All Rights Reserved.*
    """
)

# Parse excitation list
lambda_ex_list = []
for token in lambda_ex_list_str.split(","):
    token = token.strip()
    if token:
        try:
            lambda_ex_list.append(float(token))
        except ValueError:
            st.error(f"Invalid excitation value: {token}")

# Pull common params from Page 1
background = float(params_p1.get("background", 990.0))
lambda_0_dict = params_p1.get("lambda_0_dict", {})  # filename -> nm

# ---------------- Utilities ----------------
def rewind_safe(file_like):
    try:
        file_like.seek(0)
    except Exception:
        pass

def infer_emission_axis_from_csv(file_like) -> np.ndarray:
    """
    MATCHES Page-1 layout:
      - drop the first 2 header rows
      - col 0 is emission wavelength (nm)
      - last row is lamp (so exclude it from the emission axis)
    """
    rewind_safe(file_like)
    df = pd.read_csv(file_like, header=None)
    if len(df) <= 3:
        raise RuntimeError("CSV too short to infer emission axis.")
    df = df.drop(index=[0, 1]).reset_index(drop=True)
    lamb = pd.to_numeric(df.iloc[:-1, 0], errors="coerce").to_numpy(dtype=float)
    if np.isnan(lamb).any():
        raise RuntimeError("Non-numeric emission axis entries detected.")
    rewind_safe(file_like)
    return lamb

def _pair_sample_files_loose(file_list):
    """Pairs par/perp without requiring a 'dye' label."""
    import re
    def _name(x): return x.name if hasattr(x, "name") else os.path.basename(x)
    pairs = {}
    for x in file_list:
        fname = _name(x).lower()
        base, ext = os.path.splitext(fname)
        if ext != ".csv":
            continue
        m_suffix = re.search(r'([_-])(par|perp)$', base)
        m_prefix = re.match(r'^(par|perp)[_-]', base)
        if m_suffix:
            kind = "parallel" if m_suffix.group(2) == "par" else "perpendicular"
            label = base[:m_suffix.start()].strip("_- ")
        elif m_prefix:
            kind = "parallel" if m_prefix.group(1) == "par" else "perpendicular"
            label = base[m_prefix.end():].strip("_- ")
        else:
            continue
        if not label: label = "unnamed"
        pairs.setdefault(label, {})
        if kind in pairs[label]:
            raise ValueError(f"Duplicate {kind} for label '{label}': {_name(x)}")
        pairs[label][kind] = x
    return {lab: d for lab, d in pairs.items() if set(d.keys()) == {"parallel","perpendicular"}}

# ---------------- Build pairs from uploaded files ----------------
if correction_method == "Use dye (vector Cj)":
    # Requires a dye pair
    try:
        dye_pair, sample_pairs = get_file_pairs(uploaded_files)  # (dye_pair_obj, sample_pairs_obj)
    except Exception as e:
        st.error(f"Dye mode selected but failed to find dye files: {e}")
        st.stop()
    dye_label = list(dye_pair.keys())[0]
    dye_par  = dye_pair[dye_label]["parallel"]
    dye_perp = dye_pair[dye_label]["perpendicular"]
else:
    # No-dye: pair samples without requiring a dye
    try:
        sample_pairs = _pair_sample_files_loose(uploaded_files)
    except Exception as e:
        st.error(f"Pairing error: {e}")
        st.stop()
    dye_par = dye_perp = None

if not sample_pairs:
    st.warning("No sample pairs found. Filenames must contain 'par'/'perp'.")
    st.stop()

# ---------------- Run ----------------
if st.button("Compute emission-sliced anisotropy", type="primary"):
    with st.spinner("Processing slices..."):
        results_sliced = {}  # sample_label -> {lambda_ex_star: DataFrame}

        for sample_label, spair in sample_pairs.items():
            # Lookup λ0 by filename (as set on Page 1's sidebar)
            lam0 = None
            for cand in (spair["parallel"], spair["perpendicular"]):
                fname = getattr(cand, "name", str(cand))
                if fname in lambda_0_dict:
                    lam0 = float(lambda_0_dict[fname])
                    break
            if lam0 is None:
                st.error(f"Missing λ₀ for sample '{sample_label}'. Set it on Page 1.")
                continue

            # Emission axis from parallel CSV (same convention as Page 1)
            try:
                lambda_em_local = infer_emission_axis_from_csv(spair["parallel"])
            except Exception as ex:
                st.error(f"Could not infer emission axis for '{sample_label}': {ex}")
                continue

            # Show the effective window so you can confirm it (~25–30 nm)
            if len(lambda_em_local) > 1:
                em_step = float(np.median(np.diff(lambda_em_local)))
                est_window_nm = em_step * (slice_points * num_slices)
                st.caption(f"{sample_label}: estimated emission step ≈ {em_step:.3f} nm • total window ≈ {est_window_nm:.1f} nm")

            per_ex_results = {}
            for lam_ex_star in lambda_ex_list:
                try:
                    if correction_method == "Use dye (vector Cj)":
                        # --- DYE branch (unchanged) ---
                        df_slice = emission_sliced_anisotropy_at_fixed_exc(
                            sample_par_path = spair["parallel"],
                            sample_perp_path= spair["perpendicular"],
                            dye_par_path    = dye_par,   # signature parity
                            dye_perp_path   = dye_perp,  # signature parity
                            lambda_em       = lambda_em_local,
                            lambda_ex       = np.asarray(lambda_ex, dtype=float),
                            lambda_0        = float(lam0),
                            lambda_ex_star  = float(lam_ex_star),
                            background      = float(background),
                            Cj_vector       = np.asarray(Cj_vector, dtype=float),
                            slice_points    = slice_points,
                            num_slices      = num_slices,
                        )
                    else:
                        # --- NO-DYE branch (THIS IS THE PART YOU NEED) ---

                        # 1) get C* for THIS sample from Page 1 results table
                        C_star = None
                        try:
                            df_page1 = st.session_state.samples_results_dict.get(sample_label)
                            if df_page1 is not None and "Correction Factor (Cj)" in df_page1.columns:
                                vals = pd.to_numeric(df_page1["Correction Factor (Cj)"],
                                                    errors="coerce").to_numpy(dtype=float)
                                # In no-dye, this column is a constant scalar replicated; use median to be safe.
                                C_star = float(np.nanmedian(vals))
                        except Exception:
                            C_star = None

                        # 2) if we couldn't read it (e.g. first run), recompute C* like Page 1
                        if C_star is None:
                            pre = _calculate_average_intensities(
                                spair["parallel"], spair["perpendicular"],
                                {"background": background, "window_size": 15,
                                "lambda_0_dict": {getattr(spair["parallel"], "name", "par.csv"): lam0}}
                            )
                            lam_ex_tmp = pre["lambda_ex"]
                            par_avg    = pre["par_avg_lamp_corr"]
                            perp_avg   = pre["perp_avg_lamp_corr"]
                            lam_ref = float(st.session_state.get("no_dye_refs", {}).get(sample_label, 450.0))
                            j_ref  = int(np.argmin(np.abs(lam_ex_tmp - lam_ref)))
                            eps    = 1e-12
                            denom  = perp_avg[j_ref] if abs(perp_avg[j_ref]) > eps else (eps if perp_avg[j_ref] == 0 else np.sign(perp_avg[j_ref]) * eps)
                            C_star = float(par_avg[j_ref] / denom)

                        # 3) make sure we have an excitation axis (Page 1 convention)
                        lam_ex_axis = np.asarray(lambda_ex, dtype=float) if lambda_ex is not None else None
                        if lam_ex_axis is None:
                            rewind_safe(spair["parallel"])
                            df_tmp = pd.read_csv(spair["parallel"], header=None).drop(index=[0, 1]).reset_index(drop=True)
                            m_cols = df_tmp.shape[1] - 1
                            lam_ex_axis = np.arange(450.0, 450.0 + m_cols, dtype=float)

                        # 4) call the corrected NO-DYE slicer (note the C_star=... argument)
                        df_slice = emission_sliced_anisotropy_no_dye(
                            sample_par_path = spair["parallel"],
                            sample_perp_path= spair["perpendicular"],
                            lambda_em       = lambda_em_local,
                            lambda_ex       = lam_ex_axis,
                            lambda_0        = float(lam0),
                            lambda_ex_star  = float(lam_ex_star),
                            background      = float(background),
                            C_star          = float(C_star),          # <<< pass C*
                            slice_points    = slice_points,
                            num_slices      = num_slices,
                        )

                    per_ex_results[lam_ex_star] = df_slice

                except Exception as ex:
                    st.error(f"Error at {sample_label} @ {lam_ex_star} nm: {ex}")

            if per_ex_results:
                results_sliced[sample_label] = per_ex_results

        if results_sliced:
            st.session_state.sliced_results = results_sliced
            st.success("Done! Scroll down to view tables and plots.")

# ---------------- Output ----------------
if "sliced_results" in st.session_state:
    for sample_label, ex_map in st.session_state.sliced_results.items():
        st.subheader(f"Sample: {sample_label}")

        # Overlay scatter (matches your reference style; tight axes)
        fig_overlay, _ = plot_emission_overlay_scatter(
            ex_map,
            sample_label=sample_label,
            xaxis="lambda",
            #xlim_nm=(xlim_min, xlim_max),
            #ylim=(ylim_min, ylim_max),
        )
        st.pyplot(fig_overlay)

        # Optional: per-λ_ex tables & downloads
        with st.expander("Show per-excitation tables"):
            for lam_ex_star, df_slice in ex_map.items():
                st.markdown(f"**λ_ex = {int(lam_ex_star)} nm**")
                st.dataframe(
                    df_slice[["Emission Center (nm)", "Emission Energy (eV)", "Anisotropy (slice)"]],
                    use_container_width=True,
                )
                csv_bytes = df_slice.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"Download CSV — {sample_label} @ {int(lam_ex_star)} nm",
                    data=csv_bytes,
                    file_name=f"emission_sliced_anisotropy_{sample_label}_{int(lam_ex_star)}nm.csv",
                    mime="text/csv",
                )
else:
    st.info("Set parameters and click **Compute emission-sliced anisotropy**.")