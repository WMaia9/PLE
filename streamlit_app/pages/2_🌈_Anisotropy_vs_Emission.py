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
    emission_sliced_anisotropy_at_fixed_exc,  # fixed in data_processing to use excitation *columns*
)
from src.plotting import (
    plot_emission_overlay_scatter,  # new scatter overlay that matches your reference style
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

# Pull excitation axis and Cj from Page 1; if missing, derive from g_factor_df
lambda_ex = st.session_state.get("lambda_ex")
Cj_vector = st.session_state.get("Cj_vector")
g_factor_df = st.session_state.get("g_factor_df")

if lambda_ex is None and g_factor_df is not None:
    for col in ["Excitation Wavelength (nm)", "lambda_ex_nm"]:
        if col in g_factor_df.columns:
            lambda_ex = g_factor_df[col].to_numpy()
            break

if Cj_vector is None and g_factor_df is not None:
    cj_col = next((c for c in ["Cj (Par / Perp)", "Cj", "G_factor", "G (Par/Perp)"] if c in g_factor_df.columns), None)
    if cj_col:
        Cj_vector = g_factor_df[cj_col].to_numpy()

missing = [k for k, v in dict(lambda_ex=lambda_ex, Cj_vector=Cj_vector).items() if v is None]
if missing:
    st.error(f"Missing required data from Page 1: {missing}. Please re-run Page 1.")
    st.stop()

# ---------------- UI ----------------
st.sidebar.header("Slicing Parameters")
lambda_ex_list_str = st.sidebar.text_input("Fixed excitation(s) in nm (comma-separated)", value="532, 561")
slice_points = int(st.sidebar.number_input("Emission slice width (points)", min_value=1, max_value=101, value=15, step=1))
num_slices   = int(st.sidebar.number_input("Number of slices", min_value=5, max_value=100, value=15, step=1))
xlim_min, xlim_max = 568.0, 598.0  # default window to mimic the reference figure
ylim_min, ylim_max = -0.024, 0.010

# Parse excitation list
lambda_ex_list = []
for token in lambda_ex_list_str.split(","):
    token = token.strip()
    if token:
        try:
            lambda_ex_list.append(float(token))
        except ValueError:
            st.error(f"Invalid excitation value: {token}")

background = float(st.session_state.params.get("background", 0.0))
lambda_0_dict = st.session_state.params.get("lambda_0_dict", {})  # filename -> nm

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

# ---------------- Build pairs from uploaded files ----------------
dye_pair, sample_pairs = get_file_pairs(uploaded_files)  # (dye_pair_obj, sample_pairs_obj)

# Resolve dye file objects (not strictly needed for Page 2 math, but we keep parity)
dye_label = list(dye_pair.keys())[0]
dye_par  = dye_pair[dye_label]["parallel"]
dye_perp = dye_pair[dye_label]["perpendicular"]

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

            # Emission axis from parallel CSV (same as Page-1 convention)
            try:
                lambda_em_local = infer_emission_axis_from_csv(spair["parallel"])
            except Exception as ex:
                st.error(f"Could not infer emission axis for '{sample_label}': {ex}")
                continue

            per_ex_results = {}
            for lam_ex_star in lambda_ex_list:
                try:
                    df_slice = emission_sliced_anisotropy_at_fixed_exc(
                        sample_par_path = spair["parallel"],
                        sample_perp_path= spair["perpendicular"],
                        dye_par_path    = dye_par,   # kept for signature parity
                        dye_perp_path   = dye_perp,  # kept for signature parity
                        lambda_em       = lambda_em_local,
                        lambda_ex       = np.asarray(lambda_ex, dtype=float),
                        lambda_0        = float(lam0),
                        lambda_ex_star  = float(lam_ex_star),
                        background      = float(background),
                        Cj_vector       = np.asarray(Cj_vector, dtype=float),
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

        # Overlay scatter (matches your reference style; tight y/x ranges)
        fig_overlay, _ = plot_emission_overlay_scatter(
            ex_map,
            sample_label=sample_label,
            xaxis="lambda",
            xlim_nm=(xlim_min, xlim_max),
            ylim=(ylim_min, ylim_max),
        )
        st.pyplot(fig_overlay)

        # Optional: also show each per-λ_ex table
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
