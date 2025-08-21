# streamlit_app/pages/1_Anisotropy_vs_Excitation.py

# ------------ Imports ----------------
import sys
import os
import io
import zipfile
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yaml

# We are inside /streamlit_app/pages now — go two levels up so "src" is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data_processing import (
    get_file_pairs,
    calculate_g_factor_from_dye,
    process_single_sample,
    process_single_sample_no_dye,
    smooth_cj_vector,
    smooth_cj_moving_average,
    smooth_cj_gaussian,
    _calculate_average_intensities,
)

from src.plotting import (
    plot_lamp_functions_all,
    plot_raw_vs_lamp_corrected_all,
    plot_sample_corrected_only,
    plot_correction_factor,
    plot_dye_intensity_comparison,
    plot_corrected_intensities_all_samples,
    plot_anisotropy_all_samples,
    plot_anisotropy_individual,
    plot_correction_factor_smoothed,
    # Plotly versions
    plot_lamp_functions_all_plotly,
    plot_raw_vs_lamp_corrected_all_plotly,
    plot_sample_corrected_only_plotly,
    plot_correction_factor_plotly,
    plot_dye_intensity_comparison_plotly,
    plot_corrected_intensities_all_samples_plotly,
    plot_anisotropy_all_samples_plotly,
    plot_anisotropy_individual_plotly,
    plot_correction_factor_smoothed_plotly,
)

# ---------------- Helper ----------------
@st.cache_data
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def create_zip_in_memory(params, file_names, g_factor_df, samples_results_dict) -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        config_data = {
            "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": params,
            "input_files": file_names,
        }
        zip_file.writestr("run_config.yaml", yaml.dump(config_data, sort_keys=False))

        if not g_factor_df.empty:
            zip_file.writestr("processed_data/g_factor_data.csv", g_factor_df.to_csv(index=False))
        for label, df in samples_results_dict.items():
            zip_file.writestr(f"processed_data/{label}_result.csv", df.to_csv(index=False))

        # Build dict for plotting utilities (names-based mapping)
        all_files_results = {}
        if not g_factor_df.empty:
            try:
                dye_pair_names, _ = get_file_pairs(file_names)
                dye_label = list(dye_pair_names.keys())[0]
                all_files_results[dye_label] = g_factor_df.to_dict("list")
            except Exception:
                pass
        all_files_results.update({label: df.to_dict("list") for label, df in samples_results_dict.items()})

        def save_fig_to_zip(fig, name: str):
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format="png", bbox_inches="tight")
            zip_file.writestr(f"plots/{name}.png", img_buffer.getvalue())
            plt.close(fig)

        # Plots
        fig1, _ = plot_lamp_functions_all(all_files_results);            save_fig_to_zip(fig1, "1_lamp_functions_all")
        fig2, _ = plot_raw_vs_lamp_corrected_all(all_files_results);     save_fig_to_zip(fig2, "2_raw_vs_lamp_corrected")
        fig3, _ = plot_sample_corrected_only(samples_results_dict);       save_fig_to_zip(fig3, "3_sample_corrected_only")

        if not g_factor_df.empty:
            fig4, _ = plot_correction_factor(g_factor_df);               save_fig_to_zip(fig4, "4_correction_factor")
            fig5, _ = plot_dye_intensity_comparison(g_factor_df);        save_fig_to_zip(fig5, "5_dye_intensity_comparison")

        fig6, _ = plot_corrected_intensities_all_samples(samples_results_dict); save_fig_to_zip(fig6, "6_corrected_intensities_all_samples")
        fig7, _ = plot_anisotropy_all_samples(samples_results_dict);     save_fig_to_zip(fig7, "7_anisotropy_all_samples")
        figs_individual = plot_anisotropy_individual(samples_results_dict)
        for i, fig in enumerate(figs_individual):
            sample_label = list(samples_results_dict.keys())[i]
            save_fig_to_zip(fig, f"8_anisotropy_individual_{sample_label}")

    return zip_buffer.getvalue()

def _pair_sample_files_loose(file_list):
    """
    Pairs *_par / *_perp files WITHOUT requiring a 'dye' label.
    Accepts either UploadedFile objects or string names.
    Returns: {label: {"parallel": <same type>, "perpendicular": <same type>}}
    """
    import re
    def _name(x):
        return x.name if hasattr(x, "name") else os.path.basename(x)

    pairs = {}
    for x in file_list:
        fname = _name(x).lower()
        base, ext = os.path.splitext(fname)
        if ext != ".csv":
            continue
        # prefix: par_<label>.csv or suffix: <label>_par.csv (also allow '-')
        m_suffix = re.search(r'([_-])(par|perp)$', base)
        m_prefix = re.match(r'^(par|perp)[_-]', base)
        if m_suffix:
            kind = "parallel" if m_suffix.group(2) == "par" else "perpendicular"
            label = base[:m_suffix.start()].strip("_- ")
        elif m_prefix:
            kind = "parallel" if m_prefix.group(1) == "par" else "perpendicular"
            label = base[m_prefix.end():].strip("_- ")
        else:
            # skip files that don't indicate par/perp
            continue
        if not label:
            label = "unnamed"
        pairs.setdefault(label, {})
        if kind in pairs[label]:
            raise ValueError(f"Duplicate {kind} file for label '{label}': {_name(x)}")
        pairs[label][kind] = x

    # keep only complete pairs
    pairs = {lab: d for lab, d in pairs.items() if set(d.keys()) == {"parallel", "perpendicular"}}
    return pairs

# ---------------- Page config ----------------
st.set_page_config(page_title="Anisotropy vs. Excitation", layout="wide")
st.title("🔬 Photoluminescence Anisotropy — Anisotropy vs. Excitation")

# ---------------- Sidebar UI ----------------
st.sidebar.header("Analysis Parameters")

# Choose correction method first
st.sidebar.subheader("Correction Method")
_default_method = st.session_state.get("params", {}).get("correction_method", "Use dye (vector Cj)")
correction_method = st.sidebar.radio(
    "Choose correction method",
    ["Use dye (vector Cj)", "No dye (scalar C*)"],
    index=0 if _default_method == "Use dye (vector Cj)" else 1,
)

# Global fallback only; actual λ_ref is chosen later in No-dye flow
lambda_ref_nm = 450.0

uploaded_files = st.sidebar.file_uploader(
    "Upload ALL CSV files (parallel & perpendicular)",
    type=["csv"],
    accept_multiple_files=True,
)

# Reuse files from session so you don't need to re-upload when navigating back
if not uploaded_files and st.session_state.get("uploaded_files"):
    uploaded_files = st.session_state.uploaded_files
    st.sidebar.caption("Using previously uploaded files from session.")

# Initialize session state keys once
if "results_generated" not in st.session_state:
    st.session_state.results_generated = False
if "phase" not in st.session_state:
    st.session_state.phase = None

if uploaded_files:
    st.sidebar.subheader("Advanced Options")

    # Only show smoothing when using the dye-based method
    if correction_method == "Use dye (vector Cj)":
        smoothing_option = st.sidebar.selectbox(
            "🧹 Apply Smoothing to Correction Curve (Cj)",
            ["None", "Moving Average", "Savitzky-Golay", "Gaussian"],
            index=0
        )
    else:
        smoothing_option = "None"
        st.sidebar.caption("Smoothing applies only to dye-based Cj. Disabled in No dye mode.")

    params = {}
    st.sidebar.subheader("Global Parameters")
    params["background"] = st.sidebar.number_input("Background Level", value=990.0)
    params["window_size"] = st.sidebar.number_input("Window Size (points)", value=15, min_value=3, step=2)
    params["smoothing_option"] = smoothing_option

    st.sidebar.subheader("Peak Emission λ₀ (nm)")

    # For UI only, we pass file NAMES so the user assigns lambda0 by label
    file_names = [f.name for f in uploaded_files]

    if correction_method == "Use dye (vector Cj)":
        dye_pair_names, sample_pair_names = get_file_pairs(file_names)  # may raise if no dye
        all_pairs = {**dye_pair_names, **sample_pair_names}
    else:
        # no-dye: build only sample pairs, no dye required
        sample_pair_names = _pair_sample_files_loose(file_names)
        all_pairs = {**sample_pair_names}

    lambda_0_dict = {}
    for label, pair in all_pairs.items():
        default_lambda = 583.0
        lambda_0_dict[pair["parallel"]] = st.sidebar.number_input(
            f"λ₀ for {label}", value=default_lambda, key=f"lambda0_{label}"
        )
        # keep parallel/perpendicular consistent
        lambda_0_dict[pair["perpendicular"]] = lambda_0_dict[pair["parallel"]]

    params["lambda_0_dict"] = lambda_0_dict

    # ---------------- Run Analysis ----------------
    if st.sidebar.button("Run Analysis", type="primary"):
        with st.spinner("Processing data... Please wait."):
            try:
                file_names = [f.name for f in uploaded_files]
                name_to_obj = {f.name: f for f in uploaded_files}

                samples_results_dict = {}
                g_factor_df = pd.DataFrame()

                if correction_method == "Use dye (vector Cj)":
                    # Names → pairs (requires dye)
                    dye_pair_names, sample_pair_names = get_file_pairs(file_names)
                    # Names → objects
                    dye_pair_obj = {
                        list(dye_pair_names.keys())[0]: {
                            "parallel": name_to_obj[list(dye_pair_names.values())[0]["parallel"]],
                            "perpendicular": name_to_obj[list(dye_pair_names.values())[0]["perpendicular"]],
                        }
                    }
                    sample_pairs_obj = {
                        lab: {"parallel": name_to_obj[p["parallel"]], "perpendicular": name_to_obj[p["perpendicular"]]}
                        for lab, p in sample_pair_names.items()
                    }

                    # 1) Dye G-factor (Cj)
                    g_factor_df = calculate_g_factor_from_dye(dye_pair_obj, params)

                    # Optional smoothing
                    cj_col = next((c for c in ["Cj (Par / Perp)", "Cj", "G_factor", "G (Par/Perp)"] if c in g_factor_df.columns), None)
                    if cj_col:
                        try:
                            cj_values = g_factor_df[cj_col].to_numpy()
                            if smoothing_option == "Moving Average":
                                g_factor_df["Cj Smoothed"] = smooth_cj_moving_average(cj_values, window_size=params["window_size"])
                            elif smoothing_option == "Savitzky-Golay":
                                g_factor_df["Cj Smoothed"] = smooth_cj_vector(cj_values, window_length=params["window_size"], polyorder=2)
                            elif smoothing_option == "Gaussian":
                                g_factor_df["Cj Smoothed"] = smooth_cj_gaussian(cj_values, sigma=2.0)
                            if "Cj Smoothed" in g_factor_df:
                                st.info(f"✅ Smoothing method applied: {smoothing_option}")
                        except Exception as e:
                            st.warning(f"Smoothing failed: {e}")
                    else:
                        st.warning("Cj column not found. Smoothing skipped.")

                    # 2) Process each sample using dye vector
                    for label, pair in sample_pairs_obj.items():
                        sample_df = process_single_sample(label, pair, g_factor_df, params)
                        samples_results_dict[label] = sample_df

                    # ---- Store core results (DYE branch) ----
                    params["correction_method"] = correction_method
                    params["lambda_ref_nm"] = float(lambda_ref_nm)  # unused here but ok to store
                    st.session_state.results_generated = True
                    st.session_state.g_factor_df = g_factor_df
                    st.session_state.samples_results_dict = samples_results_dict
                    st.session_state.params = params
                    st.session_state.file_names = file_names
                    st.session_state.uploaded_files = uploaded_files
                    st.session_state.phase = "done"

                    # Optional: excitation axis for page 2 (only exists in dye mode table)
                    for col in ["Excitation Wavelength (nm)", "lambda_ex_nm"]:
                        if col in g_factor_df.columns:
                            st.session_state.lambda_ex = g_factor_df[col].to_numpy()
                            break

                    st.success("Analysis complete! See results below or open the Emission page when ready.")

                else:
                    # NO DYE: enter pick-ref phase; don't compute results yet
                    sample_pairs_obj = _pair_sample_files_loose(uploaded_files)
                    if not sample_pairs_obj:
                        raise ValueError("No sample pairs detected. Please upload *_par.csv and *_perp.csv files.")

                    # init state for pick-ref flow
                    params["correction_method"] = correction_method
                    params["lambda_ref_nm"] = float(lambda_ref_nm)  # fallback if not saved per sample
                    st.session_state.phase = "pick_ref"
                    st.session_state.sample_pairs_obj_store = sample_pairs_obj
                    st.session_state.params = params
                    st.session_state.file_names = file_names
                    st.session_state.uploaded_files = uploaded_files
                    if "no_dye_refs" not in st.session_state:
                        st.session_state.no_dye_refs = {}

                    # Clear any prior results so the Results section does NOT render yet
                    st.session_state.g_factor_df = pd.DataFrame()
                    st.session_state.samples_results_dict = {}
                    st.session_state.results_generated = False

                    st.success("Now choose the reference point for a sample below, then click 'Save ref for this sample' to generate results.")

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.exception(e)
                st.session_state.results_generated = False

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

# -------------- No-dye pick-ref phase (only after Run Analysis) --------------
# REPLACE your entire No-dye picker block in:
# streamlit_app/pages/1_Anisotropy_vs_Excitation.py
# (the block that starts with: if st.session_state.get("phase") == "pick_ref":)

if st.session_state.get("phase") == "pick_ref":
    st.header("Pick Reference Excitation (No dye)")

    sample_pairs_obj = st.session_state.sample_pairs_obj_store
    params_preview   = st.session_state.params

    # ensure the dict exists and DO NOT auto-fill other samples
    if "no_dye_refs" not in st.session_state:
        st.session_state.no_dye_refs = {}

    # 1) choose which sample to preview
    sel_label = st.selectbox(
        "Select a sample to preview",
        sorted(sample_pairs_obj.keys()),
        key="no_dye_sel_label",
    )

    # 2) build preview using your preprocessing
    pre = _calculate_average_intensities(
        sample_pairs_obj[sel_label]["parallel"],
        sample_pairs_obj[sel_label]["perpendicular"],
        params_preview,
    )
    lambda_ex = pre["lambda_ex"]
    par_avg   = pre["par_avg_lamp_corr"]
    perp_avg  = pre["perp_avg_lamp_corr"]

    # default index = previously saved nm for THIS sample (if any), else 450 nm
    default_nm = float(st.session_state.no_dye_refs.get(sel_label, 450.0))
    j_default  = int(np.argmin(np.abs(lambda_ex - default_nm)))

    j_star = st.slider(
        "Move the marker to choose λ_ref (index on excitation axis)",
        min_value=0,
        max_value=len(lambda_ex)-1,
        value=j_default,
        key=f"no_dye_ref_idx_{sel_label}",
    )
    sel_nm = float(lambda_ex[j_star])
    st.caption(f"Selected λ_ref for **{sel_label}** ≈ {sel_nm:.1f} nm")

    # plot Par/Perp (lamp-corr) + vertical marker
    fig = go.Figure()
    fig.add_scatter(x=lambda_ex, y=par_avg,  mode="lines", name="Par Avg (lamp-corr)")
    fig.add_scatter(x=lambda_ex, y=perp_avg, mode="lines", name="Perp Avg (lamp-corr)")
    fig.add_vline(x=sel_nm)
    st.plotly_chart(fig, use_container_width=True)

    # 3) save options (NO automatic propagation!)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✓ Save λ_ref for THIS sample only"):
            st.session_state.no_dye_refs[sel_label] = sel_nm
            st.success(f"Saved λ_ref = {sel_nm:.1f} nm for {sel_label}")

    with c2:
        if st.button("✓ Save this λ_ref for ALL samples (overwrite)"):
            for lab in sample_pairs_obj.keys():
                st.session_state.no_dye_refs[lab] = sel_nm
            st.success(f"Saved λ_ref = {sel_nm:.1f} nm for ALL samples")

    # 4) show current per-sample refs (what will be used)
    rows = []
    for lab in sorted(sample_pairs_obj.keys()):
        nm = st.session_state.no_dye_refs.get(lab, None)
        rows.append({
            "Sample": lab,
            "λ_ref saved (nm)": f"{nm:.1f}" if isinstance(nm, (int, float)) else "— (will use default 450)"
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # 5) compute results using SAVED refs ONLY (unset → default 450)
    #    This avoids the issue where one pick (e.g., 583) accidentally applies to others.
    if st.button("▶ Compute results (No dye)"):
        results = {}
        summary_rows = []
        for label, pair in sample_pairs_obj.items():
            lam_ref_for_label = float(st.session_state.no_dye_refs.get(label, 450.0))
            df, meta = process_single_sample_no_dye(
                label, pair, st.session_state.params, lambda_ref_nm=lam_ref_for_label
            )
            results[label] = df
            summary_rows.append({
                "Sample": label,
                "λ_ref used (nm)": float(meta["lambda_ref_nm"]),
                "C* used": float(meta["C_star"]),
            })

        # finalize state and show results section
        st.session_state.g_factor_df = pd.DataFrame()  # no dye => empty
        st.session_state.samples_results_dict = results
        st.session_state.results_generated = True
        st.session_state.phase = "done"
        st.session_state.no_dye_summary = pd.DataFrame(summary_rows)
        st.success("Results generated. Scroll down to see tables and plots.")

# ---------- Quick action: re-open No-dye picker without re-uploading ----------
if (
    st.session_state.get("results_generated")
    and st.session_state.get("params", {}).get("correction_method") == "No dye (scalar C*)"
    and st.session_state.get("uploaded_files")
):
    if st.button("🔁 Re-pick reference (No dye)"):
        try:
            # Rebuild sample pairs from stored files and enter pick_ref phase
            sample_pairs_obj = _pair_sample_files_loose(st.session_state.uploaded_files)
            st.session_state.sample_pairs_obj_store = sample_pairs_obj
            st.session_state.phase = "pick_ref"
            # Hide results until user saves the new reference
            st.session_state.results_generated = False
            st.rerun()
        except Exception as e:
            st.error(f"Could not re-open reference picker: {e}")

# ---------------- Results section ----------------
if st.session_state.results_generated:
    st.success("Analysis Complete!")
    st.header("Results and Diagnostics")

    g_factor_df = st.session_state.g_factor_df
    samples_results_dict = st.session_state.samples_results_dict
    params = st.session_state.params
    file_names = st.session_state.file_names

    # Which method was used
    correction_method = params.get("correction_method", "Use dye (vector Cj)")

    # Download everything as a ZIP
    zip_bytes = create_zip_in_memory(params, file_names, g_factor_df, samples_results_dict)
    st.download_button(
        label="⬇️ Download All Results (.zip)",
        data=zip_bytes,
        file_name=f"anisotropy_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip",
    )

    # Dye table (only show in dye mode)
    if correction_method == "Use dye (vector Cj)" and not g_factor_df.empty:
        st.subheader("G-Factor (Dye) Data")
        st.dataframe(g_factor_df, use_container_width=True)

    for label, df in samples_results_dict.items():
        st.subheader(f"Results for {label}")
        st.dataframe(df, use_container_width=True)

    # Plots
    st.header("Diagnostic Plots")

    # Prepare results for plotting (works in both modes)
    samples_for_plotting = {label: df.to_dict("list") for label, df in samples_results_dict.items()}
    all_files_results = dict(samples_for_plotting)
    if correction_method == "Use dye (vector Cj)" and not g_factor_df.empty:
        try:
            dye_pair_names, _ = get_file_pairs(file_names)
            dye_label = list(dye_pair_names.keys())[0]
            all_files_results = {dye_label: g_factor_df.to_dict("list"), **all_files_results}
        except Exception:
            pass

    # Dye-only plots
    if correction_method == "Use dye (vector Cj)" and not g_factor_df.empty:
        st.subheader("Correction Factor (Cj or G-Factor) – Original")
        fig_dye_g = plot_correction_factor_plotly(g_factor_df)
        st.plotly_chart(fig_dye_g)

        if params.get("smoothing_option") != "None" and "Cj Smoothed" in g_factor_df.columns:
            st.subheader(f"Correction Factor (Cj) – Smoothed ({params.get('smoothing_option')})")
            fig_dye_smooth = plot_correction_factor_smoothed_plotly(g_factor_df)
            st.plotly_chart(fig_dye_smooth)

            # Replace Cj column downstream…
            cj_col = next((c for c in ["Cj (Par / Perp)", "Cj", "G_factor", "G (Par/Perp)"] if c in g_factor_df.columns), None)
            if cj_col:
                g_factor_df[cj_col] = g_factor_df["Cj Smoothed"]
                st.success(f"✅ Smoothed values are now replacing '{cj_col}' in G-Factor Data for downstream calculations.")
                for sample_label, df in samples_results_dict.items():
                    if "Excitation Wavelength (nm)" in df.columns:
                        excitation = df["Excitation Wavelength (nm)"]
                        cj_interp = np.interp(
                            excitation,
                            g_factor_df["Excitation Wavelength (nm)"],
                            g_factor_df["Cj Smoothed"]
                        )
                        df["Correction Factor (Cj)"] = cj_interp

        st.subheader("Dye: Parallel vs Perpendicular Intensity")
        fig_dye_comp = plot_dye_intensity_comparison_plotly(g_factor_df)
        st.plotly_chart(fig_dye_comp)

    # Shared plots (both modes)
    st.subheader("Lamp Functions – All Samples")
    st.plotly_chart(plot_lamp_functions_all_plotly(all_files_results))

    st.subheader("Raw vs Lamp-Corrected – All Samples")
    st.plotly_chart(plot_raw_vs_lamp_corrected_all_plotly(all_files_results))

    st.subheader("Corrected Intensities – Individual Samples")
    st.plotly_chart(plot_sample_corrected_only_plotly(samples_results_dict))

    st.subheader("Corrected Intensities – All Samples")
    st.plotly_chart(plot_corrected_intensities_all_samples_plotly(samples_results_dict))

    st.subheader("Anisotropy – All Samples")
    st.plotly_chart(plot_anisotropy_all_samples_plotly(samples_results_dict))

    st.subheader("Anisotropy – Individual Samples")
    for fig in plot_anisotropy_individual_plotly(samples_results_dict):
        st.plotly_chart(fig)

elif not uploaded_files:
    st.info("Please upload your CSV data files to begin.")