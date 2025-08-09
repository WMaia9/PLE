# streamlit_app/pages/1_Anisotropy_vs_Excitation.py

import sys
import os
import io
import zipfile
from datetime import datetime

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yaml

# We are inside /streamlit_app/pages now — go two levels up so "src" is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data_processing import get_file_pairs, calculate_g_factor_from_dye, process_single_sample
from src.plotting import (
    plot_lamp_functions_all,
    plot_raw_vs_lamp_corrected_all,
    plot_sample_corrected_only,
    plot_correction_factor,
    plot_dye_intensity_comparison,
    plot_corrected_intensities_all_samples,
    plot_anisotropy_all_samples,
    plot_anisotropy_individual,
)

# ---------------- Helper ----------------
@st.cache_data
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def create_zip_in_memory(params, file_names, g_factor_df, samples_results_dict) -> bytes:
    """Create a ZIP in memory containing config, processed data, and plots."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Config / metadata
        config_data = {
            "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": params,
            "input_files": file_names,
        }
        zip_file.writestr("run_config.yaml", yaml.dump(config_data, sort_keys=False))

        # Processed CSVs
        zip_file.writestr("processed_data/g_factor_data.csv", g_factor_df.to_csv(index=False))
        for label, df in samples_results_dict.items():
            zip_file.writestr(f"processed_data/{label}_result.csv", df.to_csv(index=False))

        # Build dict for plotting utilities (names-based mapping)
        dye_pair_names = get_file_pairs(file_names)[0]  # returns (dye_pair_names, sample_pair_names)
        dye_label = list(dye_pair_names.keys())[0]
        all_files_results = {
            dye_label: g_factor_df.to_dict("list"),
            **{label: df.to_dict("list") for label, df in samples_results_dict.items()},
        }

        def save_fig_to_zip(fig, name: str):
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format="png", bbox_inches="tight")
            zip_file.writestr(f"plots/{name}.png", img_buffer.getvalue())
            plt.close(fig)

        # Plots
        fig1, _ = plot_lamp_functions_all(all_files_results)
        save_fig_to_zip(fig1, "1_lamp_functions_all")

        fig2, _ = plot_raw_vs_lamp_corrected_all(all_files_results)
        save_fig_to_zip(fig2, "2_raw_vs_lamp_corrected")

        fig3, _ = plot_sample_corrected_only(samples_results_dict)
        save_fig_to_zip(fig3, "3_sample_corrected_only")

        fig4, _ = plot_correction_factor(g_factor_df)
        save_fig_to_zip(fig4, "4_correction_factor")

        fig5, _ = plot_dye_intensity_comparison(g_factor_df)
        save_fig_to_zip(fig5, "5_dye_intensity_comparison")

        fig6, _ = plot_corrected_intensities_all_samples(samples_results_dict)
        save_fig_to_zip(fig6, "6_corrected_intensities_all_samples")

        fig7, _ = plot_anisotropy_all_samples(samples_results_dict)
        save_fig_to_zip(fig7, "7_anisotropy_all_samples")

        figs_individual = plot_anisotropy_individual(samples_results_dict)
        for i, fig in enumerate(figs_individual):
            sample_label = list(samples_results_dict.keys())[i]
            save_fig_to_zip(fig, f"8_anisotropy_individual_{sample_label}")

    return zip_buffer.getvalue()

# ---------------- Page config ----------------
st.set_page_config(page_title="Anisotropy vs. Excitation", layout="wide")
st.title("🔬 Photoluminescence Anisotropy — Anisotropy vs. Excitation")

# ---------------- Sidebar UI ----------------
st.sidebar.header("Analysis Parameters")

uploaded_files = st.sidebar.file_uploader(
    "Upload ALL CSV files (parallel & perpendicular)",
    type=["csv"],
    accept_multiple_files=True,
)

# Initialize session state keys once
if "results_generated" not in st.session_state:
    st.session_state.results_generated = False

if uploaded_files:
    params = {}
    st.sidebar.subheader("Global Parameters")
    params["background"] = st.sidebar.number_input("Background Level", value=990.0)
    params["window_size"] = st.sidebar.number_input("Window Size (points)", value=15, min_value=3, step=2)

    st.sidebar.subheader("Peak Emission λ₀ (nm)")

    # For UI only, we pass file NAMES so the user assigns lambda0 by label
    file_names = [f.name for f in uploaded_files]
    dye_pair_names, sample_pair_names = get_file_pairs(file_names)
    all_pairs = {**dye_pair_names, **sample_pair_names}

    lambda_0_dict = {}
    for label, pair in all_pairs.items():
        default_lambda = 583.0
        lambda_0_dict[pair["parallel"]] = st.sidebar.number_input(
            f"λ₀ for {label}", value=default_lambda, key=f"lambda0_{label}"
        )
        # keep parallel/perpendicular consistent
        lambda_0_dict[pair["perpendicular"]] = lambda_0_dict[pair["parallel"]]

    params["lambda_0_dict"] = lambda_0_dict

    if st.sidebar.button("Run Analysis", type="primary"):
        with st.spinner("Processing data... Please wait."):
            try:
                # For the pipeline, we pass file OBJECTS (not names)
                dye_pair_obj, sample_pairs_obj = get_file_pairs(uploaded_files)

                # 1) Dye G-factor (Cj) table using your original function
                g_factor_df = calculate_g_factor_from_dye(dye_pair_obj, params)

                # 2) Process each sample (emission-averaged anisotropy)
                samples_results_dict = {}
                for label, pair in sample_pairs_obj.items():
                    sample_df = process_single_sample(label, pair, g_factor_df, params)
                    samples_results_dict[label] = sample_df

                # ---- Store core results for later display & download ----
                st.session_state.results_generated = True
                st.session_state.g_factor_df = g_factor_df
                st.session_state.samples_results_dict = samples_results_dict
                st.session_state.params = params
                st.session_state.file_names = file_names
                st.session_state.uploaded_files = uploaded_files  # for Page 2 reuse

                # ---- NEW (safe): store excitation axis + Cj for Page 2 ----
                # Excitation axis from dye table (adjust column names if yours differ)
                lambda_ex = None
                for col in ["Excitation Wavelength (nm)", "lambda_ex_nm"]:
                    if col in g_factor_df.columns:
                        lambda_ex = g_factor_df[col].to_numpy()
                        break
                if lambda_ex is not None:
                    st.session_state.lambda_ex = lambda_ex
                else:
                    st.warning(
                        "Could not infer excitation axis from g_factor_df. "
                        "Page 2 will attempt to derive it from the dye file if needed."
                    )

                # Cj (G-factor) vector aligned with lambda_ex
                cj_col = next(
                    (c for c in ["Cj (Par / Perp)", "Cj", "G_factor", "G (Par/Perp)"] if c in g_factor_df.columns),
                    None,
                )
                if cj_col:
                    st.session_state.Cj_vector = g_factor_df[cj_col].to_numpy()
                else:
                    st.warning(
                        "Could not find a Cj column in g_factor_df. "
                        "Page 2 will require re-running Page 1 if Cj is missing."
                    )

                st.success("Analysis complete! See results below or open the Emission page when ready.")

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

# ---------------- Results section ----------------
if st.session_state.results_generated:
    st.success("Analysis Complete!")
    st.header("Results and Diagnostics")

    g_factor_df = st.session_state.g_factor_df
    samples_results_dict = st.session_state.samples_results_dict
    params = st.session_state.params
    file_names = st.session_state.file_names

    # Download everything as a ZIP
    zip_bytes = create_zip_in_memory(params, file_names, g_factor_df, samples_results_dict)
    st.download_button(
        label="⬇️ Download All Results (.zip)",
        data=zip_bytes,
        file_name=f"anisotropy_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip",
    )

    # DataFrames
    st.subheader("G-Factor (Dye) Data")
    st.dataframe(g_factor_df, use_container_width=True)

    for label, df in samples_results_dict.items():
        st.subheader(f"Results for {label}")
        st.dataframe(df, use_container_width=True)

    # Plots
    st.header("Diagnostic Plots")

    dye_pair_names, _ = get_file_pairs(file_names)
    dye_label = list(dye_pair_names.keys())[0]

    dye_results_dict_for_plot = g_factor_df.to_dict("list")
    samples_for_plotting = {label: df.to_dict("list") for label, df in samples_results_dict.items()}
    all_files_results = {dye_label: dye_results_dict_for_plot, **samples_for_plotting}

    fig1, _ = plot_lamp_functions_all(all_files_results)
    st.pyplot(fig1)

    fig2, _ = plot_raw_vs_lamp_corrected_all(all_files_results)
    st.pyplot(fig2)

    fig3, _ = plot_sample_corrected_only(samples_results_dict)
    st.pyplot(fig3)

    fig4, _ = plot_correction_factor(g_factor_df)
    st.pyplot(fig4)

    fig5, _ = plot_dye_intensity_comparison(g_factor_df)
    st.pyplot(fig5)

    fig6, _ = plot_corrected_intensities_all_samples(samples_results_dict)
    st.pyplot(fig6)

    fig7, _ = plot_anisotropy_all_samples(samples_results_dict)
    st.pyplot(fig7)

    figs_individual = plot_anisotropy_individual(samples_results_dict)
    for fig in figs_individual:
        st.pyplot(fig)

elif not uploaded_files:
    st.info("Please upload your CSV data files to begin.")
