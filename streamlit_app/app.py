# streamlit_app/app.py

import sys
import os
import streamlit as st
import pandas as pd
import io
import zipfile
import yaml
from datetime import datetime
import matplotlib.pyplot as plt

# Add the project's root directory to Python's path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

# --- Helper Functions ---
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def create_zip_in_memory(params, file_names, g_factor_df, samples_results_dict):
    """Creates a ZIP file in memory containing all data and plots."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        config_data = {
            "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": params,
            "input_files": file_names
        }
        zip_file.writestr("run_config.yaml", yaml.dump(config_data, sort_keys=False))
        zip_file.writestr(f"processed_data/g_factor_data.csv", g_factor_df.to_csv(index=False))
        for label, df in samples_results_dict.items():
            zip_file.writestr(f"processed_data/{label}_result.csv", df.to_csv(index=False))

        dye_label = list(get_file_pairs(file_names)[0].keys())[0]
        all_files_results = {
            dye_label: g_factor_df.to_dict('list'),
            **{label: df.to_dict('list') for label, df in samples_results_dict.items()}
        }
        
        def save_fig_to_zip(fig, name):
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', bbox_inches='tight')
            zip_file.writestr(f"plots/{name}.png", img_buffer.getvalue())
            plt.close(fig)

        fig1, _ = plot_lamp_functions_all(all_files_results)
        save_fig_to_zip(fig1, '1_lamp_functions_all')
        
        fig2, _ = plot_raw_vs_lamp_corrected_all(all_files_results)
        save_fig_to_zip(fig2, '2_raw_vs_lamp_corrected')

        fig3, _ = plot_sample_corrected_only(samples_results_dict)
        save_fig_to_zip(fig3, '3_sample_corrected_only')

        fig4, _ = plot_correction_factor(g_factor_df)
        save_fig_to_zip(fig4, '4_correction_factor')

        fig5, _ = plot_dye_intensity_comparison(g_factor_df)
        save_fig_to_zip(fig5, '5_dye_intensity_comparison')

        fig6, _ = plot_corrected_intensities_all_samples(samples_results_dict)
        save_fig_to_zip(fig6, '6_corrected_intensities_all_samples')

        fig7, _ = plot_anisotropy_all_samples(samples_results_dict)
        save_fig_to_zip(fig7, '7_anisotropy_all_samples')

        figs_individual = plot_anisotropy_individual(samples_results_dict)
        for i, fig in enumerate(figs_individual):
            sample_label = list(samples_results_dict.keys())[i]
            save_fig_to_zip(fig, f'8_anisotropy_individual_{sample_label}')

    return zip_buffer.getvalue()

# --- Page Configuration ---
st.set_page_config(page_title="Anisotropy Analysis", layout="wide")
st.title("🔬 Photoluminescence Anisotropy Analysis")

# --- Sidebar for Controls ---
st.sidebar.header("Analysis Parameters")

uploaded_files = st.sidebar.file_uploader(
    "Upload ALL CSV files (par & perp)",
    type=["csv"],
    accept_multiple_files=True
)

# Initialize session state to store results
if 'results_generated' not in st.session_state:
    st.session_state.results_generated = False

if uploaded_files:
    params = {}
    st.sidebar.subheader("Global Parameters")
    params['background'] = st.sidebar.number_input("Background Level", value=990.0)
    params['window_size'] = st.sidebar.number_input("Window Size (points)", value=15, min_value=3, step=2)
    st.sidebar.subheader("Peak Emission λ₀ (nm)")
    
    file_names = [f.name for f in uploaded_files]
    dye_pair, sample_pairs = get_file_pairs(file_names)
    all_pairs = {**dye_pair, **sample_pairs}
    lambda_0_dict = {}
    
    for label, pair in all_pairs.items():
        default_lambda = 583.0
        lambda_0_dict[pair['parallel']] = st.sidebar.number_input(f"λ₀ for {label}", value=default_lambda, key=label)
        lambda_0_dict[pair['perpendicular']] = lambda_0_dict[pair['parallel']]

    params['lambda_0_dict'] = lambda_0_dict

    if st.sidebar.button("Run Analysis", type="primary"):
        with st.spinner('Processing data... Please wait.'):
            try:
                dye_pair_obj, sample_pairs_obj = get_file_pairs(uploaded_files)
                g_factor_df = calculate_g_factor_from_dye(dye_pair_obj, params)
                
                samples_results_dict = {}
                for label, pair in sample_pairs_obj.items():
                    sample_df = process_single_sample(label, pair, g_factor_df, params)
                    samples_results_dict[label] = sample_df
                
                # --- Store results in session state ---
                st.session_state.results_generated = True
                st.session_state.g_factor_df = g_factor_df
                st.session_state.samples_results_dict = samples_results_dict
                st.session_state.params = params
                st.session_state.file_names = file_names
                
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.exception(e)
                st.session_state.results_generated = False

# --- Results Display Section ---
# This block is now outside the button click, so it runs every time.
# It will only display results if they have been generated and stored.
if st.session_state.results_generated:
    st.success("Analysis Complete!")
    st.header("Results and Diagnostics")

    # Retrieve results from session state
    g_factor_df = st.session_state.g_factor_df
    samples_results_dict = st.session_state.samples_results_dict
    params = st.session_state.params
    file_names = st.session_state.file_names

    # --- Download Button ---
    zip_bytes = create_zip_in_memory(params, file_names, g_factor_df, samples_results_dict)
    st.download_button(
        label="⬇️ Download All Results (.zip)",
        data=zip_bytes,
        file_name=f"anisotropy_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip",
    )
    
    # --- Display DataFrames ---
    st.subheader("G-Factor (Dye) Data")
    st.dataframe(g_factor_df)

    for label, df in samples_results_dict.items():
        st.subheader(f"Results for {label}")
        st.dataframe(df)

    # --- Display Plots ---
    st.header("Diagnostic Plots")
    
    dye_pair_names, _ = get_file_pairs(file_names)
    dye_label = list(dye_pair_names.keys())[0]
    
    # We must convert the dataframes to dicts of lists for the plotting functions
    dye_results_dict_for_plot = g_factor_df.to_dict('list')
    samples_for_plotting = {label: df.to_dict('list') for label, df in samples_results_dict.items()}
    all_files_results = {dye_label: dye_results_dict_for_plot, **samples_for_plotting}
    
    # Plotting calls are now restored here
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