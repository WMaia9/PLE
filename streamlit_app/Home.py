# streamlit_app/Home.py  (or your current home filename)

import streamlit as st
import pandas as pd
import io, zipfile, yaml, os, sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# so we can import your plotting helper from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.plotting import plot_emission_overlay_scatter

from src.data_processing import get_file_pairs
from src.plotting import (
    # Page 1 (matplotlib) – these generate PNG-friendly figures
    plot_lamp_functions_all,
    plot_raw_vs_lamp_corrected_all,
    plot_sample_corrected_only,
    plot_correction_factor,
    plot_dye_intensity_comparison,
    plot_corrected_intensities_all_samples,
    plot_anisotropy_all_samples,
    plot_anisotropy_individual,
    # Page 2 overlay (matplotlib)
    plot_emission_overlay_scatter,
)


st.set_page_config(page_title="PLE Anisotropy Suite", layout="wide")
st.title("PLE Anisotropy — Analysis Suite")

st.markdown("""
Use the sidebar to navigate between pages.

### 1) Anisotropy vs. Excitation (emission-averaged)
- Upload all *_par / *_perp CSVs at once  
- Choose **Use dye (vector Cj)** *or* **No dye (scalar C\*)**  
- The app applies background & lamp corrections and computes **r(λ_ex)**  
- In **No dye**, you’ll pick a **reference excitation** (λ_ref) on a plot; that sets a single **C\*** for the sample

### 2) Anisotropy vs. Emission (fixed λ_ex, sliced)
- Enter fixed excitation(s) in nm (comma-separated)
- Uses the same corrections; in **No dye** it reuses the **same C\*** from Page 1
""")

def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(s))

def build_full_results_zip() -> bytes:
    """Collect Page 1 + Page 2 outputs from session and zip them."""
    params          = st.session_state.get("params", {})
    file_names      = st.session_state.get("file_names", [])  # names from Page 1
    g_factor_df     = st.session_state.get("g_factor_df", pd.DataFrame())
    samples_results = st.session_state.get("samples_results_dict", {})  # {label: df}
    sliced_results  = st.session_state.get("sliced_results", {})        # {label: {lam_ex: df}}
    no_dye_summary  = st.session_state.get("no_dye_summary", pd.DataFrame())

    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:

        # ---------- config / metadata ----------
        meta = {
            "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": params,
            "input_files": file_names,
            "Excitation_present": bool(samples_results),
            "Emission_present": bool(sliced_results),
        }
        z.writestr("run_config.yaml", yaml.dump(meta, sort_keys=False))

        # ============================
        # Page 1 — Excitation outputs
        # ============================
        if isinstance(g_factor_df, pd.DataFrame) and not g_factor_df.empty:
            z.writestr("processed_data/Excitation/g_factor_data.csv", g_factor_df.to_csv(index=False))

        for label, df in samples_results.items():
            z.writestr(
                f"processed_data/Excitation/{_safe_name(label)}_excitation_result.csv",
                df.to_csv(index=False)
            )

        if isinstance(no_dye_summary, pd.DataFrame) and not no_dye_summary.empty:
            z.writestr("processed_data/Excitation/no_dye_summary.csv", no_dye_summary.to_csv(index=False))

        # ---- Page 1 plots (matplotlib) ----
        def _save_fig_to_zip(fig, name: str):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
            z.writestr(f"plots/Excitation/{name}.png", buf.getvalue())
            plt.close(fig)

        if samples_results:
            # Build dicts expected by your plotting funcs
            # Start with samples
            samples_for_plotting = {lab: df.to_dict("list") for lab, df in samples_results.items()}
            all_files_results = dict(samples_for_plotting)

            # If dye available, include it like Page 1 does
            if not g_factor_df.empty:
                try:
                    dye_pair_names, _ = get_file_pairs(file_names)  # returns ({dye_label: {...}}, {...})
                    dye_label = list(dye_pair_names.keys())[0]
                except Exception:
                    dye_label = "dye"
                all_files_results = {dye_label: g_factor_df.to_dict("list"), **all_files_results}

            # Generate same plots as Page 1 (skip dye-only when no dye)
            fig1, _ = plot_lamp_functions_all(all_files_results);                     _save_fig_to_zip(fig1, "1_lamp_functions_all")
            fig2, _ = plot_raw_vs_lamp_corrected_all(all_files_results);              _save_fig_to_zip(fig2, "2_raw_vs_lamp_corrected")
            fig3, _ = plot_sample_corrected_only(samples_results);                     _save_fig_to_zip(fig3, "3_sample_corrected_only")

            if not g_factor_df.empty:
                fig4, _ = plot_correction_factor(g_factor_df);                        _save_fig_to_zip(fig4, "4_correction_factor")
                fig5, _ = plot_dye_intensity_comparison(g_factor_df);                 _save_fig_to_zip(fig5, "5_dye_intensity_comparison")

            fig6, _ = plot_corrected_intensities_all_samples(samples_results);         _save_fig_to_zip(fig6, "6_corrected_intensities_all_samples")
            fig7, _ = plot_anisotropy_all_samples(samples_results);                    _save_fig_to_zip(fig7, "7_anisotropy_all_samples")

            figs_individual = plot_anisotropy_individual(samples_results)
            for i, fig in enumerate(figs_individual):
                sample_label = list(samples_results.keys())[i]
                _save_fig_to_zip(fig, f"8_anisotropy_individual_{_safe_name(sample_label)}")

        # ==========================
        # Page 2 — Emission outputs
        # ==========================
        if sliced_results:
            for label, ex_map in sliced_results.items():
                # CSVs per excitation
                for lam_ex_star, df_slice in ex_map.items():
                    nm = int(round(float(lam_ex_star)))
                    z.writestr(
                        f"processed_data/Emission/{_safe_name(label)}/{nm}nm_slices.csv",
                        df_slice.to_csv(index=False)
                    )

                # Overlay plot (same style you see on Page 2)
                try:
                    fig_overlay, _ = plot_emission_overlay_scatter(
                        ex_map,
                        sample_label=label,
                        xaxis="lambda",
                        xlim_nm=(568.0, 598.0),
                        ylim=(-0.020, 0.020),  # your updated y-range
                    )
                    buf = io.BytesIO()
                    fig_overlay.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                    plt.close(fig_overlay)
                    z.writestr(f"plots/Emission/{_safe_name(label)}_overlay.png", buf.getvalue())
                except Exception as e:
                    z.writestr(f"logs/Emission_plot_error_{_safe_name(label)}.txt", str(e))

    return zbuf.getvalue()

# ──────────────────────────────────────────────────────────────────────────────
# Session status
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### Session status")

params = st.session_state.get("params", {})
method = params.get("correction_method")
uploaded_files = st.session_state.get("uploaded_files", [])
results_generated = st.session_state.get("results_generated", False)

cols = st.columns(3)
cols[0].metric("Files in session", f"{len(uploaded_files)}")
cols[1].metric("Last method", method or "—")
cols[2].metric("Results ready", "Yes" if results_generated else "No")

if uploaded_files:
    with st.expander("Show uploaded files"):
        st.write([getattr(f, "name", str(f)) for f in uploaded_files])

# Show No-dye per-sample refs (if any)
if method == "No dye (scalar C*)" and st.session_state.get("no_dye_refs"):
    st.markdown("#### No-dye: λ_ref saved per sample")
    df_refs = pd.DataFrame({
        "Sample": list(st.session_state.no_dye_refs.keys()),
        "λ_ref (nm)": [float(v) for v in st.session_state.no_dye_refs.values()],
    })
    st.dataframe(df_refs, use_container_width=True)

# No-dye overview
if method == "No dye (scalar C*)":
    st.markdown("#### No-dye: λ_ref per sample")

    # If the summary exists (created after saving on Page 1), show that:
    if "no_dye_summary" in st.session_state and not st.session_state.no_dye_summary.empty:
        st.dataframe(st.session_state.no_dye_summary[["Sample", "λ_ref used (nm)", "C* used"]],
                     use_container_width=True)
    else:
        # Otherwise, build a table from results/refs so all samples appear
        rows = []
        sample_results = st.session_state.get("samples_results_dict", {})
        refs = st.session_state.get("no_dye_refs", {})
        for lab in sorted(sample_results.keys()):
            df = sample_results.get(lab)
            lam_used = None
            if df is not None and "lambda_ref_nm_used" in df.columns:
                # preferred: read from the results table (constant column)
                lam_used = float(pd.to_numeric(df["lambda_ref_nm_used"], errors="coerce").dropna().median())
            elif lab in refs:
                lam_used = float(refs[lab])
            else:
                lam_used = float(st.session_state.get("params", {}).get("lambda_ref_nm", 450.0))
            rows.append({"Sample": lab, "λ_ref (nm)": lam_used})

        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ------------------------------------------------------------------
# Download everything (Pages 1 & 2) from Home
# ------------------------------------------------------------------
have_p1 = bool(st.session_state.get("samples_results_dict"))
have_p2 = bool(st.session_state.get("sliced_results"))
if have_p1 or have_p2:
    zip_bytes = build_full_results_zip()
    st.download_button(
        "⬇️ Download ALL results (Excitation + Emission)",
        data=zip_bytes,
        file_name=f"ple_anisotropy_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip",
        use_container_width=True,
    )
else:
    st.info("No results to download yet. Run Page 1 and/or Page 2 first.")

# ──────────────────────────────────────────────────────────────────────────────
# Quick actions
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("### Quick actions")

c1, c2 = st.columns(2)

with c1:
    repick_disabled = not uploaded_files or method != "No dye (scalar C*)"
    if st.button("🔁 Re-open No-dye reference picker on Page 1", disabled=repick_disabled):
        # Page 1 will rebuild pairs from uploaded_files; just set the phase flag and hide results
        st.session_state.phase = "pick_ref"
        st.session_state.results_generated = False
        st.success("Go to **Anisotropy vs. Excitation** (Page 1) from the sidebar — the No-dye picker will be open.")

with c2:
    if st.button("🧹 Clear session / start over"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# --- About section ---
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