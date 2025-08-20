# streamlit_app/Home.py  (or your current home filename)

import streamlit as st
import pandas as pd

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

# Optional: quick view of last summary created after saving refs
if method == "No dye (scalar C*)" and "no_dye_summary" in st.session_state:
    st.markdown("#### No-dye: last C* summary")
    st.dataframe(st.session_state.no_dye_summary, use_container_width=True)

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