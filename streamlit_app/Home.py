import streamlit as st

st.set_page_config(page_title="PLE Anisotropy Suite", layout="wide")
st.title("PLE Anisotropy — Analysis Suite")

st.markdown("""
Use the sidebar to navigate:

**1. Anisotropy vs. Excitation (emission-averaged)**  
Loads files, computes dye G-factor (Cj), applies lamp/background corrections, and plots r as a function of excitation.

**2. Anisotropy vs. Emission (fixed λ_ex, sliced)**  
Reuses the corrections to compute r across the emission band at fixed excitation(s).
""")

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

