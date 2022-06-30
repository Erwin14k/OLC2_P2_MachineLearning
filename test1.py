import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="A/B Testing App", page_icon="📊", initial_sidebar_state="expanded"
)

st.write(
    """
# 📊 OLC2 MACHINE LEARNING
Upload your Dataset.
"""
)

uploaded_file = st.file_uploader("Upload CSV", type=".csv")

ab_default = None
result_default = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.markdown("### Dataset preview")
    st.write(df)

    