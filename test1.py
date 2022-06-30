import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="A/B Testing App", page_icon="📊", initial_sidebar_state="expanded"
)

st.write(
    """
# 📊 A/B Testing App
Upload your experiment results to see the significance of your A/B test.
"""
)

uploaded_file = st.file_uploader("Upload CSV", type=".csv")

ab_default = None
result_default = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.markdown("### Data preview")
    st.dataframe(df.head())

    st.markdown("### Select columns for analysis")
    with st.form(key="my_form"):
        ab = st.multiselect(
            "A/B column",
            options=df.columns,
            help="Select which column refers to your A/B testing labels.",
            default=ab_default,
        )