import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="OLC2 MACHINE LEARNING", page_icon="📊", initial_sidebar_state="expanded"
)

st.write(
    """
# 📊 OLC2 MACHINE LEARNING
Upload your Dataset.
"""
)

uploaded_file = st.file_uploader("Upload a document.", type=".csv")


#Se verifica que si se haya cargado un archivo a la aplicación
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.markdown("### Dataset preview")
    st.write(df)
    #Se hace visible un selectBox con todos los algoritmos disponibles a realizar
    option = st.selectbox(
    '¿What Algorithm do you want to use in the previously loaded data set?',
    ('linear regression', 'polynomial regression', 'Gaussian classifier','Decision tree classifier','neural networks'))
    st.write('You selected:', option)
    if(option=='linear regression'):
        st.write('holaaaaaaaa')

    