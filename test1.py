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
    keys=df.head()
    #Se hace visible un selectBox con todos los algoritmos disponibles a realizar
    option = st.selectbox(
        '¿What Algorithm do you want to use in the previously loaded data set?',
        ('None','linear regression', 'polynomial regression', 'Gaussian classifier','Decision tree classifier','neural networks'))
    parameters_of_x=["None"]
    parameters_of_y=["None"]
    if(option=='linear regression'):
        st.markdown("### Linear Regression")
        for column_name in keys.columns:
            parameters_of_x.append(column_name)
            parameters_of_y.append(column_name)
        options_in_x = st.selectbox(
            '¿What attribute will be taken in X?',parameters_of_x)
        options_in_y = st.selectbox(
            '¿What attribute will be taken in Y?',parameters_of_y)
        year_of_prediction = st.text_input('Year Of Prediction', 'Ex. 2023')
        if (options_in_x!='None' and options_in_y!='None' and year_of_prediction!='Ex. 2023'):
            st.markdown("### All done")
            


def linearRegression(options_in_x,options_in_y):
    print()
        
        

    