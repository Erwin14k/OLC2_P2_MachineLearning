import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from PIL import Image

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

def linearRegression(options_in_x,options_in_y,data,date):
    X = np.asarray(data[options_in_x]).reshape(-1, 1)
    Y = data[options_in_y]
    linear_regression = LinearRegression()
    linear_regression.fit(X, Y)
    Y_pred = linear_regression.predict(X)
    st.write(f'Mean Error: {mean_squared_error(Y, Y_pred, squared=True)}')
    st.write(f'Regression Coefficient: {linear_regression.coef_}' )
    st.write(f'R2: {r2_score(Y, Y_pred)}')
    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.show()
    Y_new = linear_regression.predict([[(int(date))]])
    st.write(f'Result: {Y_new}')
    plt.savefig("aa.png")
    plt.close()
    image = Image.open('aa.png')
    st.image(image, caption='Linear Regression')


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
            linearRegression(options_in_x,options_in_y,df,year_of_prediction)
            



        
        

    