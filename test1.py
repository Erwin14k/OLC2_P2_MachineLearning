import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures;
from sklearn.metrics import mean_squared_error, r2_score
from PIL import Image
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB;
from sklearn import preprocessing

st.set_page_config(
    page_title="OLC2 MACHINE LEARNING", page_icon="📊", initial_sidebar_state="expanded"
)

st.write(
    """
# 📊 OLC2 MACHINE LEARNING
Erwin14k.
"""
)


uploaded_file = st.file_uploader("Upload your Dataset.", type={".csv",".json",".xls",".xlsx"})

def linearRegression(options_in_x,options_in_y,data,date):
    X = np.asarray(data[options_in_x]).reshape(-1, 1)
    Y = data[options_in_y]
    linear_regression = LinearRegression()
    linear_regression.fit(X, Y)
    coeficient=np.array2string(linear_regression.coef_)
    coeficient=coeficient.replace("[","")
    coeficient=coeficient.replace("]","")
    Y_pred = linear_regression.predict(X)
    st.write(f'Mean Error: {mean_squared_error(Y, Y_pred, squared=True)}')
    st.write(f'Regression Coefficient: {coeficient}' )
    st.write(f'R2: {r2_score(Y, Y_pred)}')
    
    Y_new = linear_regression.predict([[(int(date))]])
    result=np.array2string(Y_new)
    result=result.replace("[","")
    result=result.replace("]","")
    st.write(f'Result: {result}')
    st.markdown("### Trend Prediction")
    st.write(Y_pred)
    
    intercept=np.array2string(linear_regression.intercept_)
    intercept=intercept.replace("[","")
    intercept=intercept.replace("]","")
    st.markdown("### Trend Function")
    st.write(f'y= {coeficient}X + {intercept}')
    # =========================== Graphs ===========================================
    st.markdown("### Dot Plot - Sparse Data")
    plt.scatter(X, Y)
    plt.savefig("linealDots.png")
    plt.close()
    image2 = Image.open('linealDots.png')
    st.image(image2, caption='Dot Plot - Sparse Data')
    st.markdown("### Trend Plot")
    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.savefig("trend.png")
    plt.close()
    image = Image.open('trend.png')
    st.image(image, caption='Trend Plot')
    # ================================================================================


def polinomialRegression(degree_datum,options_in_x,options_in_y,data,date):
    X = np.asarray(data[options_in_x]).reshape(-1, 1)
    Y = data[options_in_y]
    pf = PolynomialFeatures(degree = int(degree_datum))
    x_trans = pf.fit_transform(X)
    regr = LinearRegression()
    regr.fit(x_trans, Y)
    y_pred = regr.predict(x_trans)
    rmse = np.sqrt(mean_squared_error(Y, y_pred))
    r2 = r2_score(Y, y_pred)
    st.write(f'RMSE: {rmse}')
    st.write(f'R^2: {r2}')
    st.markdown("### Trend Prediction")
    st.write(y_pred)
    pred = int(date)
    x_new_min = pred
    x_new_max = pred
    x_new = np.linspace(x_new_min, x_new_max, 1)
    x_new = x_new[:, np.newaxis]
    x_trans = pf.fit_transform(x_new)
    st.write(f'Result: {regr.predict(x_trans)}')
    # =========================== Graphs ===========================================
    st.markdown("### Dot Plot - Sparse Data")
    plt.scatter(X, Y, color='green')
    plt.savefig("polynomialDots.png")
    plt.close()
    image = Image.open('polynomialDots.png')
    st.image(image, caption=f'Dot Plot - Sparse Data - Degree: {degree_datum}')
    st.markdown("### Trend Plot")
    plt.scatter(X, Y, color='green')
    plt.plot(X, y_pred, color='blue')
    plt.savefig("trendPoly.png")
    plt.close()
    image2 = Image.open('trendPoly.png')
    st.image(image2, caption=f'Trend Plot - Degree: {degree_datum}')
    # ================================================================================


def decisionTreeClassifier(all_data,data_to_analyze,columns,test_values):
    all_features=[]
    le=preprocessing.LabelEncoder()
    for column in columns:
        if column!=data_to_analyze and column.upper() !="NO":
            temp=all_data[column].tolist()
            temp2=le.fit_transform(temp)
            all_features.append(temp2)
    
    features = list(zip(*all_features) )
    testing=le.fit_transform(test_values)
    clf = DecisionTreeClassifier().fit(features, testing)
    plot_tree(clf, filled=True)
    plt.savefig("tree.png")
    plt.close()
    image2 = Image.open('tree.png')
    st.markdown("### Decision Tree Classifier")
    st.image(image2, caption=f'Decision Tree Classifier')

def gaussianClasiffier(all_data,data_to_analyze,columns,test_values,predicted_values):
    all_features=[]
    le=preprocessing.LabelEncoder()
    for column in columns:
        if column!=data_to_analyze and column.upper() !="NO":
            temp=all_data[column].tolist()
            temp2=le.fit_transform(temp)
            all_features.append(temp2)
    
    features = list(zip(*all_features) )
    testing=le.fit_transform(test_values)
    model=GaussianNB()
    model.fit(features, testing)
    predicted=model.predict([predicted_values])
    st.write(f'Predicted Value: {predicted}')
    

    
    


#Se verifica que si se haya cargado un archivo a la aplicación
if uploaded_file:
    if uploaded_file.type.find("csv") != -1:
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
        elif(option=='polynomial regression'):
            st.markdown("### Polynomial Regression")
            for column_name in keys.columns:
                parameters_of_x.append(column_name)
                parameters_of_y.append(column_name)
            options_in_x = st.selectbox(
                '¿What attribute will be taken in X?',parameters_of_x)
            options_in_y = st.selectbox(
                '¿What attribute will be taken in Y?',parameters_of_y)
            degree_of_prediction = st.text_input('¿What degree do you want for the regression?', 'Ex. 2')
            year_of_prediction = st.text_input('Year Of Prediction', 'Ex. 2023')
            if (options_in_x!='None' and options_in_y!='None' and year_of_prediction!='Ex. 2023' and degree_of_prediction!='Ex. 2'):
                polinomialRegression(degree_of_prediction,options_in_x,options_in_y,df,year_of_prediction)
        elif(option=='Gaussian classifier'):
            st.markdown("### Gaussian classifier")
            for column_name in keys.columns:
                parameters_of_x.append(column_name)
                parameters_of_y.append(column_name)
            options_in_x = st.selectbox(
                '¿What attribute will be taken in to analyze?',parameters_of_x)
            values = st.text_input('Write the predicted values separated by commas.', 'Ex. 2,4,5')
            if (options_in_x!='None' and values !='Ex. 2,4,5'):
                test_values=df[options_in_x].tolist()
                gaussianClasiffier(df,options_in_x,keys.columns,test_values,values.split(","))
        elif(option=='Decision tree classifier'):
            st.markdown("### Decision Tree Classifier")
            for column_name in keys.columns:
                parameters_of_x.append(column_name)
                parameters_of_y.append(column_name)
            options_in_x = st.selectbox(
                '¿What attribute will be taken in to analyze?',parameters_of_x)
            if (options_in_x!='None' ):
                test_values=df[options_in_x].tolist()
                decisionTreeClassifier(df,options_in_x,keys.columns,test_values)
    if uploaded_file.type.find("json") != -1:
        df = pd.read_json(uploaded_file)
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
        elif(option=='polynomial regression'):
            st.markdown("### Polynomial Regression")
            for column_name in keys.columns:
                parameters_of_x.append(column_name)
                parameters_of_y.append(column_name)
            options_in_x = st.selectbox(
                '¿What attribute will be taken in X?',parameters_of_x)
            options_in_y = st.selectbox(
                '¿What attribute will be taken in Y?',parameters_of_y)
            degree_of_prediction = st.text_input('¿What degree do you want for the regression?', 'Ex. 2')
            year_of_prediction = st.text_input('Year Of Prediction', 'Ex. 2023')
            if (options_in_x!='None' and options_in_y!='None' and year_of_prediction!='Ex. 2023' and degree_of_prediction!='Ex. 2'):
                polinomialRegression(degree_of_prediction,options_in_x,options_in_y,df,year_of_prediction)
    if uploaded_file.type.find("xlsx") != -1 or uploaded_file.type.find("xls") != -1:
        df = pd.read_excel(uploaded_file)
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
        elif(option=='polynomial regression'):
            st.markdown("### Polynomial Regression")
            for column_name in keys.columns:
                parameters_of_x.append(column_name)
                parameters_of_y.append(column_name)
            options_in_x = st.selectbox(
                '¿What attribute will be taken in X?',parameters_of_x)
            options_in_y = st.selectbox(
                '¿What attribute will be taken in Y?',parameters_of_y)
            degree_of_prediction = st.text_input('¿What degree do you want for the regression?', 'Ex. 2')
            year_of_prediction = st.text_input('Year Of Prediction', 'Ex. 2023')
            if (options_in_x!='None' and options_in_y!='None' and year_of_prediction!='Ex. 2023' and degree_of_prediction!='Ex. 2'):
                polinomialRegression(degree_of_prediction,options_in_x,options_in_y,df,year_of_prediction)

            



        
        

    