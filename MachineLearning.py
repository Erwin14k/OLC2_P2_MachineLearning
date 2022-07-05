#============================IMPORTS====================================
#=======================================================================
from base64 import standard_b64encode
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
#=======================================================================
#=======================================================================

st.set_page_config(
    page_title="OLC2 MACHINE LEARNING", page_icon="📊", initial_sidebar_state="expanded"
)

st.write(
    """
# 📊 OLC2 MACHINE LEARNING
I'm Erwin Vásquez - aka Erwin14k 👋
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
    if(intercept.find("-")!=-1):
        st.write(f'y= {coeficient}$X$ {intercept}')
    else:
        st.write(f'y= {coeficient}$X$ + {intercept}')

    
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
    #st.write(regr.coef_)
    counter=int(degree_datum)
    equation_result="y="
    for temp in np.flip(regr.coef_):
        if(counter>1):
            if(str(temp).find("-")!=-1):
                equation_result+=f'{temp}$X^{counter}$ '
                counter=counter-1
            else:
                equation_result+=f'+{temp}$X^{counter}$ '
                counter=counter-1
        elif(counter==1):
            if(str(temp).find("-")!=-1):
                equation_result+=f'{temp}$X$ '
                counter=counter-1
            else:
                equation_result+=f'+{temp}$X$ '
                counter=counter-1
    intercept=np.array2string(regr.intercept_)
    intercept=intercept.replace("[","")
    intercept=intercept.replace("]","")
    if(intercept.find("-")!=-1):
        equation_result+=f'{intercept}'
    else:
        equation_result+=f'+{intercept}'

    y_pred = regr.predict(x_trans)
    rmse = np.sqrt(mean_squared_error(Y, y_pred))
    r2 = r2_score(Y, y_pred)
    st.write(f'RMSE: {rmse}')
    st.write(f'R^2: {r2}')
    st.markdown("### Trend Prediction")
    st.write(y_pred)
    st.markdown("### Trend Function")
    st.write(f'{equation_result}')
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

def gaussianClasiffier(all_data,data_to_analyze,columns,test_values,predicted_values):
    all_features=[]
    le=preprocessing.LabelEncoder()
    model=GaussianNB()
    for column in columns:
        if column!=data_to_analyze and column.upper() !="NO":
            temp=all_data[column].tolist()
            temp2=le.fit_transform(temp)
            all_features.append(temp2)
    
    features = list(zip(*all_features) )
    testing=le.fit_transform(test_values)
    
    model.fit(features, testing)
    final_values=[]
    temp_list=[]
    
    for temp in predicted_values:
        temp_list.append(int(temp))
    final_values.append(temp_list)
    #st.write(final_values)
    predicted=model.predict(final_values)
    predicted=le.inverse_transform(predicted)
    predicted=np.array2string(predicted)
    predicted=predicted.replace("[","")
    predicted=predicted.replace("]","")
    
    
    st.write(f'Predicted Value: {predicted}')

def decisionTreeClassifier(all_data,data_to_analyze,columns,test_values,predicted_values):
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
    # =========================== Graphs ===========================================
    plot_tree(clf, filled=True)
    plt.savefig("tree.png")
    plt.close()
    image2 = Image.open('tree.png')
    st.markdown("### Decision Tree Classifier")
    st.image(image2, caption=f'Decision Tree Classifier')
    final_values=[]
    temp_list=[]
    for temp in predicted_values:
        temp_list.append(int(temp))
    final_values.append(temp_list)
    prediction=clf.predict(final_values)
    prediction=le.inverse_transform(prediction)
    st.write(f'Prediction: {prediction}')
    # ================================================================================


def neuralNetworks(all_data,data_to_analyze,columns,test_values,predicted_values):
    all_features=[]
    le=preprocessing.LabelEncoder()
    for column in columns:
        if column!=data_to_analyze:
            temp=all_data[column].tolist()
            temp2=le.fit_transform(temp)
            all_features.append(temp2)
    features = list(zip(*all_features) )
    testing=le.fit_transform(test_values)
    #x_train,x_test,y_train=train_test_split(all_features,testing)
    mlp=MLPClassifier(hidden_layer_sizes=(100,100,100),max_iter=1000, alpha=0.0001,solver='adam',random_state=21,tol=0.0000000001)
    mlp.fit(features,testing)
    predictions=mlp.predict(testing)
    #st.write(classification_report(y_test,predictions))
    st.write(predictions)



    

    
    


#It is verified that if a file has been uploaded to the application
if uploaded_file:
    if uploaded_file.type.find("csv") != -1:
        df = pd.read_csv(uploaded_file)
        st.markdown("### Dataset preview")
        st.write(df)
        keys=df.head()
        #A selectBox is made visible with all the available algorithms to perform
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
                values=values.split(",")
                le=preprocessing.LabelEncoder()
                values=le.fit_transform(values)
                #st.write(values)
                gaussianClasiffier(df,options_in_x,keys.columns,test_values,values)
        elif(option=='Decision tree classifier'):
            st.markdown("### Decision Tree Classifier")
            for column_name in keys.columns:
                parameters_of_x.append(column_name)
                parameters_of_y.append(column_name)
            options_in_x = st.selectbox(
                '¿What attribute will be taken in to analyze?',parameters_of_x)
            values = st.text_input('Write the predicted values separated by commas.', 'Ex. 2,4,5')
            if (options_in_x!='None' and values !='Ex. 2,4,5' ):
                test_values=df[options_in_x].tolist()
                values=values.split(",")
                le=preprocessing.LabelEncoder()
                values=le.fit_transform(values)
                decisionTreeClassifier(df,options_in_x,keys.columns,test_values,values)
        elif(option=='neural networks'):
            st.markdown("### Neural Networks")
            for column_name in keys.columns:
                parameters_of_x.append(column_name)
                parameters_of_y.append(column_name)
            options_in_x = st.selectbox(
                '¿What attribute will be taken in to analyze?',parameters_of_x)
            values = st.text_input('Write the predicted values separated by commas.', 'Ex. 2,4,5')
            if (options_in_x!='None' and values !='Ex. 2,4,5' ):
                test_values=df[options_in_x].tolist()
                values=values.split(",")
                le=preprocessing.LabelEncoder()
                values=le.fit_transform(values)
                neuralNetworks(df,options_in_x,keys.columns,test_values,values)
    if uploaded_file.type.find("json") != -1:
        df = pd.read_json(uploaded_file)
        st.markdown("### Dataset preview")
        st.write(df)
        keys=df.head()
        #A selectBox is made visible with all the available algorithms to perform
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
        #A selectBox is made visible with all the available algorithms to perform
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

            



        
        

    