import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import seaborn as sns
import numpy as np
st.write("""
# Simple TB death Prediction App
This app predicts the **Death during tuberculosis** 
""")

st.sidebar.header('User Input Parameters(please select patients features here)')
def user_input_features():
    age = st.sidebar.slider('age', 15, 100)
    weight = st.sidebar.slider('weight', 20, 120)
    hiv = st.sidebar.selectbox('hiv', ('positive','unknown','negative'))
    form = st.sidebar.selectbox('form',('extrapulmonary','smear_negative_pulmonary', 'smear_positive_pulmonary'))
    data = {'age': age,
            'weight': weight,
            'hiv': hiv,
            'form':form}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

deathTB= pd.read_csv('https://raw.githubusercontent.com/pefura/Prediction-tuberculosis-death/streamlit_deathTB/deathTB.csv',sep=';', header=0)
# Slectionner les prédicteurs et la variable réponse
y = deathTB['death']
X = deathTB.drop('death', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Lister les types de variables de chaque catégorie
numerical_features = ['age', 'weight']
categorical_features = ['hiv', 'form']

# Construire les pipelines qui vont permettre de faire les transformations succèssives de chaque type de variables
numerical_pipeline = make_pipeline(SimpleImputer(strategy='mean'), RobustScaler())
categorical_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder())

preprocessor = make_column_transformer((numerical_pipeline, numerical_features),
                                   (categorical_pipeline, categorical_features))
# Model fit
model = make_pipeline(preprocessor,SGDClassifier(loss='log'))

model.fit(X_train, y_train)
model.score(X_test, y_test)

# Prediction
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)
prediction_proba_percent=prediction_proba*100

st.subheader('Probability of death(%), death is coded 1')

st.write(prediction_proba_percent)

st.subheader('High risk of death')
st.write(prediction)
#st.write(prediction)




