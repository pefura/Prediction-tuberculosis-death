import streamlit as st
import pandas as pd
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
# # Machine Learning TB death Prediction App

This app predicts the risk of **Death during tuberculosis** 

By Pefura-Yone et al. (BMC Infect Dis. doi: 10.1186/s12879-017-2309-9)** 
""")

st.sidebar.header('User Input Parameters(please select patients features here)')


def user_input_features():
    age = st.slider('age', 15, 100)
    weight = st.slider('weight', 20, 120)
    hiv = st.selectbox('hiv', ('positive', 'unknown', 'negative'))
    form = st.selectbox('form', ('smear_positive_pulmonary', 'smear_negative_pulmonary', 'extrapulmonary'))
    data = {'age': age,
            'weight': weight,
            'form': form,
            'hiv': hiv}

    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

deathTB = pd.read_csv('https://raw.githubusercontent.com/pefura/Prediction-tuberculosis-death/streamlit_deathTB/deathTB_2.csv', sep=';', header=0)

deathTB.death = deathTB.death.astype(str)
deathTB.form = deathTB.form.astype(str)
deathTB.hiv = deathTB.hiv.astype(str)

deathTB[["age"]] = deathTB[["age"]].apply(pd.to_numeric, errors='coerce')
deathTB[["weight"]] = deathTB[["weight"]].apply(pd.to_numeric, errors='coerce')

# Slectionner les prédicteurs et la variable réponse
y = deathTB['death']
X = deathTB.drop('death', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Lister les types de variables de chaque catégorie
numerical_features = ['age', 'weight']
categorical_features = ['hiv', 'form']

# Construire les pipelines qui vont permettre de faire les transformations succèssives de chaque type de variables
numerical_pipeline = make_pipeline(SimpleImputer(strategy='median'), RobustScaler())
categorical_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder())

preprocessor = make_column_transformer((numerical_pipeline, numerical_features),
                                       (categorical_pipeline, categorical_features))
# Model fit
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

model = make_pipeline(preprocessor, SGDClassifier(loss='log',random_state=100))
model.fit(X_train, y_train)
model.score(X_test, y_test)


# Prediction
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)
prediction_proba_percent = prediction_proba * 100
proba = prediction_proba[:, 1]
prediction_proba_percent = proba * 100

st.subheader('Probability of death(%)')
st.write(prediction_proba_percent)

probability_risk= {"low risk":"probability < 5%", "moderate risk": "probability  5-15%", "high risk": "probability >15%"}


#st.write(prediction)
st.subheader('Risk of death')
st.write( '''
low risk: probability < 5%

intermediate risk: probability 5-15%

high risk: probability >15% ''')
