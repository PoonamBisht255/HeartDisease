# importing basic library

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Importing models

from sklearn.linear_model import LogisticRegression
import pickle

# model Evaluation
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import plot_roc_curve





def main():
    df = pd.read_csv('Data/farmingham.csv')
    df.drop(axis=1, columns=['education'], inplace=True)
    df.rename(columns={'male': 'gender'}, inplace=True)

    # filling glucose null values with median of the columns
    df['glucose'] = df['glucose'].fillna(df['glucose'].median())
    df.dropna(inplace=True)

    scaler = StandardScaler()

    x = df.iloc[:, 0:14].values
    scaler.fit(x)
    x = scaler.transform(x)
    y = df['TenYearCHD'].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=11)

    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)













    st.title('Heart Disease')
    Gender = str(st.selectbox('Gender:',('Male','Female')))
    if(Gender=="Yes"):
        Gender=1
    else:
        Gender=0
    Age = int(st.number_input('Age:',format='%d',step=None))
    Smoke = str(st.selectbox('Smoker:',('Yes','No')))
    if(Smoke=="Yes"):
        Smoke=1
    else:
        Smoke=0


    Ciggerates = st.number_input('Ciggerates per day:',step=None)
    Bp = str(st.selectbox('BP Medication ?',('Yes','No')))
    if(Bp=="Yes"):
        Bp=1
    else:
        Bp=0
    stroke = str(st.selectbox('previously had a stroke ?',('Yes','No')))
    if(stroke=="Yes"):
        stroke=1
    else:
        stroke=0
    hyp = str(st.selectbox('Is the patient hypertensive ?',('Yes','No')))
    if(hyp=="Yes"):
        hyp=1
    else:
        hyp=0
    diab = str(st.selectbox('Is the patient diabetic ?',('Yes','No')))
    if(diab=="Yes"):
        diab=1
    else:
        diab=0
    totchol = st.number_input('total cholesterol level:',step=None)
    sbp = st.number_input('systolic blood pressure:',step=None)
    dbp = st.number_input('diastolic blood pressure:',step=None)
    bmi = st.number_input('BMI:',step=None)
    HeartRate = st.number_input('HeartRate:',step=None)
    glucose = st.number_input('Glucose Level:',step=None)
    if st.button("submit"):
        output=model.predict([[int(Gender),int(Age),int(Smoke),float(Ciggerates),float(Bp),int(stroke),int(hyp),int(diab),float(totchol),float(sbp),float(dbp),float(bmi),float(HeartRate),float(glucose)]])
        if output[0]==1:
            Output="The Patient is suffering from heart disease!!"
        else:
            Output = "The Patient is healthy!!"
        st.markdown(Output)
        pickle.dump(model, open('model.pkl', 'wb'))









if __name__ == '__main__':
	main()
