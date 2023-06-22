import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl','rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data['model']
enc_cn = data['enc_cn']
enc_ed = data['enc_ed']

def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some information to predict the salary """)
    countries = {                                                
        "United States of America",                                
        "Germany",                                                 
        "United Kingdom of Great Britain and Northern Ireland",    
        "India",                                                  
        "Canada",                                                  
        "France",                                                  
        "Brazil",                                                  
        "Spain",                                                    
        "Netherlands",                                              
        "Australia",                                                
        "Italy",                                                    
        "Poland",                                                   
        "Sweden",                                                   
        "Russian Federation",                                       
        "Switzerland",
        "Others"
    }
    education = {
        "Master's degree", 
        "Bachelor's degree", 
        'Less than a Bachelors',
        'Post grad'
    }
    country = st.selectbox("Country",countries)
    educations = st.selectbox("Eductaion Level",education)
    experience = st.slider("Years of Experience",0,50,3)
    ok = st.button("Calculate Salary")
    if ok:
        x = np.array([[country,educations,experience ]])
        x[:, 0] = enc_cn.transform(x[:,0])
        x[:, 1] = enc_ed.transform(x[:,1])
        x= x.astype(float)
        salary = regressor.predict(x)
        st.subheader(f"The estimated salary in 2023 is ${salary[0]:.2f}")
