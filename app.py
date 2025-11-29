
import streamlit as st
import numpy as np
import joblib

# Cargar modelo y escalador
modelo = joblib.load("modelo_aids_rf.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Predicción de VIH", layout="centered")

st.title("🧬 Predicción de Infección por VIH")
st.write("Modelo de Machine Learning - Random Forest")

st.subheader("Ingrese los datos del paciente")

cd40 = st.number_input("CD4 inicial", 0, 2000, value=350)
cd420 = st.number_input("CD4 seguimiento", 0, 2000, value=300)
cd80 = st.number_input("CD8 inicial", 0, 2000, value=400)
cd820 = st.number_input("CD8 seguimiento", 0, 2000, value=380)
preanti = st.number_input("Pretratamiento", 0, 5000, value=0)

gender = st.selectbox("Género (0=F, 1=M)", [0, 1])
race = st.selectbox("Raza (0/1)", [0, 1])
treat = st.selectbox("Tratamiento (0=No,1=Si)", [0, 1])
symptom = st.selectbox("Síntomas (0=No,1=Si)", [0, 1])
offtrt = st.selectbox("Abandono tratamiento (0=No,1=Si)", [0, 1])

if st.button("Predecir"):
    datos = np.array([[cd40, cd420, cd80, cd820, preanti,
                        gender, race, treat, symptom, offtrt]])
    
    datos = scaler.transform(datos)
    pred = modelo.predict(datos)[0]
    prob = modelo.predict_proba(datos)[0][1]

    if pred == 1:
        st.error(f"Paciente INFECTADO ✅ (Probabilidad: {prob:.2%})")
    else:
        st.success(f"Paciente NO infectado ❎ (Probabilidad: {prob:.2%})")
