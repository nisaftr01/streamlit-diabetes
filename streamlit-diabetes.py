import pickle
import streamlit as st
import numpy as np # Kita butuh numpy untuk reshape

# 1. Membaca Model
diabete_model = pickle.load(open('diabetes_model.sav', 'rb'))

# 2. Membaca Scaler (YANG BARU)
scaler = pickle.load(open('scaler.sav', 'rb'))

# Judul web
st.title('Data Mining Prediksi Diabetes')

# Input Data
col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.text_input('Input Nilai Pregnancies')
    Glucose = st.text_input('Input Nilai Glucose')
    BloodPressure = st.text_input('Input Nilai Blood Pressure')
    SkinThickness = st.text_input('Input Nilai Skin Thickness')

with col2:
    Insulin = st.text_input('Input Nilai Insulin')
    BMI = st.text_input('Input Nilai BMI')
    DiabetesPedigreeFunction = st.text_input('Input Nilai Diabetes Pedigree Function')
    Age = st.text_input('Input Nilai Age')

# Code untuk prediksi
diab_diagnosis = ''

if st.button('Test Prediksi Diabetes'):
    try:
        # Ubah input menjadi list angka float
        input_data = [
            float(Pregnancies), float(Glucose), float(BloodPressure), 
            float(SkinThickness), float(Insulin), float(BMI), 
            float(DiabetesPedigreeFunction), float(Age)
        ]
        
        # Ubah menjadi numpy array
        input_data_as_numpy_array = np.array(input_data)
        
        # Reshape array (agar bentuknya (1, -1) sesuai permintaan model)
        input_data_reshape = input_data_as_numpy_array.reshape(1, -1)
        
        # --- STANDARDISASI DATA (PENTING!) ---
        # Gunakan scaler yang sudah di-load untuk mengubah data mentah jadi data standar
        std_data = scaler.transform(input_data_reshape)
        
        # Lakukan prediksi menggunakan data yang sudah di-standarisasi
        diab_prediction = diabete_model.predict(std_data)
        
        if(diab_prediction[0] == 0): # Sesuaikan dengan label di dataset (biasanya 0=Tidak, 1=Ya)
            diab_diagnosis = 'Pasien TIDAK terkena Diabetes'
            st.success(diab_diagnosis)
        else:
            diab_diagnosis = 'Pasien TERKENA Diabetes'
            st.warning(diab_diagnosis)
            
    except ValueError:
        st.error("Harap isi semua kolom dengan angka yang valid.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")