import streamlit as st
import pickle
import numpy as np

st.set_page_config(layout="wide")

st.title('Micromechanical Properties Prediction Application')

# inputs

col1, col2, col3 = st.columns(3)

fly_ash_type_dict = {"No Fly Ash": 0, "Class C": 1, "Class F": 2, "Grade I": 3}
sand_type_dict = {"Silica Sand": 1, "Crushed Sand": 2, "Gravel Sand": 3, "River Sand": 5}

with col1:
    st.write('Mix Proportions Ratio')
    fly_ash = st.number_input('Fly Ash (0 to 4.4)', min_value = 0.0, max_value = 4.4)
    fly_ash_type_label = st.selectbox('Fly Ash Type', options=list(fly_ash_type_dict.keys()))
    fly_ash_type = fly_ash_type_dict[fly_ash_type_label]
    sand = st.number_input('Sand(0.7 to 2)', min_value = 0.0, max_value = 2.0)
    sand_type_label = st.selectbox('Sand Type', options=list(sand_type_dict.keys()))
    sand_type = sand_type_dict[sand_type_label]
    avg_sand_size = st.number_input('Average Sand Size(µm) (0 to 943)', min_value = 0.0, max_value = 6.9)
    max_sand_size = st.number_input('Max Sand Size(µm) (0 to 4750)', min_value = 0.0, max_value = 3.3)
    w_b = st.number_input('Water/Binder (0.16 to .87)', min_value = 0.16, max_value = .87)
    sp = st.number_input('Superplasticizer/Binder (.12 to 2.72)', min_value = 0.12, max_value = 2.72)
    
with col2:
    st.write('PVA Fiber Properties')
    fibre_length = st.number_input('Fibre Length(mm) (8 to 18)', min_value = 8.0, max_value = 18.0)
    fibre_volume = st.number_input('Fibre Volume(%) (1 to 2.5)', min_value = 1.0, max_value = 2.5)
    fibre_dia = st.number_input('Fibre Diameter(µm) (34 to 40)', min_value = 34.0, max_value = 40.0)

features = np.array([fly_ash, fly_ash_type, sand, sand_type, avg_sand_size,max_sand_size, w_b, sp, fibre_length, fibre_volume,  fibre_dia])
features = features.reshape(1, -1)

with col3:
    st.write('Predictions')

    if st.button('Predict'):
        model = pickle.load(open('cs.sav', 'rb'))
        pred = model.predict(features)
        st.write(f'CS value: {pred[0]}')
        
        model = pickle.load(open('ts.sav', 'rb'))
        pred = model.predict(features)
        st.write(f'TS value: {pred[0]}')
        
        model = pickle.load(open('fs.sav', 'rb'))
        pred = model.predict(features)
        st.write(f'FS value: {pred[0]}')
