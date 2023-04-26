import streamlit as st
import pickle
import numpy as np

st.set_page_config(layout="wide")

st.title('CS, TS and FS Prediction App')

# inputs

col1, col2, col3 = st.columns(3)
with col1:
    st.write('Mix Proportions Ratio')
    fly_ash = st.number_input('Fly Ash C (0 to 4.4)', min_value = 0.0, max_value = 4.4)
    sand2 = st.number_input('Sand Zone 2 (0 to .4)', min_value = 0.0, max_value = .4)
    sand3 = st.number_input('Sand Zone 3 (0 to .3)', min_value = 0.0, max_value = .3)
    sand4 = st.number_input('Sand Zone 4 (0 to 6.9)', min_value = 0.0, max_value = 6.9)
    limestone = st.number_input('Limestone (0 to 3.3)', min_value = 0.0, max_value = 3.3)
    bsf = st.number_input('Blast Furnace Slag (0 to 2.3)', min_value = 0.0, max_value = 2.3)
    silica_fume = st.number_input('Silica Fume (0 to 0.75)', min_value = 0.0, max_value = 0.75)
    w_b = st.number_input('Water/Binder (0.04 to 3.6)', min_value = 0.04, max_value = 3.6)
    sp = st.number_input('Superplasticizer (0 to .9)', min_value = 0.0, max_value = .9)
with col2:
    st.write('PVA Fiber Properties')
    fibre_length = st.number_input('Fibre Length(mm) (6 to 20)', min_value = 6.0, max_value = 20.0)
    fibre_volume = st.number_input('Fibre Volume(%) (0 to 8.71)', min_value = 0.0, max_value = 8.71)
    fibre_dia = st.number_input('Fibre Dia(Um) (12 to 80)', min_value = 12.0, max_value = 80.0)
    fibre_density = st.number_input('Fibre Density(Kg/M3) (970 to 1846)', min_value = 970.0, max_value = 1846.0)
    fibre_elasticity = st.number_input('Fibre Elasticity(Mpa) (16.9 to 66)', min_value = 16.9, max_value = 66.0)

features = np.array([fly_ash, sand2, sand3, sand4, limestone, bsf, silica_fume, w_b, sp, fibre_length, fibre_volume, fibre_elasticity, fibre_dia, fibre_density])
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
