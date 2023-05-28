
import streamlit as st
import pickle
import numpy as np


st.set_page_config(layout="wide")

st.title('Micromechanical Properties Prediction Application')

# Load scaler, PCA, and models
scaler_cs = pickle.load(open('scaler_cs.pkl', 'rb'))
scaler_ts = pickle.load(open('scaler_ts.pkl', 'rb'))
scaler_fs = pickle.load(open('scaler_fs.pkl', 'rb'))
pca_cs = pickle.load(open('pca_cs.pkl', 'rb'))
pca_ts = pickle.load(open('pca_ts.pkl', 'rb'))
pca_fs = pickle.load(open('pca_fs.pkl', 'rb'))
model_cs = pickle.load(open('cs.sav', 'rb'))
model_ts = pickle.load(open('ts.sav', 'rb'))
model_fs = pickle.load(open('fs.sav', 'rb'))

# inputs
col1, col2, col3 = st.columns(3)

fly_ash_type_dict = {"No Fly Ash": 0, "Class C": 1, "Class F": 2, "Grade I": 3}
sand_type_dict = {"Silica Sand": 1, "Crushed Sand": 2, "Gravel Sand": 3, "Dune Sand" : 4, "River Sand": 5}
fiber_type_dict = {"PVA Fiber" : 1, "PE Fiber" : 2}

with col1:
    st.write('Mix Proportions Ratio')
    fly_ash = st.number_input('Fly Ash (0 to 4.4)', min_value = 0.0, max_value = 4.4)
    fly_ash_type_label = st.selectbox('Fly Ash Type', options=list(fly_ash_type_dict.keys()))
    fly_ash_type = fly_ash_type_dict[fly_ash_type_label]
    sand = st.number_input('Sand(0 to 2.2)', min_value = 0.0, max_value = 2.2)
    sand_type_label = st.selectbox('Sand Type', options=list(sand_type_dict.keys()))
    sand_type = sand_type_dict[sand_type_label]
    avg_sand_size = st.number_input('Average Sand Size(µm) (0 to 943)', min_value = 0.0, max_value = 943.0)
    max_sand_size = st.number_input('Max Sand Size(µm) (0 to 4750)', min_value = 0.0, max_value = 4750.00)
    limestone = st.number_input('Limestone (0 to 3.3)', min_value = 0.0, max_value = 3.3)
    limestone_max_size = st.number_input('Limestone Max Size (0 to 300)', min_value = 0.0, max_value = 300.0)
    bfs = st.number_input('Blast Furnace Slag (0 to 2.3)', min_value = 0.0, max_value = 2.3)
    silica_fume = st.number_input('Silica Fume (0 to 0.58)', min_value = 0.0, max_value = 0.58)
    w_b = st.number_input('Water/Binder (0.12 to 0.61)', min_value = 0.12, max_value = 0.61)
    sp = st.number_input('Superplasticizer/Binder (0 to 2.5)', min_value = 0.0, max_value = 2.5)

with col2:
    st.write('PVA Fiber Properties')
    fiber_type_label = st.selectbox('Fiber Type', options=list(fiber_type_dict.keys()))
    fiber_type = fiber_type_dict[fiber_type_label]
    fibre_length = st.number_input('Fibre Length(mm) (8 to 18)', min_value = 8.0, max_value = 18.0)
    fibre_volume = st.number_input('Fibre Volume(%) (0.41 to 3)', min_value = 0.41, max_value = 3)
    fibre_elasticity = st.number_input('Fibre Elasticity(Gpa) (10 to 116)', min_value = 10.0, max_value = 116.0)
    fibre_dia = st.number_input('Fibre Diameter(µm) (24 to 200)', min_value = 24.0, max_value = 200.0)
    tensile_strength = st.number_input('Tensile Strength(Mpa) (1275 to 3000)', min_value = 1275.0, max_value = 3000.0)
    fibre_density = st.number_input('Fibre Density(Kg/m3) (970 to 1600)', min_value = 970.0, max_value = 1600.0)

# Data pre-processing and prediction
features_cs = np.array([fly_ash, fly_ash_type, sand, sand_type, avg_sand_size, max_sand_size, limestone, limestone_max_size, bfs, silica_fume, w_b, sp, fiber_type, fibre_length, fibre_volume, fibre_elasticity, fibre_dia, tensile_strength, fibre_density ])
features_ts = np.array([fly_ash, fly_ash_type, sand, sand_type, avg_sand_size, max_sand_size, limestone, limestone_max_size, bfs, silica_fume, w_b, sp, fiber_type, fibre_length, fibre_volume, fibre_elasticity, fibre_dia, tensile_strength, fibre_density ])
features_fs = np.array([fly_ash, fly_ash_type, sand, sand_type, avg_sand_size, max_sand_size, limestone, limestone_max_size, bfs, w_b, sp, fiber_type, fibre_length, fibre_volume, fibre_elasticity, fibre_dia, tensile_strength, fibre_density ])

scaled_features_cs = scaler_cs.transform(features_cs.reshape(1, -1))
scaled_features_ts = scaler_ts.transform(features_ts.reshape(1, -1))
scaled_features_fs = scaler_fs.transform(features_fs.reshape(1, -1))

pca_features_cs = pca_cs.transform(scaled_features_cs)
pca_features_ts = pca_ts.transform(scaled_features_ts)
pca_features_fs = pca_fs.transform(scaled_features_fs)

with col3:
    st.write('Predictions')

    if st.button('Predict'):
        pred_cs = model_cs.predict(pca_features_cs)
        st.write(f'CS value: {pred_cs[0]}')

        pred_ts = model_ts.predict(pca_features_ts)
        st.write(f'TS value: {pred_ts[0]}')

        pred_fs = model_fs.predict(pca_features_fs)
        st.write(f'FS value: {pred_fs[0]}')
