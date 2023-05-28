import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import base64


xls = pd.ExcelFile('Final Data.xlsx')
df_cs = pd.read_excel(xls, 'CS')
df_ts = pd.read_excel(xls, 'TS')
df_fs = pd.read_excel(xls, 'FS')

df_cs = df_cs.drop(['Ref.', 'S no', 'Aspect Ratio', 'Cement'], axis=1)
df_ts = df_ts.drop(['S no', 'Ref.', 'Aspect Ratio', 'Cement'], axis=1)
df_fs = df_fs.drop(['S no', 'Ref.', 'Aspect Ratio','Cement','Silica Fume'], axis=1)

cs_x, ts_x, fs_x = df_cs.drop(['Compressive Strength (Mpa)'], axis=1), df_ts.drop(['Tensile Strain (%)'], axis=1), df_fs.drop(['Flexural Strength (Mpa)'], axis=1)

scaler_cs = StandardScaler()
scaler_ts = StandardScaler()
scaler_fs = StandardScaler()
cs_x, ts_x, fs_x = scaler_cs.fit_transform(cs_x), scaler_ts.fit_transform(ts_x), scaler_fs.fit_transform(fs_x)

n_components_cs = 17
pca_cs = PCA(n_components=n_components_cs)
pca_data_cs = pca_cs.fit_transform(cs_x)

n_components_ts = 15
n_components_fs = 13
pca_ts = PCA(n_components=n_components_ts)
pca_fs = PCA(n_components=n_components_fs)
pca_data_ts = pca_ts.fit_transform(ts_x)
pca_data_fs = pca_fs.fit_transform(fs_x)

st.set_page_config(
    page_title="Micromechanical Properties Prediction",
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="expanded",
)
iit_logo = "iitr.jpg"
image = open(iit_logo, "rb").read()
st.markdown(
    f'<img src="data:image/jpg;base64,{base64.b64encode(image).decode("utf-8")}" '
    'style="width: 1150px; height: 190px;">',
    unsafe_allow_html=True
)
iit_175_logo = "logo_175.png"
image = open(iit_175_logo, "rb").read()
st.markdown(
    f"""
    <div style="position: absolute; top: -200px; right: 25px;">
        <img src="data:image/jpg;base64,{base64.b64encode(image).decode("utf-8")}" style="width: 340px; height: 100px;">
    </div>
    """,
    unsafe_allow_html=True
)

st.title('🧱 ECC Micromechanical Properties Prediction Application')

st.write("""
In this application, you can predict the Compressive Strength (CS), Tensile Strain (TS), and Flexural Strength (FS) 
of a material based on its properties. 
Please input the required material properties in the fields below to get the predictions.
""")

model_cs = pickle.load(open('cs.sav', 'rb'))
model_ts = pickle.load(open('ts.sav', 'rb'))
model_fs = pickle.load(open('fs.sav', 'rb'))

col1, col2, col3 = st.columns(3)

fly_ash_type_dict = {"No Fly Ash": 0, "Class C": 1, "Class F": 2, "Grade I": 3}
sand_type_dict = {"Silica Sand": 1, "Crushed Sand": 2, "Gravel Sand": 3, "Dune Sand" : 4, "River Sand": 5}
fiber_type_dict = {"PVA Fiber" : 1, "PE Fiber" : 2}

with col1:
    st.header('📊 Mix Proportions Ratio')
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
    st.header('🔬 PVA Fiber Properties')
    fiber_type_label = st.selectbox('Fiber Type', options=list(fiber_type_dict.keys()))
    fiber_type = fiber_type_dict[fiber_type_label]
    fibre_length = st.number_input('Fibre Length(mm) (8 to 18)', min_value = 8.0, max_value = 18.0)
    fibre_volume = st.number_input('Fibre Volume(%) (0.41 to 3)', min_value = 0.41, max_value = 3.0)
    fibre_elasticity = st.number_input('Fibre Elasticity(Gpa) (10 to 116)', min_value = 10.0, max_value = 116.0)
    fibre_dia = st.number_input('Fibre Diameter(µm) (24 to 200)', min_value = 24.0, max_value = 200.0)
    tensile_strength = st.number_input('Tensile Strength(Mpa) (1275 to 3000)', min_value = 1275.0, max_value = 3000.0)
    fibre_density = st.number_input('Fibre Density(Kg/m3) (970 to 1600)', min_value = 970.0, max_value = 1600.0)

features_cs = np.array([fly_ash, fly_ash_type, sand, sand_type, avg_sand_size, max_sand_size, limestone, limestone_max_size, bfs, silica_fume, w_b, sp, fiber_type, fibre_length, fibre_volume, fibre_elasticity, fibre_dia, tensile_strength, fibre_density ])
features_ts = np.array([fly_ash, fly_ash_type, sand, sand_type, avg_sand_size, max_sand_size, limestone, limestone_max_size, bfs, silica_fume, w_b, sp, fiber_type, fibre_length, fibre_volume, fibre_elasticity, fibre_dia, tensile_strength, fibre_density ])
features_fs = np.array([fly_ash, fly_ash_type, sand, sand_type, avg_sand_size, max_sand_size, limestone, limestone_max_size, bfs, w_b, sp, fiber_type, fibre_length, fibre_volume, fibre_elasticity, fibre_dia, tensile_strength, fibre_density ])

scaled_features_cs = scaler_cs.transform(features_cs.reshape(1, -1))
scaled_features_ts = scaler_ts.transform(features_ts.reshape(1, -1))
scaled_features_fs = scaler_fs.transform(features_fs.reshape(1, -1))

pca_features_cs = pca_cs.transform(scaled_features_cs)
pca_features_ts = pca_ts.transform(scaled_features_ts)
pca_features_fs = pca_fs.transform(scaled_features_fs)

prediction_cs = model_cs.predict(pca_features_cs)
prediction_ts = model_ts.predict(pca_features_ts)
prediction_fs = model_fs.predict(pca_features_fs)

with col3:
    st.header('📝 Prediction')
    if st.button('Predict'):
        st.write(f'Compressive Strength (CS): {prediction_cs[0]} Mpa')
        st.write(f'Tensile Strain (TS): {prediction_ts[0]} %')
        st.write(f'Flexural Strength (FS): {prediction_fs[0]} Mpa')
