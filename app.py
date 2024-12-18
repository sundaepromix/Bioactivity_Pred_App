import streamlit as st
import pandas as pd
from PIL import Image
import subprocess
import os
import base64
import pickle
from sklearn.feature_selection import VarianceThreshold

# Molecular descriptor calculator
def desc_calc():
    # Performs the descriptor calculation
    bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml -dir ./ -file descriptors_output.csv"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    os.remove('molecule.smi')

# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href

# Model building
def build_model(input_data):
    # Reads in saved regression model
    load_model = pickle.load(open('enoyl_acyl_carrier_protein_reductase_model.pkl', 'rb'))
    # Apply model to make predictions
    prediction = load_model.predict(input_data)
    st.header('**Prediction output**')
    prediction_output = pd.Series(prediction, name='pIC50')
    molecule_name = pd.Series(load_data.iloc[:,0], name='molecule_name')  # Get first column
    df = pd.concat([molecule_name, prediction_output], axis=1)
    st.write(df)
    st.markdown(filedownload(df), unsafe_allow_html=True)

# Page title
st.markdown("""
# Bioactivity Prediction App (Enoyl_Acyl_Carrier_Protein_Reductase)
This app allows you to predict the bioactivity towards inhibting the `Enoyl Acyl Carrier Protein Reductase` enzyme. `Enoyl Acyl Carrier Protein Reductase` is a drug target for Tuberculosis's disease.
""")

# Sidebar
st.sidebar.header('1. Upload your CSV data')
uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])

if st.sidebar.button('Predict'):
    load_data = pd.read_table(uploaded_file, sep=' ', header=None)
    load_data.to_csv('molecule.smi', sep = '\t', header = False, index = False)

    st.header('**Original input data**')
    st.write(load_data)

    with st.spinner("Calculating descriptors..."):
        desc_calc()

    # Read in calculated descriptors and display the dataframe
    st.header('**Calculated molecular descriptors**')
    desc = pd.read_csv('Bioactivity data folder/descriptors_output.csv')
    
    # Apply the same feature selection as during training
    selection = VarianceThreshold(threshold=(.8 * (1 - .8)))    
    X = selection.fit_transform(desc.iloc[:,1:])  # Exclude 'Name' column
    
    st.write(desc)
    st.write(desc.shape)

    # Show the subset of descriptors
    st.header('**Subset of descriptors from previously built models**')
    st.write(X)
    st.write(X.shape)

    # Apply trained model to make prediction on query compounds
    build_model(X)
else:
    st.info('Upload input data in the sidebar to start!')