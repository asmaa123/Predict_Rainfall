
import pandas as pd
import joblib as jb
import streamlit as st
import sklearn
# Load your data, encoder, scaler, and model here
df_result = jb.load('selector_result.pkl')
encoder = jb.load('Encoder.pkl')
scalar = jb.load('Scaler.pkl')
model = jb.load('Model.pkl')

def Predict_Rainfall(Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindGustSpeed, WindDir9am, RISK_MM):
    data = pd.DataFrame(columns=df_result.columns)
    data.at[0, 'Location'] = Location
    data.at[0, 'MinTemp'] = MinTemp
    data.at[0, 'MaxTemp'] = MaxTemp
    data.at[0, 'Rainfall'] = Rainfall
    data.at[0, 'Evaporation'] = Evaporation
    data.at[0, 'Sunshine'] = Sunshine
    data.at[0, 'WindGustDir'] = WindGustDir
    data.at[0, 'WindGustSpeed'] = WindGustSpeed
    data.at[0, 'WindDir9am'] = WindDir9am
    data.at[0, 'RISK_MM'] = RISK_MM

    # Encoding and scaling
    data['Location'] = encoder.transform(data[['Location']])
    data['WindGustDir'] = encoder.transform(data[['WindGustDir']])
    data['WindDir9am'] = encoder.transform(data[['WindDir9am']])


    data[df_result.columns] = scalar.transform(data[df_result.columns])

    # Predict rainfall
    result = model.predict(data)
    return result
page_icon_url = 'download.ipg'

st.set_page_config(
    page_icon=page_icon_url,
    page_title='Rainfall Prediction'
)

def main():

    Location = st.selectbox('Location',['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree',
       'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond',
       'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown',
       'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat',
       'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura',
       'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns',
       'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa',
       'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport',
       'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston',
       'AliceSprings', 'Darwin', 'Katherine', 'Uluru'])
    MinTemp = st.number_input('MinTemp', min_value=0.0)
    MaxTemp = st.number_input('MaxTemp', min_value=0.0)
    Rainfall = st.number_input('Rainfall', min_value=0.0)
    Evaporation = st.number_input('Evaporation', min_value=0.0)
    Sunshine = st.number_input('Sunshine', min_value=0.0)
    WindGustDir = st.selectbox('WindGustDir',['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'])
    WindGustSpeed = st.number_input('WindGustSpeed', min_value=0.0)
    WindDir9am = st.selectbox('WindDir9am',['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'])
    RISK_MM = st.number_input('RISK_MM', min_value=0.0)

    if st.button('Predict Rainfall'):
        rainfall = Predict_Rainfall(Location, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustDir, WindGustSpeed, WindDir9am, RISK_MM)
        st.write(f'Predicted Rainfall: {rainfall[0]} mm')

if __name__ == '__main__':
    main()
