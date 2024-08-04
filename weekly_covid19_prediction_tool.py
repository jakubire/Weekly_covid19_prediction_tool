import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional

# Load the saved model
model = tf.keras.models.load_model('my_usa_cv19_model.keras')

# Set the background color of the app to dark
st.markdown(
    """
    <style>
    .main {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to split the sequence
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Streamlit app title
st.title('COVID-19 Weekly Predictions')

# User inputs for the last three weeks of data
st.subheader('Enter the last three weeks of data')

week1_cases = st.number_input('Week 1 - Daily New Cases')
week1_deaths = st.number_input('Week 1 - Daily New Deaths')
week1_active = st.number_input('Week 1 - Active Cases')

week2_cases = st.number_input('Week 2 - Daily New Cases')
week2_deaths = st.number_input('Week 2 - Daily New Deaths')
week2_active = st.number_input('Week 2 - Active Cases')

week3_cases = st.number_input('Week 3 - Daily New Cases')
week3_deaths = st.number_input('Week 3 - Daily New Deaths')
week3_active = st.number_input('Week 3 - Active Cases')

if st.button('Predict'):
    # Combine the data into a sequence
    input_data = np.array([
        [week1_cases, week1_deaths, week1_active],
        [week2_cases, week2_deaths, week2_active],
        [week3_cases, week3_deaths, week3_active]
    ])

    # Ensure no zeros to avoid log(0) errors
    input_data[input_data == 0] = 1

    # Log transform the data
    input_data_log = np.log(input_data)

    # Reshape for the model
    input_data_log = input_data_log.reshape((1, 3, 3))

    # Make predictions
    prediction_log = model.predict(input_data_log)
    prediction = np.exp(prediction_log)  # Reverse log transform

    # Display the predictions
    st.subheader('Prediction for the next week')
    st.write(f'Predicted Daily New Cases: {prediction[0][0]}')
    st.write(f'Predicted Daily New Deaths: {prediction[0][1]}')
    st.write(f'Predicted Active Cases: {prediction[0][2]}')


st.markdown(
    """
    <style>
    .css-18e3th9 {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .css-1d391kg {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .css-1l02zno {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)
