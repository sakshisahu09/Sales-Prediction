import numpy as np
import streamlit as st
import pickle

# Load the trained sales model
with open("sales_model.pkl", 'rb') as f:
    model = pickle.load(f)

# Web page title and banner
st.title("Sales Prediction")
st.image("Sales_Prediction.webp", use_container_width=True)  # Make sure this image is in the same folder

# Input sliders for advertising budget
tv_budget = st.slider("TV Advertising Budget ($k)", 0.0, 300.0)
radio_budget = st.slider("Radio Advertising Budget ($k)", 0.0, 50.0)
newspaper_budget = st.slider("Newspaper Advertising Budget ($k)", 0.0, 100.0)

# Predict button
if st.button("Predict Sales"):
    input_data = np.array([[tv_budget, radio_budget, newspaper_budget]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Sales : {prediction[0]:.2f} units")