#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("tf_bridge_model.h5")

# Load the preprocessing pipeline
with open("preprocess.pkl", "rb") as f:
    preprocess = pickle.load(f)

scaler = preprocess["scaler"]
y_mean = preprocess["y_mean"]
y_std = preprocess["y_std"]

# Streamlit UI
st.title("Bridge Load Capacity Prediction")
st.write("Enter bridge specifications below to predict the maximum load capacity (tons).")

# Input fields
span_ft = st.number_input("Span (ft)", min_value=50, max_value=1000, step=10)
deck_width_ft = st.number_input("Deck Width (ft)", min_value=10, max_value=100, step=1)
age_years = st.number_input("Age (years)", min_value=1, max_value=200, step=1)
num_lanes = st.number_input("Number of Lanes", min_value=1, max_value=10, step=1)
condition_rating = st.slider("Condition Rating (1-5)", 1, 5, 3)

# Material selection (one-hot encoding)
material_options = ["Steel", "Concrete", "Composite"]
material = st.selectbox("Bridge Material", material_options)

# Process material input
material_encoding = [1 if material == "Concrete" else 0, 1 if material == "Composite" else 0]

# Create feature array
features = np.array([[span_ft, deck_width_ft, age_years, num_lanes, condition_rating] + material_encoding])

# Standardize features
features_scaled = scaler.transform(features)

# Predict load capacity
if st.button("Predict"):
    prediction_scaled = model.predict(features_scaled)
    prediction = (prediction_scaled * y_std) + y_mean
    st.success(f"Predicted Maximum Load Capacity: {prediction[0][0]:.2f} tons")

