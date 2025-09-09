import streamlit as st
import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

st.title("🌤️ Weather Prediction ANN/RNN")

# Sidebar for parameters
st.sidebar.header(" Model Parameters")
model_type = st.sidebar.radio("Choose Model Type", ["ANN", "RNN"])
epochs = st.sidebar.slider("Epochs", 100, 1000, step=10, value=10)
hidden_size = st.sidebar.slider("Hidden Layer Size", 2, 20, value=6)
learning_rate = st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=0.01, step=0.0001, format="%.4f")

uploaded_file = st.file_uploader(" Upload your weather CSV (date column first, label last)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.to_csv("uploaded_weather.csv", index=False)
    st.success(" File uploaded and saved!")

    if st.button(f"Train {model_type}"):
        import os
        st.info(" Training in progress...")
        if model_type == "ANN":
            os.system(f"python ann_weather_from_csv.py {epochs} {hidden_size} {learning_rate}")
        else:
            os.system(f"python rnn_weather_from_csv.py {epochs} {hidden_size} {learning_rate}")
        st.success(f" {model_type} Training complete!")

    st.subheader("🌡️ Try a Prediction")
    data = df.drop(columns=df.columns[0])
    features = data.columns[:-1]
    input_vals = []

    for feature in features:
        val = st.number_input(f"{feature}", value=float(data[feature].mean()))
        input_vals.append(val)

    if st.button("Predict Weather"):
        x = np.array([input_vals])
        if model_type == "ANN":
            scaler_min = np.load("weather_scaler_minmax.npy")
            scaler_scale = np.load("weather_scaler_scale.npy")
            x_scaled = (x - scaler_min) * scaler_scale

            w_data = np.load("multi_weather_weights.npz")
            w1 = w_data["w1"]
            w2 = w_data["w2"]

            h = sigmoid(np.dot(x_scaled, w1))
            out = sigmoid(np.dot(h, w2))

            labels = np.load("weather_labels.npz")["classes"]

        else:  # RNN
            scaler_min = np.load("weather_scaler_minmax_rnn.npy")
            scaler_scale = np.load("weather_scaler_scale_rnn.npy")
            x_scaled = (x - scaler_min) * scaler_scale

            w_data = np.load("rnn_weather_weights.npz")
            Wxh = w_data["Wxh"]
            Whh = w_data["Whh"]
            Why = w_data["Why"]

            h_prev = np.zeros((Wxh.shape[1],))
            h = sigmoid(np.dot(x_scaled, Wxh) + np.dot(h_prev, Whh))
            out = sigmoid(np.dot(h, Why))

            labels = np.load("weather_labels_rnn.npz")["classes"]

        prediction_idx = np.argmax(out)

        st.write(f" Prediction: **{labels[prediction_idx]}**")

        # Check shape to index correctly
        if out.ndim == 2:
            confidence = out[0][prediction_idx]
        else:  # 1D vector
            confidence = out[prediction_idx]

        st.write(f" Confidence: {confidence:.2f}")

