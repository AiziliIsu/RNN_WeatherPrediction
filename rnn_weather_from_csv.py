# rnn_weather_from_csv.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
import sys
from scipy.special import softmax

# Read parameters from CLI
epochs = int(sys.argv[1])
hidden_size = int(sys.argv[2])
learning_rate = float(sys.argv[3])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

# Load data
df = pd.read_csv("uploaded_weather.csv")
df = df.drop(columns=df.columns[0])  # Drop Date

X = df.iloc[:, :-1].values
y_raw = df.iloc[:, -1].values

# Normalize input
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encode labels
encoder = LabelBinarizer()
y = encoder.fit_transform(y_raw)

# Save encoder and scaler
np.savez("weather_labels_rnn.npz", classes=encoder.classes_)
np.save("weather_scaler_minmax_rnn.npy", scaler.min_)
np.save("weather_scaler_scale_rnn.npy", scaler.scale_)

# Initialize RNN weights
np.random.seed(1)
input_size = X.shape[1]
output_size = y.shape[1]

Wxh = 2 * np.random.random((input_size, hidden_size)) - 1  # input to hidden
Whh = 2 * np.random.random((hidden_size, hidden_size)) - 1  # hidden to hidden
Why = 2 * np.random.random((hidden_size, output_size)) - 1  # hidden to output

# Train
for epoch in range(epochs):
    total_loss = 0
    for i in range(len(X_scaled)):
        x = X_scaled[i]
        target = y[i]

        h_prev = np.zeros((hidden_size,))
        h = sigmoid(np.dot(x, Wxh) + np.dot(h_prev, Whh))
        y_pred = softmax(np.dot(h, Why))

        error = target - y_pred
        total_loss += np.sum(error ** 2)

        dy = error * sigmoid_deriv(y_pred)
        dh = dy.dot(Why.T) * sigmoid_deriv(h)

        Why += learning_rate * np.outer(h, dy)
        Whh += learning_rate * np.outer(h_prev, dh)
        Wxh += learning_rate * np.outer(x, dh)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# Save weights
np.savez("rnn_weather_weights.npz", Wxh=Wxh, Whh=Whh, Why=Why)
