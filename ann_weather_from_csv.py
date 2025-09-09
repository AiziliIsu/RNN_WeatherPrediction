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
df = df.drop(columns=df.columns[0])  # Drop first column (Date)

X = df.iloc[:, :-1].values
y_raw = df.iloc[:, -1].values

# Normalize input
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encode labels
encoder = LabelBinarizer()
y = encoder.fit_transform(y_raw)

# Save encoder and scaler for inference
np.savez("weather_labels.npz", classes=encoder.classes_)
np.save("weather_scaler_minmax.npy", scaler.min_)
np.save("weather_scaler_scale.npy", scaler.scale_)

# Initialize network
np.random.seed(1)
input_size = X.shape[1]
output_size = y.shape[1]

w1 = 2 * np.random.random((input_size, hidden_size)) - 1
w2 = 2 * np.random.random((hidden_size, output_size)) - 1

# Train
for epoch in range(epochs):
    l0 = X_scaled
    l1 = sigmoid(np.dot(l0, w1))
    l2 = softmax(np.dot(l1, w2))

    error = y - l2
    l2_delta = error * sigmoid_deriv(l2)

    l1_error = l2_delta.dot(w2.T)
    l1_delta = l1_error * sigmoid_deriv(l1)

    # Update using scaled learning rate
    w2 += learning_rate * l1.T.dot(l2_delta)
    w1 += learning_rate * l0.T.dot(l1_delta)

# Save weights
np.savez("multi_weather_weights.npz", w1=w1, w2=w2)
