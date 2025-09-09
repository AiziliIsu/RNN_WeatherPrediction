RNN for Weather Prediction
This project implements a foundational Recurrent Neural Network (RNN) from scratch in Python to predict weather conditions based on a CSV dataset. The code is designed to be a clear and educational example of how an RNN can be built and trained without the use of high-level deep learning frameworks.

Features
- From-Scratch Implementation: The RNN architecture, including forward and backward propagation, is built using NumPy to provide a deep understanding of its inner workings.
- Data Preprocessing: Handles data loading, feature normalization (MinMaxScaler), and one-hot encoding of target labels.
- Parameter Management: Accepts key training parameters like epochs, hidden layer size, and learning rate directly from the command line.
- Model Persistence: Saves the trained model weights and data scaling parameters to files, allowing for future use without retraining.

Technologies Used
Python: The core programming language.
NumPy: Essential for all numerical operations and matrix manipulations.
Pandas: Used for efficient data loading and handling of CSV files.
Scikit-learn: Utilized for data preprocessing utilities like MinMaxScaler and LabelBinarizer.

How to Run
To train the model, run the script from your terminal and provide the required arguments for epochs, hidden size, and learning rate:

python rnn_weather_from_csv.py 1000 50 0.01

1000: Number of training epochs.

50: The size of the hidden layer.

0.01: The learning rate.
