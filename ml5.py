import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time
import pandas as pd

class BatteryForecastingModel(tf.keras.Model):
    def __init__(self, units=128):
        super(BatteryForecastingModel, self).__init__()

        self.lstm_1 = tf.keras.layers.LSTM(units=units)
        self.lstm_2 = tf.keras.layers.LSTM(units=units)

        self.dense_1 = tf.keras.layers.Dense(units=1)
        self.dense_2 = tf.keras.layers.Dense(units=1)

    def call(self, inputs):
        x = self.lstm_1(inputs)
        x = self.lstm_2(x)

        required_energy = self.dense_1(x)
        surplus_production = self.dense_2(x)

        return required_energy, surplus_production


def load_and_preprocess_data(data_path):
    """Load and preprocess data from a CSV file."""
    try:
        print("Loading data...")
        start_time = time.time()
        data = pd.read_csv(data_path, delimiter=';')
        print("Data loaded. Time elapsed: {:.2f} seconds".format(time.time() - start_time))

        # Normalize data
        print("Normalizing data...")
        start_time = time.time()
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data.values)
        print("Data normalized. Time elapsed: {:.2f} seconds".format(time.time() - start_time))

        # Split data into training and testing sets
        print("Splitting data into training and testing sets...")
        start_time = time.time()
        x_train, x_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.2)
        print("Data split. Time elapsed: {:.2f} seconds".format(time.time() - start_time))

        return {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}. Please provide a valid file path.")
        return None
    except Exception as e:
        print(f"An error occurred during data loading and preprocessing: {e}")
        return None


def train_model(model, data, epochs=100):
    """Train the model."""
    try:
        print("Training the model...")
        start_time = time.time()
        model.compile(optimizer='adam', loss='mse')
        model.fit(data['x_train'], data['y_train'], epochs=epochs, validation_data=(data['x_test'], data['y_test']))
        print("Model trained. Time elapsed: {:.2f} seconds".format(time.time() - start_time))
    except Exception as e:
        print(f"An error occurred during model training: {e}")


def evaluate_model(model, data):
    """Evaluate the model."""
    try:
        print("Evaluating the model...")
        start_time = time.time()
        y_pred = model.predict(data['x_test'])
        mse = np.mean((y_pred - data['y_test'])**2)
        print('MSE:', mse)
        print("Model evaluated. Time elapsed: {:.2f} seconds".format(time.time() - start_time))
    except Exception as e:
        print(f"An error occurred during model evaluation: {e}")


if __name__ == '__main__':
    print("Welcome! What would you like to do?")
    print("1. Load and preprocess data")
    print("2. Train the model")
    print("3. Evaluate the model")
    print("4. Save the model")
    print("5. Exit")

    data = None
    model = BatteryForecastingModel()
    model_path = 'battery_forecasting_model.h5'

    while True:
        choice = input("Choose an option (1-5): ")

        if choice == '1':
            data_path = input("Enter the path to the data: ")
            data = load_and_preprocess_data(data_path)
        elif choice == '2':
            if data is not None:
                train_model(model, data)
            else:
                print("Please load data first!")
        elif choice == '3':
            if data is not None:
                evaluate_model(model, data)
            else:
                print("Please load data first!")
        elif choice == '4':
            model_path = input("Enter the path to save the model: ")
            model.save(model_path)
            print("Model saved.")
        elif choice == '5':
            break
        else:
            print("Unknown option. Please try again.")
