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
    # Load data
    print("Ładowanie danych...")
    start_time = time.time()
    data = pd.read_csv(data_path, delimiter=';')
    print("Dane załadowane. Czas trwania: {:.2f} sekund".format(time.time() - start_time))

    # Normalize data
    print("Normalizacja danych...")
    start_time = time.time()
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data.values)
    print("Dane znormalizowane. Czas trwania: {:.2f} sekund".format(time.time() - start_time))

    # Split data into training and testing sets
    print("Podział danych na zestawy treningowe i testowe...")
    start_time = time.time()
    x_train, x_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.2)
    print("Dane podzielone. Czas trwania: {:.2f} sekund".format(time.time() - start_time))

    return {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}


def train_model(model, data, epochs=100):
    print("Rozpoczynam trening modelu...")
    start_time = time.time()
    model.compile(optimizer='adam', loss='mse')
    model.fit(data['x_train'], data['y_train'], epochs=epochs, validation_data=(data['x_test'], data['y_test']))
    print("Model przeszkolony. Czas trwania: {:.2f} sekund".format(time.time() - start_time))


def evaluate_model(model, data):
    print("Oceniam model...")
    start_time = time.time()
    y_pred = model.predict(data['x_test'])
    mse = np.mean((y_pred - data['y_test'])**2)
    print('MSE:', mse)
    print("Model oceniony. Czas trwania: {:.2f} sekund".format(time.time() - start_time))


if __name__ == '__main__':
    print("Witaj! Co chciałbyś zrobić?")
    print("1. Załaduj i przetwórz dane")
    print("2. Przeszkol model")
    print("3. Oceń model")
    print("4. Zapisz model")
    print("5. Wyjdź")

    data = None
    model = BatteryForecastingModel()
    model_path = 'battery_forecasting_model.h5'

    while True:
        choice = input("Wybierz opcję (1-5): ")

        if choice == '1':
            data_path = input("Podaj ścieżkę do danych: ")
            data = load_and_preprocess_data(data_path)
        elif choice == '2':
            if data is not None:
                train_model(model, data)
            else:
                print("Najpierw załaduj dane!")
        elif choice == '3':
            if data is not None:
                evaluate_model(model, data)
            else:
                print("Najpierw załaduj dane!")
        elif choice == '4':
            model_path = input("Podaj ścieżkę do zapisania modelu: ")
            model.save(model_path)
            print("Model zapisany.")
        elif choice == '5':
            break
        else:
            print("Nieznana opcja. Spróbuj ponownie.")
