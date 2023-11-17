import numpy as np

# Ustalamy liczbę przykładów, długość sekwencji i liczbę cech
num_samples = 1000
sequence_length = 10
num_features = 8

# Generujemy losowe dane treningowe
x_train = np.random.rand(num_samples, sequence_length, num_features)
y_train = np.random.rand(num_samples, 1)

# Generujemy losowe dane testowe
x_test = np.random.rand(num_samples, sequence_length, num_features)
y_test = np.random.rand(num_samples, 1)

# Tworzymy słownik danych
data = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}
