from tensorflow.python import keras
import pandas as pd
import numpy as np


with open('./winequality-white.csv') as f:
    data = pd.read_csv(f, sep=';', dtype=np.float32)

train_data = data.sample(frac=0.8)
train_y = train_data.pop('quality').values
mins, maxs = train_data.values.min(0), train_data.values.max(0)
train_x = (train_data.values - mins) / maxs

test_data = data.drop(train_data.index)
test_y = test_data.pop('quality').values
test_x = (test_data.values - mins) / maxs

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(train_x, train_y, epochs=20)
model.evaluate(test_x, test_y)
