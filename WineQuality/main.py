from tensorflow.data.Dataset import from_tensor_slices
from tensorflow.python import keras
import tensorflow as tf
import pandas as pd
import numpy as np


# Get data from csv file
with open('./winequality-white.csv') as f:
    data = pd.read_csv(f, sep=';', dtype=np.float32)

# Get training data as 80% of raw data
train_data = data.sample(frac=0.8)
# Put the quality field into the results
train_y = train_data.pop('quality').values
# Get max and min for normalization
mins, maxs = train_data.values.min(0), train_data.values.max(0)
# Normalize the data
train_x = (train_data.values - mins) / maxs
train_ds = from_tensor_slices((train_x, train_y)).shuffle(10000).batch(32)

# Drops training data from raw data to get test data
test_data = data.drop(train_data.index)
# Same as above
test_y = test_data.pop('quality').values
test_x = (test_data.values - mins) / maxs
test_ds = from_tensor_slices((test_x, test_y)).shuffle(10000).batch(32)


# Creates a sequential networks with three layers
# First layer and second layer contain 100 neurons,
# and have activation function of RELU.
# Last layer yields the result through linear activation
class Model(keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = keras.layers.Dense(100, activation='relu')
        self.d2 = keras.layers.Dense(100, activation='relu')
        self.res = keras.layers.Dense(1, activation='linear')

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.res(x)


model = Model()
loss_object = keras.losses.MeanSquaredError(name='mse')
optimizer = keras.optimizers.Adam(name='adam')
metric = keras.metrics.RootMeanSquaredError(name='rmse')


@tf.function
def train_step(features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    metric(loss)


for features, labels in train_ds:
    train_step(features, labels)
