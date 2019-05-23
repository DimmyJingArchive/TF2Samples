from tensorflow.python import keras
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

# Drops training data from raw data to get test data
test_data = data.drop(train_data.index)
# Same as above
test_y = test_data.pop('quality').values
test_x = (test_data.values - mins) / maxs

# Creates a sequential networks with three layers
# First layer and second layer contain 100 neurons,
# and have activation function of RELU.
# Last layer yields the result through linear activation
model = keras.Sequential([
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])
# Compiles the model with adam optimizer, mean square error loss,
# and root mean squared error metric
model.compile(optimizer='adam', loss='mse',
              metrics=[keras.metrics.RootMeanSquaredError(name='rmse')])
# Fit the model with train data for 100 epochs, might overfit if run more
model.fit(train_x, train_y, epochs=100)
# Evaluate the model using test dataset
model.evaluate(test_x, test_y)
