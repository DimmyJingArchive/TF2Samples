from tensorflow.python import keras
import pandas as pd
import numpy as np


def process_data(array):
    # Normalizes the array by dividing by min and multiplying
    # by max, and also reshapes it to [batch_size, 1]
    return np.reshape((array - min_num) / max_num, (-1, 1))


# Gets the data from the txt as csv
with open('./AutoInsurance.txt') as f:
    data = pd.read_csv(f, '\\s+', skiprows=10, decimal=',',
                       dtype={'X': np.float32, 'Y': np.float32})
# get training data by samplying 80% of total data
train_data = data.sample(frac=0.8)
# get min and max
min_num = train_data['X'].min()
max_num = train_data['X'].max()

train_x = process_data(train_data['X'].values)
train_y = train_data['Y'].values
# get test data by dropping everything in train data from original data
test_data = data.drop(train_data.index)

test_x = process_data(test_data['X'].values)
test_y = test_data['Y'].values

# get values to predict for testing accuracy of model
predict_x = process_data(np.array([0, 20, 40, 60, 120]))

# create model with two layers. First one has 128 neurons and uses
# the activation function RELU, while the second one combines the
# result into one neuron while using linear activation function
# to preserve the output.
model = keras.models.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])
# compile the model with adam optimizer, loss is mean squared error
# and metric is mean absolute error to see average difference between
# predicted and actual values
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# fits the model with 4000 epochs
model.fit(train_x, train_y, epochs=4000)
# evaluates the model with test set
model.evaluate(test_x, test_y)
# predicts and print the result of the predicted values
print(model.predict(predict_x))
