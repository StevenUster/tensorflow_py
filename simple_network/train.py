import tensorflow as tf
import numpy as np
from tensorflow import keras

features = np.array([2, 6, 5, 10, 3, 8], dtype=float) #inputs
labels = np.array([5, 13, 11, 21, 7, 17], dtype=float) #outputs

model = tf.keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(features, labels, epochs=1000)
model.save('models/simple_numbers')
