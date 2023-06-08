from tensorflow import keras
import tensorflow as tf
import numpy as np

def prepare_data(sequence, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


n_steps_in, n_steps_out = 3, 2
sequence = [i for i in range(100)]

X, y = prepare_data(sequence, n_steps_in, n_steps_out)

X = X.reshape((X.shape[0], X.shape[1], 1))

n_features = 1

model = tf.keras.Sequential([
    keras.layers.SimpleRNN(50, activation='relu', input_shape=(n_steps_in, n_features)),
    keras.layers.Dense(n_steps_out)
])

model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=200, verbose=0)

x_input = np.array([100, 101, 102])
x_input = x_input.reshape((1, n_steps_in, n_features))
print(x_input)
prediction = model.predict(x_input, verbose=0)
print(prediction)
