from tensorflow import keras
import tensorflow as tf
import pickle

DATA = 'training_data'
IMG_SIZE = 32

features = pickle.load(open(DATA + "/features.pickle", "rb")) #inputs
labels = pickle.load(open(DATA + "/labels.pickle", "rb")) #outputs

model = tf.keras.Sequential([
    keras.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(features, labels, epochs=1000)
model.save('models/images')
