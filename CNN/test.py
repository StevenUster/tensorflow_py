import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the model (you must have trained and saved it before running this script)
model = load_model('model.h5')

# Load an image file to test, resizing it to 28x28 pixels (as required by this model)
img = cv2.imread('number.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))

# Convert the image data to the format the model is expecting
img = img.reshape(1, 28, 28, 1)
img = img.astype('float32')
img /= 255

# Predict the digit
prediction = model.predict([img])
print("Predicted digit:", np.argmax(prediction))
