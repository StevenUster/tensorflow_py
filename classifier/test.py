from tensorflow import keras
import numpy as np
import pickle

try:
    model = keras.models.load_model('classifier_model.h5')
except:
    print("Model not found. Please run train.py first.")
    exit()

features = pickle.load(open("testing_data/features.pickle", "rb"))
labels = pickle.load(open("testing_data/labels.pickle", "rb"))

predictions = model.predict(features)

predicted_labels = (predictions > 0.5).astype(int).flatten()

print("\nLabels vs Predicted values:")
for i in range(len(labels)):
    print(f'Actual: {labels[i]}, Predicted: {predicted_labels[i]}, Raw: {predictions[i][0]:.4f}')
