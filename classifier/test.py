from tensorflow import keras
import pickle

try:
    model = keras.models.load_model('classifier_model.h5')
except:
    print("Model not found. Please run train.py first.")
    exit()

features = pickle.load(open("testing_data/features.pickle", "rb"))
labels = pickle.load(open("testing_data/labels.pickle", "rb")) #outputs

print("Labels: ")
print(labels)
print("Predicted value: ")
print(model.predict(features))