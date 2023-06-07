from tensorflow import keras

try:
    model = keras.models.load_model('models/simple_numbers')
except:
    print("Model not found. Please run train.py first.")
    exit()

while True:
    i = input("Enter value: ")

    if i == 'w':
        print(model.get_weights())
        continue

    print("Predicted value: ")
    print(model.predict([float(i)]))