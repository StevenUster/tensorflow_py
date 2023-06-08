import pickle
import matplotlib.pyplot as plt

IMG_SIZE = 32

features = pickle.load(open("testing_data/features.pickle", "rb"))

for idx, feature in enumerate(features):
    image = feature.reshape(IMG_SIZE, IMG_SIZE)
    image = image / 255.0
    plt.imshow(image, cmap='gray')
    plt.savefig(f"print_images/image_{idx}.png")
