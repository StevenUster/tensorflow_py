import tensorflow as tf
import matplotlib.pyplot as plt


def generate_image(model):
    noise = tf.random.normal([1, 100])
    generated_image = model(noise, training=False)
    plt.imshow(generated_image[0, :, :, 0], cmap='gray')
    plt.savefig('generated/generated_image.png')


generator = tf.keras.models.load_model('generator_model.h5')

generate_image(generator)