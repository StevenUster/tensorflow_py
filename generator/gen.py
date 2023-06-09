import tensorflow as tf
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description='Generate images with a trained model.')
parser.add_argument('num_images', type=int, help='Number of images to generate')
args = parser.parse_args()


def generate_images(model, num_images):
    for i in range(num_images):
        noise = tf.random.normal([1, 100])
        generated_image = model(noise, training=False)
        plt.imshow(generated_image[0, :, :, 0], cmap='gray')
        plt.axis('off')
        plt.savefig(f'generated/generated_image_{i}.png', bbox_inches='tight', pad_inches=0)
        plt.clf()


generator = tf.keras.models.load_model('models/generator_model_2.h5')

generate_images(generator, args.num_images)
