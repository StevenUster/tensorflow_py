import tensorflow as tf
import matplotlib.pyplot as plt


generator = tf.keras.models.load_model('generator_model.h5')

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.savefig('generated/generated_image.png')
