import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam

def build_generator():
    model = Sequential([
        Dense(128, input_dim=100),  # 100 is the dimensionality of the input noise vector
        LeakyReLU(0.01),
        Dense(256),
        LeakyReLU(0.01),
        Dense(512),
        LeakyReLU(0.01),
        Dense(1024),
        LeakyReLU(0.01),
        Dense(784, activation='tanh')  # 784 is for 28x28 output image (if we consider it)
    ])
    return model

def build_discriminator():
    model = Sequential([
        Dense(1024, input_dim=784),  # 784 is for 28x28 input image (if we consider it)
        LeakyReLU(0.01),
        Dense(512),
        LeakyReLU(0.01),
        Dense(256),
        LeakyReLU(0.01),
        Dense(128),
        LeakyReLU(0.01),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_gan(generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# create the models
generator = build_generator()
discriminator = build_discriminator()

# compile the discriminator
discriminator.compile(loss='binary_crossentropy', optimizer=Adam())

# make the discriminator untrainable when we are training the generator
discriminator.trainable = False

# create the GAN
gan = build_gan(generator, discriminator)

# compile the GAN
gan.compile(loss='binary_crossentropy', optimizer=Adam())

import numpy as np

def train(generator, discriminator, gan, epochs=20000, batch_size=128, display_interval=1000):
    # Generate a batch of noise inputs to feed to the GAN
    input_noise = np.random.normal(0, 1, size=[batch_size, 100])

    # Generate some labels
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        # Select a random batch of real images from your dataset here
        # For this example we are just using random noise as our images
        real_images = np.random.normal(0, 1, size=[batch_size, 784])
        
        # Generate a batch of new images
        generated_images = generator.predict(input_noise)
        
        # Train the discriminator on real and fake images
        discriminator_real_loss = discriminator.train_on_batch(real_images, real_labels)
        discriminator_fake_loss = discriminator.train_on_batch(generated_images, fake_labels)
        
        # Calculate the total discriminator loss
        discriminator_loss = 0.5 * np.add(discriminator_real_loss, discriminator_fake_loss)
        
        # ---------------------
        #  Train Generator
        # ---------------------
        
        # Generate a batch of noise inputs to feed to the GAN
        input_noise = np.random.normal(0, 1, size=[batch_size, 100])
        
        # Train the GAN (with the discriminator weights frozen)
        gan_loss = gan.train_on_batch(input_noise, real_labels)

        # If at display interval, print the losses and save the model weights
        if epoch % display_interval == 0:
            print(f"Epoch {epoch}, Discriminator Loss: {discriminator_loss}, Generator Loss: {gan_loss}")
            generator.save_weights('generator.h5')
            discriminator.save_weights('discriminator.h5')

# Call the training function
train(generator, discriminator, gan)
