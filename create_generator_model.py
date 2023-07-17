import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, concatenate
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU, Dropout
from sklearn.model_selection import train_test_split



def define_generator(latent_dim, num_classes):
    model = Sequential()

    # Dense layer with input as noise vector and conditional input
    model.add(Dense(256, input_dim=latent_dim+num_classes))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1292, activation='tanh'))

    # Reshape the output to match the desired shape
    model.add(Reshape((13, 99, 1)))

    return model

    
def define_discriminator(input_shape, num_classes):
    model = Sequential()

    # Reshape the input to match the desired shape
    model.add(Reshape((13, 99, 1), input_shape=input_shape))

    # Convolutional layers
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    # Flatten the output and concatenate with the conditional input
    model.add(Flatten())
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))

    return model

    
def define_gan(generator, discriminator):
    discriminator.trainable = False  # Freeze the discriminator weights during GAN training

    model = Sequential()
    model.add(generator)
    model.add(discriminator)

    return model



def train(generator, discriminator, gan, trackset1, trackset2, epochs, batch_size):
    # Determine the number of batches per epoch
    num_batches = trackset1.shape[0] // batch_size

    for epoch in range(epochs):
        for batch in range(num_batches):
            # ---------------------
            # Train the discriminator
            # ---------------------

            # Generate random noise samples
            noise = np.random.normal(0, 1, (batch_size, latent_dim))

            # Select a random batch of real music tracks from trackset1
            real_tracks = trackset1[batch * batch_size : (batch + 1) * batch_size]

            # Generate fake music tracks using the generator
            fake_tracks = generator.predict([noise, labels1[batch * batch_size : (batch + 1) * batch_size]])

            # Create labels for real and fake tracks
            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))

            # Train the discriminator on real tracks
            d_loss_real = discriminator.train_on_batch(real_tracks, real_labels)

            # Train the discriminator on fake tracks
            d_loss_fake = discriminator.train_on_batch(fake_tracks, fake_labels)

            # Compute the average discriminator loss
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # -----------------
            # Train the generator
            # -----------------

            # Generate new random noise samples
            noise = np.random.normal(0, 1, (batch_size, latent_dim))

            # Set labels for the generated tracks as real
            gen_labels = np.ones((batch_size, 1))

            # Train the GAN with the discriminator weights fixed
            g_loss = gan.train_on_batch([noise, labels1[batch * batch_size : (batch + 1) * batch_size]], gen_labels)

            # Print the progress
            print(f"Epoch: {epoch+1}/{epochs} | Batch: {batch+1}/{num_batches} | D loss: {d_loss[0]:.4f} | G loss: {g_loss:.4f}")

        # --------------------------
        # Generate samples for evaluation
        # --------------------------

        # Generate a random noise sample
        noise = np.random.normal(0, 1, (1, latent_dim))

        # Select a random genre label
        random_label = np.random.randint(0, num_classes)

        # Generate a music track using the generator
        generated_track = generator.predict([noise, random_label])

        # Save the generated track
        # save_generated_track(generated_track, epoch)

    # Save the final models
    # save_models(generator, discriminator, gan)


    
def dataset_loader():
    BASE_PATH = 'C:/Users/buysa/OneDrive/Documents/pythonProjects/songmaker/Data/genres_original'

    # Load the preprocessed MFCC features and corresponding labels
    features = []
    labels = []

    # Loop over the audio files in the dataset
    for root, dirs, files in os.walk(BASE_PATH):
        for file in files:
            # Load the MFCC features
            feature_path = os.path.join(root, file.replace(".wav", ".npy"))
            mfcc = np.load(feature_path)

            # Load the genre label from the folder name
            label = os.path.basename(root)

            # Append the features and labels to the corresponding lists
            features.append(mfcc)
            labels.append(label)

    # Convert the lists to NumPy arrays
    features = np.array(features)
    labels = np.array(labels)

    return features, labels

# Load the dataset
trackset, labels = dataset_loader()

# Split the dataset into trackset1 and trackset2
trackset1, trackset2, labels1, labels2 = train_test_split(trackset, labels, test_size=0.2, random_state=42)


latent_dim = 100
generator = define_generator(latent_dim)
discriminator = define_discriminator()
gan = define_gan(generator, discriminator)
train(generator, discriminator, gan, trackset1, trackset2, epochs=100, batch_size=64)

