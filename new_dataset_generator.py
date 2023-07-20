import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_generator(latent_dim, output_shape):
    model = keras.Sequential()
    model.add(layers.Dense(256, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(output_shape, activation='tanh'))
    return model

def build_discriminator(input_shape):
    model = keras.Sequential()
    model.add(layers.Dense(1024, input_shape=input_shape))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def preprocess_midi_data(data):
    # Convert the 'type' field to a numerical representation or one-hot encoding
    type_mapping = {'note_on': 1, 'note_off': 0}
    numerical_types = np.array([type_mapping[event['type']] for event in data], dtype=np.float32)

    # Extract other numerical features (time, note, velocity)
    time_values = np.array([event['time'] for event in data], dtype=np.float32)
    note_values = np.array([event['note'] for event in data], dtype=np.float32)
    velocity_values = np.array([event['velocity'] for event in data], dtype=np.float32)

    # Stack all features horizontally to create the preprocessed data
    preprocessed_data = np.stack((numerical_types, time_values, note_values, velocity_values), axis=-1)

    return preprocessed_data

def train_gan(generator, discriminator, combined, data, epochs=10000, batch_size=128, latent_dim=100):
    preprocessed_data = preprocess_midi_data(data)

    half_batch = batch_size // 2

    for epoch in range(epochs):
        # Train the discriminator
        idx = np.random.randint(0, preprocessed_data.shape[0], half_batch)
        real_data = preprocessed_data[idx]
        real_labels = np.ones((half_batch, 1))

        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        generated_data = generator.predict(noise)
        generated_labels = np.zeros((half_batch, 1))

        d_loss_real = discriminator.fit(real_data, real_labels, batch_size=half_batch, verbose=0)
        d_loss_generated = discriminator.fit(generated_data, generated_labels, batch_size=half_batch, verbose=0)
        d_loss = 0.5 * np.add(d_loss_real.history['loss'], d_loss_generated.history['loss'])

        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1))
        g_loss = combined.fit(noise, valid_labels, batch_size=batch_size, verbose=0)

        # Print the progress
        print(f"Epoch: {epoch}, D Loss: {d_loss}, G Loss: {g_loss.history['loss']}")

        # Save generated sequences
        if epoch % save_interval == 0:
            save_generated_sequences(generator, epoch)

    # Save the final generator model
    generator.save('generator_model.h5')


def save_generated_sequences(generator, epoch, num_samples=10):
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    generated_data = generator.predict(noise)

    # Save generated MIDI sequences as npy files or use any other method to handle the output
    for i in range(num_samples):
        generated_sequence = generated_data[i]
        np.save(f"generated_midi_epoch_{epoch}_sample_{i}.npy", generated_sequence)

if __name__ == "__main__":
    npy_file = "output_combined.npy"  # Replace with the path to your combined MIDI data npy file

    latent_dim = 100
    midi_data = np.load(npy_file, allow_pickle=True)
    data_shape = midi_data.shape  # The shape is (8120,) which is the correct format

    # Build and compile the discriminator
    discriminator = build_discriminator(input_shape=data_shape)
    discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))

    # Build and compile the generator
    generator = build_generator(latent_dim, data_shape[0])
    generator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))

    # Combined model (stack generator and discriminator)
    z = layers.Input(shape=(latent_dim,))
    generated_data = generator(z)
    discriminator.trainable = False
    validity = discriminator(generated_data)

    combined = keras.models.Model(z, validity)
    combined.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))

    # Train the GAN
    train_gan(generator, discriminator, combined, midi_data)