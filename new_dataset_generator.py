import tensorflow as tf
from tensorflow.keras import layers
import os
import random
import mido
import numpy as np
import pretty_midi


# MIDI dosyalarının bulunduğu dizini belirtin
midi_dir = 'archive/split_midi/split_midi'

# Tüm MIDI dosyalarının isimlerini alın
midi_files = os.listdir(midi_dir)

# Rasgele 100 MIDI dosyasını seçin
selected_files = random.sample(midi_files, 100)

# Seçilen MIDI dosyalarını yükle
midi_data = []
for file_name in selected_files:
    file_path = os.path.join(midi_dir, file_name)
    midi_data.append(mido.MidiFile(file_path))


# Üreteç (Generator) Modeli Oluşturma
def build_generator(latent_dim, output_shape):
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, input_dim=latent_dim, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(output_shape, activation='tanh'))
    return model

# Ayırt Edici (Discriminator) Modeli Oluşturma
def build_discriminator(input_shape):
    model = tf.keras.Sequential()
    model.add(layers.Dense(1024, input_shape=input_shape, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# GAN Modelini Oluşturma
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# Önceden belirlediğiniz latent boyutu ve çıkış şekline göre üreteç ve ayırt edici modelleri oluşturun
latent_dim = 100  # Örnek olarak, latent boyutu 100 olarak belirledik
output_shape = 128  # Örnek olarak, çıkış şekli 128 olarak belirledik

generator = build_generator(latent_dim, output_shape)
discriminator = build_discriminator((output_shape,))

# GAN modelini oluşturun
gan = build_gan(generator, discriminator)

# GAN'ı derleyin
gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))

# Verileri hazırlayın ve eğitim döngüsünü başlatın
# Bu adımlar GAN modelinizi eğitmenizi içerir ve veri ön işleme ve döngü yönetimine bağlı olarak değişiklik gösterecektir.
# Daha uzun süre eğitim ve veri ön işleme gerektirebilir.
def midi_to_2d_array(midi_data, output_shape):
    # Initialize an empty 2D array to store the MIDI data
    midi_array = np.zeros((len(midi_data), output_shape))

    for i, midi in enumerate(midi_data):
        # Get the ticks per beat from the MIDI file
        ticks_per_beat = midi.ticks_per_beat

        # Get the total number of ticks in the MIDI file
        total_ticks = max([max(note.end for note in instrument.notes) for instrument in midi.instruments])

        # Calculate the time step duration in ticks
        ticks_per_step = total_ticks / output_shape

        # Iterate through each time step and fill the MIDI array
        for j in range(output_shape):
            tick_start = int(j * ticks_per_step)
            tick_end = int((j + 1) * ticks_per_step)

            # Count the number of active notes at the current time step
            num_active_notes = 0
            for instrument in midi.instruments:
                notes = [note for note in instrument.notes if tick_start <= note.start < tick_end]
                num_active_notes += len(notes)

            # Store the number of active notes in the MIDI array
            midi_array[i, j] = num_active_notes

    return midi_array

# GAN modelini eğitin
# Eğitim için örnek bir kod parçası:
epochs = 1000
batch_size = 32


for epoch in range(epochs):
    for i in range(0, len(midi_data), batch_size):
        # Batch verilerini hazırlayın
        batch_midi = midi_data[i:i+batch_size]
        
        # Verileri GAN modeline uygun hale getirin (örneğin, MIDI verilerini latent vektörlere dönüştürün)
        # Örnek olarak, MIDI verilerini 2D matrislere dönüştüren bir fonksiyon kullanalım:
        real_midi_2d = midi_to_2d_array(batch_midi, output_shape)

        # Eğitim verileri ve etiketleri hazırlayın
        real_labels = np.ones((batch_size, 1))  # Gerçek veri için etiketler (1)
        fake_labels = np.zeros((batch_size, 1))  # Üretilen veri için etiketler (0)
        
        # Discriminator için eğitim verilerini birleştirin
        discriminator_input = np.concatenate([real_midi_2d, generated_midi])
        discriminator_labels = np.concatenate([real_labels, fake_labels])
        
        # Discriminator'ı eğitin
        discriminator_loss = discriminator.train_on_batch(discriminator_input, discriminator_labels)

        # Üreteç için rastgele gürültü vektörleri oluşturun
        noise = tf.random.normal([batch_size, latent_dim])
        
        # Gan'ı eğitin ve gerçek etiketlerle güncelleyin
        gan_labels = np.ones((batch_size, 1))
        gan_loss = gan.train_on_batch(noise, gan_labels)
        
    # Her epoch sonunda kayıpları yazdırın
    print(f"Epoch: {epoch}, Discriminator Loss: {discriminator_loss}, GAN Loss: {gan_loss}")

# Modeli kaydedin
generator.save('gan_generator.h5')
discriminator.save('gan_discriminator.h5')
gan.save('gan_model.h5')
