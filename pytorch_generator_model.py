import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import soundfile as sf
from torch.utils import data
from torchaudio_augmentations import (
    RandomResizedCrop,
    RandomApply,
    PolarityInversion,
    Noise,
    Gain,
    HighLowPass,
    Delay,
    PitchShift,
    Reverb,
    Compose,
)


GTZAN_GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']


class GTZANDataset(data.Dataset):
    def __init__(self, data_path, split, num_samples, num_chunks, is_augmentation):
        self.data_path =  data_path if data_path else ''
        self.split = split
        self.num_samples = num_samples
        self.num_chunks = num_chunks
        self.is_augmentation = is_augmentation
        self.genres = GTZAN_GENRES
        self._get_song_list()
        if is_augmentation:
            self._get_augmentations()

    def _get_song_list(self):
        list_filename = os.path.join(self.data_path, '%s_filtered.txt' % self.split)
        with open(list_filename) as f:
            lines = f.readlines()
        self.song_list = [line.strip() for line in lines]

    def _get_augmentations(self):
        transforms = [
            RandomResizedCrop(n_samples=self.num_samples),
            RandomApply([PolarityInversion()], p=0.8),
            RandomApply([Noise(min_snr=0.3, max_snr=0.5)], p=0.3),
            RandomApply([Gain()], p=0.2),
            RandomApply([HighLowPass(sample_rate=22050)], p=0.8),
            RandomApply([Delay(sample_rate=22050)], p=0.5),
            RandomApply([PitchShift(n_samples=self.num_samples, sample_rate=22050)], p=0.4),
            RandomApply([Reverb(sample_rate=22050)], p=0.3),
        ]
        self.augmentation = Compose(transforms=transforms)

    def _adjust_audio_length(self, wav):
        if self.split == 'train':
            random_index = random.randint(0, len(wav) - self.num_samples - 1)
            wav = wav[random_index : random_index + self.num_samples]
        else:
            hop = (len(wav) - self.num_samples) // self.num_chunks
            wav = np.array([wav[i * hop : i * hop + self.num_samples] for i in range(self.num_chunks)])
        return wav

    def __getitem__(self, index):
        line = self.song_list[index]

        # get genre
        genre_name = line.split('/')[0]
        genre_index = self.genres.index(genre_name)

        # get audio
        audio_filename = os.path.join(self.data_path, 'genres', line)
        wav, fs = sf.read(audio_filename)

        # adjust audio length
        wav = self._adjust_audio_length(wav).astype('float32')

        # data augmentation
        if self.is_augmentation:
            wav = self.augmentation(torch.from_numpy(wav).unsqueeze(0)).squeeze(0).numpy()

        return wav, genre_index

    def __len__(self):
        return len(self.song_list)

def get_dataloader(data_path=None, 
                   split='train', 
                   num_samples=22050 * 29, 
                   num_chunks=1, 
                   batch_size=16, 
                   num_workers=0, 
                   is_augmentation=False):
    is_shuffle = True if (split == 'train') else False
    batch_size = batch_size if (split == 'train') else (batch_size // num_chunks)
    data_loader = data.DataLoader(dataset=GTZANDataset(data_path, 
                                                       split, 
                                                       num_samples, 
                                                       num_chunks, 
                                                       is_augmentation),
                                  batch_size=batch_size,
                                  shuffle=is_shuffle,
                                  drop_last=False,
                                  num_workers=num_workers)
    return data_loader


data_path = 'Data/genres_original'  # Set your GTZAN dataset path
split = 'train'  # Choose the split ('train', 'test', etc.)
num_samples = 22050 * 29  # Number of audio samples per example
num_chunks = 1  # Number of chunks per example (for test/validation split)
batch_size = 16  # Batch size
num_workers = 0  # Number of workers for data loading
is_augmentation = False  # Enable data augmentation

train_loader = get_dataloader(data_path, split, num_samples, num_chunks, batch_size, num_workers, is_augmentation)


class Generator(nn.Module):
    def __init__(self, input_size, output_size, num_classes):
        super(Generator, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_classes = num_classes

        # Define layers
        self.label_embedding = nn.Embedding(num_classes, input_size)
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_size),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        embedded_labels = self.label_embedding(labels)
        input = torch.cat((noise, embedded_labels), dim=1)
        output = self.fc(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Discriminator, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes

        # Define layers
        self.label_embedding = nn.Embedding(num_classes, input_size)
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input, labels):
        embedded_labels = self.label_embedding(labels)
        input = torch.cat((input, embedded_labels), dim=1)
        output = self.fc(input)
        return output

# Set hyperparameters
input_size = 100  # Size of the random noise vector
output_size = num_samples  # Size of the generated audio sample

# Instantiate models
generator = Generator(input_size, output_size,10)
discriminator = Discriminator(output_size, len(GTZAN_GENRES))


# Define loss function
adversarial_loss = nn.BCELoss()

# Define optimizers
lr = 0.0002
betas = (0.5, 0.999)
generator_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

# Additional parameters
num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


generator.to(device)
discriminator.to(device)
adversarial_loss.to(device)

for epoch in range(num_epochs):
    for batch_idx, (real_audio, labels) in enumerate(train_loader):
        batch_size = real_audio.size(0)

        # Move data to device
        real_audio = real_audio.to(device)
        labels = labels.to(device)

        # Train the discriminator
        discriminator_optimizer.zero_grad()

        real_validity = discriminator(real_audio, labels)
        real_labels = torch.ones(batch_size, 1).to(device)
        real_loss = adversarial_loss(real_validity, real_labels)

        noise = torch.randn(batch_size, input_size).to(device)
        fake_audio = generator(noise, labels)
        fake_validity = discriminator(fake_audio.detach(), labels)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        fake_loss = adversarial_loss(fake_validity, fake_labels)

        discriminator_loss = (real_loss + fake_loss) / 2
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # Train the generator
        generator_optimizer.zero_grad()

        fake_validity = discriminator(fake_audio, labels)
        generator_loss = adversarial_loss(fake_validity, real_labels)

        generator_loss.backward()
        generator_optimizer.step()

        # Print training progress
        if batch_idx % 10 == 0:
            print(f'Epoch: [{epoch+1}/{num_epochs}]\tBatch: [{batch_idx}/{len(train_loader)}]\tDiscriminator Loss: {discriminator_loss.item():.4f}\tGenerator Loss: {generator_loss.item():.4f}')
