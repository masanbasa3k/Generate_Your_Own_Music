from midiFile_to_message import midi_to_numeric
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence  # Import pad_sequence from torch.nn.utils.rnn




data = []
for i in range(1,10):
    d = midi_to_numeric(f'archive/{i}.mid')[0]
    print(len(d))
    data.append(d)

# Find the maximum sequence length in the data
max_sequence_length = max(len(seq) for seq in data)

# Pad the sequences to the maximum length
padding_token=-1
padded_data = [torch.tensor(seq + [padding_token] * (max_sequence_length - len(seq))) for seq in data]

# Convert the padded data to a PyTorch tensor
data_tensor = pad_sequence(padded_data, batch_first=True)

# Create a TensorDataset from the data
dataset = TensorDataset(data_tensor)

# Set batch size and create DataLoader
batch_size = 64  # Adjust the batch size as per your memory capacity
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self, latent_dim, vocab_size, hidden_dim, num_layers):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        # x: Input latent vector (batch_size, sequence_length, latent_dim)
        output, _ = self.lstm(x)
        output = self.fc(output)
        return output


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x: Input MIDI sequence (batch_size, sequence_length, input_dim)
        output, _ = self.lstm(x)
        output = self.fc(output)
        return output.squeeze()

# Continue from the previous code

# Set hyperparameters
latent_dim = 100  # Dimension of the latent vector
vocab_size = 124  # Size of the vocabulary (number of unique MIDI messages)
hidden_dim = 128  # Number of hidden units in LSTM layers
num_layers = 2  # Number of LSTM layers

# Initialize the GAN model
generator = Generator(latent_dim, vocab_size, hidden_dim, num_layers)
discriminator = Discriminator(vocab_size, hidden_dim, num_layers)

# Define loss functions and optimizers
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy loss
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

# Training loop
num_epochs = 100  # Adjust the number of epochs as needed

for epoch in range(num_epochs):
    for batch in data_loader:
        real_sequences = batch[0]
        batch_size = real_sequences.size(0)

        # Train the discriminator
        real_labels = torch.ones(batch_size)
        fake_labels = torch.zeros(batch_size)

        discriminator_optimizer.zero_grad()

        real_outputs = discriminator(real_sequences)
        real_loss = criterion(real_outputs, real_labels)

        latent_vector = torch.randn(batch_size, 1, latent_dim)
        fake_sequences = generator(latent_vector)
        fake_outputs = discriminator(fake_sequences.detach())
        fake_loss = criterion(fake_outputs, fake_labels)

        discriminator_loss = real_loss + fake_loss
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # Train the generator
        generator_optimizer.zero_grad()

        latent_vector = torch.randn(batch_size, 1, latent_dim)
        fake_sequences = generator(latent_vector)
        fake_outputs = discriminator(fake_sequences)
        generator_loss = criterion(fake_outputs, real_labels)

        generator_loss.backward()
        generator_optimizer.step()

    # Print progress
    print(f"Epoch [{epoch+1}/{num_epochs}]  Generator Loss: {generator_loss.item()}  Discriminator Loss: {discriminator_loss.item()}")

# After training, you can use the generator to generate MIDI sequences:
latent_vector = torch.randn(1, 1, latent_dim)  # For generating a single sequence
generated_sequence = generator(latent_vector)
