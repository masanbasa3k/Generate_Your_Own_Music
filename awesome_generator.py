from midiFile_to_message import midi_to_numeric
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence  # Import pad_sequence from torch.nn.utils.rnn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.utils.rnn as rnn_utils


data = []
for i in range(1,20):
    d = midi_to_numeric(f'archive/{i}.mid')[0]
    data.append(d) 
    
max_len = max(len(d) for d in data)

for i in range(len(data)):
    diff = max_len - len(data[i])
    if diff > 0:
        data[i] += [0] * diff  

# Pad the sequences to the maximum length in the dataset
max_sequence_length = max(len(seq) for seq in data)
padded_data = [torch.tensor(seq + [0] * (max_sequence_length - len(seq))) for seq in data]

# Convert the padded data to a PyTorch tensor
data_tensor = pad_sequence(padded_data, batch_first=True)

# Create a TensorDataset with the input data and their corresponding lengths
dataset_with_lengths = TensorDataset(data_tensor, torch.tensor([len(seq) for seq in data]))

# Create a DataLoader from the dataset with lengths
batch_size = 64  # Adjust the batch size as per your memory capacity
data_loader = DataLoader(dataset_with_lengths, batch_size=batch_size, shuffle=True)



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
    def __init__(self, input_size, hidden_dim, num_layers):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, 1)  # Output a single value for each sequence
    
    def forward(self, x, lengths):
        # x: Input MIDI sequence (batch_size, sequence_length, input_size)
        x = x.float()

        # Get the 1-dimensional lengths tensor from the input
        lengths = lengths.squeeze()

        # Check the shape of x and make sure it's 3-dimensional
        if len(x.shape) != 3:
            x = x.unsqueeze(2)

        # Pack the input sequences before feeding them to the LSTM
        x_packed = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output_packed, _ = self.lstm(x_packed)

        # Unpack the output sequences
        output, _ = rnn_utils.pad_packed_sequence(output_packed, batch_first=True)

        # Use the last time step of the LSTM output
        last_time_step = output[torch.arange(output.size(0)), lengths - 1]

        output = self.fc(last_time_step)
        return output.squeeze()






# Set hyperparameters
latent_dim = 100  # Dimension of the latent vector
vocab_size = len(data[0])  # Size of the vocabulary (number of unique MIDI messages)
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
    for batch, lengths in data_loader:  # Unpack the batch and lengths
        real_sequences = batch[0]
        batch_size = real_sequences.size(0)

        # Train the discriminator
        real_labels = torch.ones(batch_size)
        fake_labels = torch.zeros(batch_size)

        discriminator_optimizer.zero_grad()

        real_outputs = discriminator(real_sequences, lengths)  # Pass lengths to the Discriminator
        real_loss = criterion(real_outputs, real_labels)

        latent_vector = torch.randn(batch_size, 1, latent_dim)
        fake_sequences = generator(latent_vector)
        fake_outputs = discriminator(fake_sequences.detach(), [latent_vector.size(1)] * batch_size)  # Use fixed length for fake sequences
        fake_loss = criterion(fake_outputs, fake_labels)

        discriminator_loss = real_loss + fake_loss
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # Train the generator
        generator_optimizer.zero_grad()

        latent_vector = torch.randn(batch_size, 1, latent_dim)
        fake_sequences = generator(latent_vector)
        fake_outputs = discriminator(fake_sequences, [latent_vector.size(1)] * batch_size)  # Use fixed length for fake sequences
        generator_loss = criterion(fake_outputs, real_labels)

        generator_loss.backward()
        generator_optimizer.step()

    # Print progress
    print(f"Epoch [{epoch+1}/{num_epochs}]  Generator Loss: {generator_loss.item()}  Discriminator Loss: {discriminator_loss.item()}")

# After training, you can use the generator to generate MIDI sequences:
latent_vector = torch.randn(1, 1, latent_dim)  # For generating a single sequence
generated_sequence = generator(latent_vector)