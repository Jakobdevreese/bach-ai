# import libraries
import numpy as np
import mido
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence  # Import pad_sequence from rnn module

# Variables
# Define the paths to the fugue and theme folders
fugue_folder = 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\data\\Fugas'
theme_folder = 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\data\\Themas'

# variables for the model

# dataPrep
# Function: extract notes from file and return a list of tuples containing the pitch and length of each note
def extract_notes_from_track(midi_file, track_number):
    notes = []
    midi = mido.MidiFile(midi_file)
    for msg in midi.tracks[track_number]:
        if msg.type == 'note_on':
            check = [msg.note, msg.time, track_number]
            # Check if the message contains the correct information
            if all(isinstance(item, int) for item in check) and len(check) == 3:
                notes.append((msg.note, msg.time, track_number))
            else:
                continue
    return notes

# Initialize the lists
fugue_list = []
theme_list = []

# Iterate through the MIDI files in the fugue folder
for fugue_filename in os.listdir(fugue_folder):
    fugue_full_path = os.path.join(fugue_folder, fugue_filename)
    midi = mido.MidiFile(fugue_full_path)
    fugue_notes = []

    # Iterate over all tracks in the MIDI file
    for track_number, track in enumerate(midi.tracks):
        track_notes = extract_notes_from_track(fugue_full_path, track_number)

        # Check if the track contains notes
        if len(track_notes) > 0:
            fugue_notes.extend(track_notes)

    # Find the corresponding theme file in the theme folder
    theme_filename = fugue_filename.replace('.mid', '_theme.mid')
    theme_full_path = os.path.join(theme_folder, theme_filename)

    # Check if the theme file exists before trying to extract notes from it
    if os.path.exists(theme_full_path):
        theme_notes = extract_notes_from_track(theme_full_path, 0)
        
        fugue_list.append(fugue_notes)
        theme_list.append(theme_notes)

# Transform data for training
# determine max length of theme and fugue
max_theme_len = max(len(theme) for theme in theme_list)
max_fugue_len = max(len(fugue) for fugue in fugue_list)

# Pad sequences to max length
padded_theme_list = [np.pad(theme, ((0, max_theme_len - len(theme)), (0, 0)), 'constant') for theme in theme_list]
padded_fugue_list = [np.pad(fugue, ((0, max_fugue_len - len(fugue)), (0, 0)), 'constant') for fugue in fugue_list]

# Convert to NumPy arrays
theme_arrays_padded = np.array(padded_theme_list)
fugue_arrays_padded = np.array(padded_fugue_list)

# Convert to PyTorch tensors
theme_tensors = [torch.tensor(theme, dtype=torch.int64) for theme in theme_arrays_padded]
fugue_tensors = [torch.tensor(fugue, dtype=torch.int64) for fugue in fugue_arrays_padded]

# Use pad_sequence to pad sequences to the same length
theme_tensor = pad_sequence(theme_tensors, batch_first=False)  # Pad and make them batch-first
fugue_tensor = pad_sequence(fugue_tensors, batch_first=False)  # Pad and make them batch-first

# Ensure that both tensors have the same number of rows
min_samples = min(theme_tensor.shape[0], fugue_tensor.shape[0])
theme_tensor = theme_tensor[:min_samples]
fugue_tensor = fugue_tensor[:min_samples]

# Create DataLoader for batching
dataset = TensorDataset(theme_tensor, fugue_tensor)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False) # never shuffle

# Define the model
# sequence_to_sequence class
class Seq2SeqAttention(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers=n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        print("Input tensor shape: ", src.shape) #debug
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        predictions = self.fc_out(outputs)
        return predictions
    
input_dim = np.max(theme_arrays_padded) + 1  # Adjust this based on your input data
output_dim = max_fugue_len  # Adjust this based on your output data
emb_dim = 16  # Adjust the embedding dimension
hidden_dim = 32  # Adjust the hidden dimension
n_layers = 2  # Adjust the number of layers
dropout = 0.5  # Adjust the dropout rate

model = Seq2SeqAttention(input_dim, emb_dim, hidden_dim, output_dim, n_layers, dropout)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Training loop
n_epochs = 10  # Adjust the number of epochs as needed

for epoch in range(n_epochs):
    model.train()
    for batch in dataloader:
        theme_batch, fugue_batch = batch
        optimizer.zero_grad()
        
        # Forward pass (no need to permute)
        # Reshape the tensor to have shape (seq_len, batch_size, input_size)
        theme_batch = theme_batch.permute(1, 0, 2)
        output = model(theme_batch)
        
        # Forward pass
        output = model(theme_batch)
        
        # Reshape for the loss function
        output_dim = output.shape[-1]
        output = output.view(-1, output_dim)
        fugue_batch = fugue_batch.view(-1)
        
        # Compute loss and backpropagate
        loss = criterion(output, fugue_batch)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{n_epochs}] Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\version 3\\saved_models\\bachModel.pth')
