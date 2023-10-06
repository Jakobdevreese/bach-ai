import mido
import torch
import torch.nn as nn

# Define a flexible model class that adapts to the saved model's architecture
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.float()
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out

# Define the paths to the theme and output MIDI files
theme_file = 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\input\\input_theme.mid'
output_file = 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\output\\output_test_theme2.mid'

# Extract notes from the theme file
theme_notes = []
midi = mido.MidiFile(theme_file)
for msg in midi.tracks[0]:
    if msg.type == 'note_on':
        theme_notes.append(torch.tensor([msg.note, msg.time, 1], dtype=torch.float32))

# Load the saved model and create a new instance of the model with matching architecture
saved_model = torch.load('C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\version 3\\saved_models\\model_250.pth')
model = MyModel(input_size=3, hidden_size=250, output_size=3)

# Load the saved model's state_dict
model.load_state_dict(saved_model)

# Generate fugue voice for the theme
desired_length = 100  # Adjust this to your desired length
fugue_notes = []

with torch.no_grad():
    theme_tensor = torch.stack(theme_notes).unsqueeze(0)
    while len(fugue_notes) < desired_length:
        fugue_tensor = model(theme_tensor)
        # Process the fugue_tensor to get the next note
        next_note = fugue_tensor.squeeze().tolist()  # Example: [note, time, track]
        # Access the individual values of next_note and convert them to integers
        note = int(next_note[0][0])  # Extract note from tensor
        print(note)
        time = int(next_note[0][1])  # Extract time from tensor
        track = int(1)  # Extract track from tensor
        fugue_notes.append([note, time, track])  # Append the converted values


# Create a MIDI file with the fugue voice
midi = mido.MidiFile()
track = mido.MidiTrack()
midi.tracks.append(track)

# Add instrument specification (e.g., grand piano)
track.append(mido.Message('program_change', program=0))

# Set tempo (e.g., 120 BPM)
track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(60)))

# Set time signature (e.g., 4/4 time)
track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8))

for note in fugue_notes:
    # Create note_on messages for each note in the fugue
    note_on = mido.Message('note_on', note=note[0], time=note[1], channel=0)
    # Append the note_on message to the track
    track.append(note_on)
    # Create a corresponding note_off message to end the note
    note_off = mido.Message('note_off', note=note[0], time=0)
    # Append the note_off message to the track
    track.append(note_off)

# Save the MIDI file
midi.save(output_file)