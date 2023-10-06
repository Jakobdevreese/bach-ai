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
output_file = 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\output\\output_test_theme.mid'

# Extract notes from the theme file
theme_notes = []
midi = mido.MidiFile(theme_file)
for msg in midi.tracks[0]:
    if msg.type == 'note_on':
        theme_notes.append(torch.tensor([msg.note, msg.time, 1], dtype=torch.float32))

# Load the saved model and create a new instance of the model with matching architecture
saved_model = torch.load('C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\version 3\\saved_models\\bachModelV3.pth')
model = MyModel(input_size=3, hidden_size=50, output_size=3)

# Load the saved model's state_dict
model.load_state_dict(saved_model)

# Generate fugue voice for the theme
with torch.no_grad():
    # Convert theme notes to a tensor and add a batch dimension
    theme_tensor = torch.stack(theme_notes).unsqueeze(0)
    # Pass the theme tensor through the model to generate the fugue voice
    fugue_tensor = model(theme_tensor)
    # Remove the batch dimension and convert the fugue tensor to a list of notes
    fugue_notes = fugue_tensor.squeeze(0).tolist()

# Create a MIDI file with the fugue voice
midi = mido.MidiFile()
track = mido.MidiTrack()
midi.tracks.append(track)
for note in fugue_notes:
    note_on = mido.Message('note_on', note=int(note[0]), time=int(note[1]))
    track.append(note_on)
midi.save(output_file)
