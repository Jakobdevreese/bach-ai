import mido
import torch

# Load the trained model
#model = torch.load('C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\version 3\\saved_models\\bachModelV2.pth')

# Define the paths to the theme and output MIDI files
theme_file = 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\input\\input_theme.mid'
output_file = 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\output\\output_test_theme.mid'

# Extract notes from the theme file
theme_notes = []
midi = mido.MidiFile(theme_file)
for msg in midi.tracks[0]:
    if msg.type == 'note_on':
        theme_notes.append(torch.tensor([msg.note, msg.time, 1], dtype=torch.float32))

# Load the saved model
checkpoint = torch.load('C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\version 3\\saved_models\\bachModelV2.pth')
model = checkpoint

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
    note_on = mido.Message('note_on', note=note[0], time=int(note[1]))
    track.append(note_on)
midi.save(output_file)