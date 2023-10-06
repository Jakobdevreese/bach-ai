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
theme_file = 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\input\\theme_input.mid'
output_file = 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\output\\output_test_theme5.mid'

# Extract notes from the theme file
theme_notes = []  # list of tensors containing the pitch, length, and track number of each note
midi = mido.MidiFile(theme_file)  # load the MIDI theme file

# Maintain a dictionary to keep track of active notes and their start times
active_notes = {}

for msg in midi.tracks[0]:
    if msg.type == 'note_on':
        check = [msg.note, msg.time, 1]  # Assuming it's always the first track (track_number 1)

        # Check if the message contains the correct information
        if all(isinstance(item, int) for item in check) and len(check) == 3:
            note = torch.tensor([msg.note, msg.time, 1], dtype=torch.float32)
            theme_notes.append(note)

            # Store the start time of the active note
            active_notes[msg.note] = msg.time
    elif msg.type == 'note_off':
        if msg.note in active_notes:
            start_time = active_notes[msg.note]
            duration = msg.time
            note = torch.tensor([msg.note, duration, 1], dtype=torch.float32)
            theme_notes.append(note)
            del active_notes[msg.note]

# Load the saved model and create a new instance of the model with matching architecture
saved_model = torch.load('C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\version 3\\saved_models\\Model_250.pth')
model = MyModel(input_size=3, hidden_size=250, output_size=3)

# Load the saved model's state_dict
model.load_state_dict(saved_model)

# Generate fugue voice for the theme
desired_length = 100  # Adjust this to your desired length
fugue_notes = []

# Initialize the theme_tensor with the theme_notes
theme_tensor = torch.stack(theme_notes).unsqueeze(0)
desired_length = 100  # Adjust this to your desired length

 

with torch.no_grad():
    # Generate a sequence of desired_length notes in one forward pass
    fugue_tensor = model(theme_tensor)
    fugue_notes = fugue_tensor.squeeze().tolist()

# Process the fugue_notes to extract individual notes
fugue_notes = [[int(note[0]), int(note[1]), 1] for note in fugue_notes]
print(fugue_notes)

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

# Initialize the total_time to keep track of the accumulated time
total_time = 0

for note in fugue_notes:
    if isinstance(note[0], int) and -127 <= note[0] <= 127:
        note_value = abs(note[0])
        duration = abs(note[1])

        # Create note_on message for the note
        note_on = mido.Message('note_on', note=note_value, velocity=64, time=total_time)
        track.append(note_on)

        # Update the total_time by adding the duration
        total_time += duration

        # Create note_off message for the note
        note_off = mido.Message('note_off', note=note_value, velocity=64, time=total_time)
        track.append(note_off)

# Save the MIDI file
midi.save(output_file)
