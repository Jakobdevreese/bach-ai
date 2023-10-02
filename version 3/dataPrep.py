import numpy as np
import mido
import os

# Define the paths to the fugue and theme folders
fugue_folder = 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\data\\Fugas'
theme_folder = 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\data\\Themas'

# Function: extract notes from file and return a list of tuples containing the pitch and length of each note
def extract_notes_from_track(midi_file, track_number):
    notes = []
    midi = mido.MidiFile(midi_file)
    for msg in midi.tracks[track_number]:
        if msg.type == 'note_on':
            check = [msg.note, msg.time, track_number]
            # Check if the message contains the correct information
            if len(check) == 3:
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

print("Fugue List:")
print(fugue_list)
print("Theme List:")
print(theme_list)


