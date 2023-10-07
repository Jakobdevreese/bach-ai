# import the necessary packages
import os
import mido
import torch

# File paths
# Define the paths to the fugue and theme folders
fugue_folder = 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\data\\Fugas'
theme_folder = 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\data\\Themas'

# function extracting notes

def extract_notes_from_track(midi_file, track_number, theme_track_number):
    
    notes = []  # list of tensors containing the pitch, length, and track number of each note
    midi = mido.MidiFile(midi_file)  # load the MIDI file

    # Maintain a dictionary to keep track of active notes and their start times
    active_notes = {}

    for msg in midi.tracks[track_number]:
        if msg.type == 'note_on':
            if track_number == 0:  # if it is a theme track
                check = [msg.note, msg.time, theme_track_number]
            else:
                check = [msg.note, msg.time, track_number]

            # Check if the message contains the correct information
            if all(isinstance(item, int) for item in check) and len(check) == 3:
                note = torch.tensor([msg.note, msg.time, track_number], dtype=torch.int32)
                notes.append(note)
                
                # Store the start time of the active note
                active_notes[msg.note] = msg.time

        elif msg.type == 'note_off':
            if msg.note in active_notes:
                start_time = active_notes[msg.note]
                duration = msg.time
                if track_number == 0:  # if it is a theme track
                    note = torch.tensor([msg.note, duration, theme_track_number], dtype=torch.int32)
                else:
                    note = torch.tensor([msg.note, duration, track_number], dtype=torch.int32)
                notes.append(note)
                del active_notes[msg.note]

    return notes

fugue_list = []
midi = mido.MidiFile('C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\data\\Fugas\\WTCI_Fugue2.mid')

for track_number, track in enumerate(midi.tracks):
    notes = extract_notes_from_track('C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\data\\Fugas\\WTCI_Fugue2.mid', track_number, 0)
    if len(notes) > 0:
        fugue_list.append(extract_notes_from_track('C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\data\\Fugas\\WTCI_Fugue2.mid', track_number, 0))

print(fugue_list)
