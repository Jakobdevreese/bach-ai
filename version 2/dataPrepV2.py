import mido
import os


# Directory paths containing the MIDI files for fugues and themes
fugue_folder = 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\Data\\Fugas'  # Replace with the path to your fugue folder
theme_folder = 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\Data\\Themas'  # Replace with the path to your theme folder

# Output text file paths for fugues and themes
fugue_file_path = 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\version 2\\dataset\\fugues.txt'  # Output file for fugues
theme_file_path = 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\version 2\\dataset\\themes.txt'  # Output file for themes


# Function: extract notes from file and return a list of tuples containing the pitch and length of each note
def extract_notes_from_track(midi_file, track_number):
    notes = []
    midi = mido.MidiFile(midi_file)
    for msg in midi.tracks[track_number]:
        if msg.type == 'note_on':
            notes.append((msg.note, msg.time))
    return notes

# Function: write notes to text file
def write_notes_to_file(notes, file_path):
    with open(file_path, 'a') as file:
        for note in notes:
            pitch, length = note
            file.write(f"{pitch},{length}\n")


# Initialize the text files (clear existing content) only once
open(fugue_file_path, 'w').close()
open(theme_file_path, 'w').close()

# Iterate through the MIDI files in the fugue folder
for fugue_filename in os.listdir(fugue_folder):
    if fugue_filename.endswith('.mid'):
        fugue_full_path = os.path.join(fugue_folder, fugue_filename)
        midi = mido.MidiFile(fugue_full_path)
        fugue_length = 0
        
        # Iterate over all tracks in the MIDI file
        for track_number, track in enumerate(midi.tracks):
            track_notes = extract_notes_from_track(fugue_full_path, track_number)
        
            write_notes_to_file(track_notes, fugue_file_path)
            fugue_length += len(track_notes)
            
        # Find the corresponding theme file in the theme folder
        theme_filename = fugue_filename.replace('.mid', '_theme.mid')
        theme_full_path = os.path.join(theme_folder, theme_filename)
        
        # Check if the theme file exists before trying to extract notes from it
        if os.path.exists(theme_full_path):
            theme_track_notes = extract_notes_from_track(theme_full_path, 0)  # Use track number 0 for the theme
            
            # Calculate how many times to repeat the theme notes
            theme_times = fugue_length // len(theme_track_notes)
            remaining_notes = fugue_length % len(theme_track_notes)

            # Write the theme notes to the file
            for i in range(theme_times):
                write_notes_to_file(theme_track_notes, theme_file_path)
            
            # Write the theme notes for any remaining fugue notes
            if remaining_notes > 0:
                write_notes_to_file(theme_track_notes[:remaining_notes], theme_file_path)
            
        else:
            print(f"Theme File does not exist for {fugue_filename}")

print("Conversion completed.")