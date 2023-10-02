import pretty_midi

# Load the MIDI file
midi_file_path = 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\input\\input_theme.mid'  # Replace with the path to your MIDI file
midi_data = pretty_midi.PrettyMIDI(midi_file_path)

# Extract note values from the MIDI file
theme_notes = []

for instrument in midi_data.instruments:
    for note in instrument.notes:
        theme_notes.append(int(note.pitch))

# Convert the list of notes to a comma-separated string
theme_notes_str = ','.join(map(str, theme_notes))

# Save the theme to a text file
output_file_path = 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\input\\input_theme.txt'  # Specify the desired output file path
with open(output_file_path, 'w') as file:
    file.write(theme_notes_str)