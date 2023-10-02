import os
import pretty_midi

# Function to extract notes from a MIDI file
def extract_notes_from_midi(midi_file_path):
    notes = []
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file_path)
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                notes.append(note.pitch)
    except Exception as e:
        print(f"Error processing {midi_file_path}: {str(e)}")

    return notes

# Function to process a directory of MIDI files and save as a text file
def process_midi_directory(directory_path, output_file):
    dataset = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".mid"):
            midi_file_path = os.path.join(directory_path, filename)
            notes = extract_notes_from_midi(midi_file_path)
            dataset.extend(notes)

    # Save the dataset to a text file in the specified output directory
    output_file_path = os.path.join(output_directory, output_file)
    with open(output_file_path, 'w') as file:
        file.write(','.join(map(str, dataset)))

# Specify the directories containing your fugue and theme MIDI files
fugue_directory = 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\Data\\Fugas'
theme_directory = 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\Data\\Themas'

# Specify the output directory for the text files
output_directory = 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\Output'

# Specify the output file names for the datasets
fugue_output_file = 'fugue_dataset.txt'
theme_output_file = 'theme_dataset.txt'

# Process the fugue and theme MIDI files separately and save as text files
process_midi_directory(fugue_directory, fugue_output_file)
process_midi_directory(theme_directory, theme_output_file)
