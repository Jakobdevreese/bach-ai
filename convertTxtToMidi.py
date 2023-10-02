import pretty_midi

# Load the generated fugue from the text file
def load_generated_fugue(file_path):
    with open(file_path, 'r') as file:
        generated_fugue = file.read().split(',')
    generated_fugue = [int(note) for note in generated_fugue]
    return generated_fugue

# Convert the generated fugue to a MIDI file
def convert_to_midi(generated_fugue, output_file_path):
    midi_data = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)

    for note in generated_fugue:
        note_start = 0  # Adjust as needed
        note_end = 1    # Adjust as needed
        note_velocity = 64  # Adjust as needed
        note_obj = pretty_midi.Note(
            velocity=note_velocity,
            pitch=note,
            start=note_start,
            end=note_end
        )
        piano.notes.append(note_obj)

    midi_data.instruments.append(piano)
    midi_data.write(output_file_path)

# Specify the path to the generated fugue text file
generated_fugue_path = 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\Output\\generated_fugue.txt'

# Specify the path where you want to save the MIDI file
output_midi_path = 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\Output\\generated_fugue.mid'

# Load the generated fugue and convert it to a MIDI file
generated_fugue = load_generated_fugue(generated_fugue_path)
convert_to_midi(generated_fugue, output_midi_path)
