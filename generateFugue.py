import numpy as np
import tensorflow as tf

# Load the saved model using tf.keras.models
loaded_model = tf.keras.models.load_model('C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\Output\\bachFugueV1.h5')

# Load the theme from a text file
def load_theme(file_path):
    with open(file_path, 'r') as file:
        theme = file.read().split(',')
    
    # Filter out non-integer values and convert to integers
    theme = [int(note) for note in theme if note.strip().isdigit()]
    
    return theme

# Generate a fugue from a given theme
def generate_fugue(model, theme, sequence_length, num_notes_to_generate):
    generated_fugue = list(theme)

    for _ in range(num_notes_to_generate):
        input_sequence = generated_fugue[-sequence_length:]
        input_sequence = np.array(input_sequence).reshape(1, -1, 1)

        # Predict the next note using the model
        predicted_note = model.predict(input_sequence)
        predicted_note = np.argmax(predicted_note, axis=-1)

        generated_fugue.append(predicted_note[0][0])

    return generated_fugue

# Specify the path to the theme text file and the number of notes to generate
theme_file_path = 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\input\\input_theme.txt'  # Adjust as needed
num_notes_to_generate = 128  # Adjust as needed

# Load the theme and generate the fugue
theme = load_theme(theme_file_path)
generated_fugue = generate_fugue(loaded_model, theme, 9, num_notes_to_generate)  # 32 is the sequence length used during training (see trainModel.py)

# Save the generated fugue to a text file
with open('C:\\Users\\Jakob\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\Output\\generated_fugue.txt', 'w') as file:
    file.write(','.join(map(str, generated_fugue)))

