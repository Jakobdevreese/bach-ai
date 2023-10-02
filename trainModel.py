import numpy as np
import tensorflow as tf

# Load the dataset from text files
def load_dataset(file_path):
    with open(file_path, 'r') as file:
        data = file.read().split(',')
    data = [int(note) for note in data]
    return data

# Preprocess the data into input and target sequences
def preprocess_data(data, sequence_length):
    input_sequences = []
    target_sequences = []

    for i in range(len(data) - sequence_length):
        input_seq = data[i:i + sequence_length]
        target_seq = data[i + sequence_length]
        input_sequences.append(input_seq)
        target_sequences.append(target_seq)

    return np.array(input_sequences), np.array(target_sequences)

# Load and preprocess your fugue and theme datasets
fugue_data = load_dataset('C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\Output\\fugue_dataset.txt')
theme_data = load_dataset('C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\Output\\theme_dataset.txt')

# Define sequence length and vocabulary size (number of unique notes)
sequence_length = 9  # Adjust this based on your preference
vocab_size = len(set(fugue_data + theme_data))

# Preprocess the data
fugue_input, fugue_target = preprocess_data(fugue_data, sequence_length)
theme_input, theme_target = preprocess_data(theme_data, sequence_length)

# Combine fugue and theme data
input_data = np.concatenate((fugue_input, theme_input))
target_data = np.concatenate((fugue_target, theme_target))

# Define and compile the model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, input_shape=(sequence_length, 1)),
    tf.keras.layers.Dense(vocab_size, activation='softmax')  # Update output units to vocab_size
])

# Compile the model with the "sparse_categorical_crossentropy" loss function
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(input_data, target_data, epochs=20, batch_size=50)

# Save the trained model
model.save('C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\Output\\bachFugueV1.h5')


