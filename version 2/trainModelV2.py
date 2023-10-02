import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, Embedding, Dense, Flatten, Reshape
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder

# Model variables
sequence_length = 9  # Length of the input sequences
num_unique_pitch = 1280  # Number of unique pitches in the dataset
num_unique_length = 1280  # Number of unique lengths in the dataset
embedding_dim = 64  # Dimension of the embedding layer
num_epochs = 10  # Number of training epochs
test_size_factor = 0.2  # Percentage of the dataset to use for validation

# Output text file paths for fugues and themes
fugue_text_file_path = 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\version 2\\dataset\\fugues.txt'  # Output file for fugues
theme_text_file_path = 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\version 2\\dataset\\themes.txt'  # Output file for themes

# Process the text data into sequences of integers
def convert_text_to_sequences(text, sequence_length):
    # Step 1: Split text data into individual lines
    lines = [line for line in text.split('\n') if line]
    
    # Step 2: Create a mapping from unique notes to integers for pitch and length separately
    unique_pitch = sorted(set(line.split(',')[0] for line in lines))
    unique_length = sorted(set(line.split(',')[1] for line in lines))
    
    pitch_to_int = {note: i for i, note in enumerate(unique_pitch)}
    length_to_int = {length: i for i, length in enumerate(unique_length)}
    
    # Step 3: Replace notes with integers
    sequences = []
    
    for line in lines:
        pitch, length = line.split(',')  # Split each line into pitch and length
        pitch_int = pitch_to_int[pitch]
        length_int = length_to_int[length]
        sequences.append((pitch_int, length_int))
        
    # Step 4: Create sequences of a fixed length
    note_sequences = []
    for i in range(0, len(sequences) - sequence_length, 1):
        seq = sequences[i:i + sequence_length]
        note_sequences.append(seq)
    
    return note_sequences, pitch_to_int, length_to_int

# Load your text data from the fugue and theme files
with open(fugue_text_file_path, 'r') as fugue_file, open(theme_text_file_path, 'r') as theme_file:
    fugue_text = fugue_file.read()
    theme_text = theme_file.read()

# Convert text data to sequences of integers
fugue_sequences, fugue_pitch_to_int, fugue_length_to_int = convert_text_to_sequences(fugue_text, sequence_length)
theme_sequences, theme_pitch_to_int, theme_length_to_int = convert_text_to_sequences(theme_text, sequence_length)

# Define the input layer
input_layer = Input(shape=(sequence_length,))

# Add an embedding layer
embedding_layer = Embedding(input_dim=num_unique_pitch + num_unique_length, output_dim=embedding_dim)(input_layer)

# Reshape the embedding layer output to match the expected input shape of the transformer
reshaped_embedding = Reshape((-1, sequence_length, 128))(embedding_layer)

# Add a transformer-based layer (e.g., a multi-head self-attention layer)
# Adjust the configuration based on your dataset and requirements
transformer_layer = tf.keras.layers.MultiHeadAttention(
    num_heads=4,
    key_dim=128,  # Updated key_dim to match the reshaped embedding dimension
    value_dim=128,  # Updated value_dim to match the reshaped embedding dimension
    dropout=0.1,
)(reshaped_embedding, reshaped_embedding)

# Flatten the transformer output
flatten_layer = Flatten()(transformer_layer)

# Add output layers for pitch and length
pitch_output = Dense(units=num_unique_pitch, activation='softmax')(flatten_layer)
length_output = Dense(units=num_unique_length, activation='softmax')(flatten_layer)

# Create a model with two separate outputs
model = Model(inputs=input_layer, outputs=[pitch_output, length_output])

# Compile the model with appropriate loss functions
model.compile(optimizer=Adam(learning_rate=0.001), loss=['categorical_crossentropy', 'categorical_crossentropy'], metrics=['accuracy'])

# Convert your integer sequences to NumPy arrays and pad them
fugue_sequences = pad_sequences(fugue_sequences, maxlen=sequence_length, padding='post')
theme_sequences = pad_sequences(theme_sequences, maxlen=sequence_length, padding='post')

# Initialize one-hot encoders for pitch and length
pitch_encoder = OneHotEncoder(sparse=False, categories='auto')
length_encoder = OneHotEncoder(sparse=False, categories='auto')

# Fit and transform the encoders on your data
fugue_one_hot_pitch = pitch_encoder.fit_transform(fugue_sequences[:, :, 0].reshape(-1, 1))
theme_one_hot_pitch = pitch_encoder.transform(theme_sequences[:, :, 0].reshape(-1, 1))

fugue_one_hot_length = length_encoder.fit_transform(fugue_sequences[:, :, 1].reshape(-1, 1))
theme_one_hot_length = length_encoder.transform(theme_sequences[:, :, 1].reshape(-1, 1))

# Reshape one-hot encoded sequences to match the expected shape
fugue_one_hot_pitch = fugue_one_hot_pitch.reshape(fugue_sequences.shape[0], sequence_length, -1)
theme_one_hot_pitch = theme_one_hot_pitch.reshape(theme_sequences.shape[0], sequence_length, -1)

fugue_one_hot_length = fugue_one_hot_length.reshape(fugue_sequences.shape[0], sequence_length, -1)
theme_one_hot_length = theme_one_hot_length.reshape(theme_sequences.shape[0], sequence_length, -1)

# Split your data into training and validation sets for pitch and length
fugue_inputs_train, fugue_inputs_val, fugue_targets_train_pitch, fugue_targets_val_pitch, fugue_targets_train_length, fugue_targets_val_length = train_test_split(
    fugue_sequences, fugue_one_hot_pitch, fugue_one_hot_length, test_size=test_size_factor, random_state=42)

theme_inputs_train, theme_inputs_val, theme_targets_train_pitch, theme_targets_val_pitch, theme_targets_train_length, theme_targets_val_length = train_test_split(
    theme_sequences, theme_one_hot_pitch, theme_one_hot_length, test_size=test_size_factor, random_state=42)

# Train the model on fugues
model.fit(fugue_inputs_train, [fugue_targets_train_pitch, fugue_targets_train_length], epochs=num_epochs, batch_size=64, validation_data=(fugue_inputs_val, [fugue_targets_val_pitch, fugue_targets_val_length]))

# Train the model on themes
model.fit(theme_inputs_train, [theme_targets_train_pitch, theme_targets_train_length], epochs=num_epochs, batch_size=64, validation_data=(theme_inputs_val, [theme_targets_val_pitch, theme_targets_val_length]))

# Save the trained model for future use
model.save('music_generation_model.h5')
