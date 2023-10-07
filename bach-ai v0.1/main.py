# Bach AI v0.1 - main.py

# 07/10/2023 - Jakob De Vreese

# import the necessacery function files
import dataprep
import openingMessage
import userInput
import paddingSeq
import bachAiModel


# Import the necessary libraries
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# Feedback to user with opening message
# Ask for user input for the parameters
# Start dataprep 
# Pad the sequences
# Start training
# End message with relevant information

# Feedback to user with opening message
openingMessage.opening_message()

# Ask for user input for the parameters
hidden_size, temperature, learning_rate, batch_size, n_epochs, filename = userInput.userInput()

cwd = os.getcwd() # Get the current working directory
filename_path = os.path.join(cwd, r"saved_models\{}".format(filename)) # file to save the model

# Start dataprep
# Call the extract_notes function to get the fugue count, fugue notes, and theme notes
fugue_count, fugue_notes, theme_notes = dataprep.extract_notes()

# Pad the sequences
fugue_list_padded, theme_list_padded = paddingSeq.padding_lists(fugue_notes, theme_notes)

# Start training
bachAiModel.bach_ai_model(fugue_list_padded, theme_list_padded, n_epochs, learning_rate, batch_size, hidden_size, temperature, filename_path)

# End message with relevant information
print("")
print("")
print("Thank you for using this program!")
print("")
print("Let's recap the parameters you entered:")
print("")  
print("Hidden size: ", hidden_size)
print("Temperature: ", temperature)
print("Learning rate: ", learning_rate)
print("Batch size: ", batch_size)
print("Number of epochs: ", n_epochs)
print("")
print("The model is saved as: ", filename_path)
print("")
print("You can now use the model to generate fugues!")
print("")
print("Have fun!")
print("")
print("")
print("")
print("   --- created by: Jakob De Vreese ---")
print("   --- version 0.1 ---")
print("   --- 07/10/2023 ---")
print("")


# End of program

