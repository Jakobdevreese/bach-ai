# import the necessary packages
import mido
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence 
from tqdm import tqdm

# convert midi files to dataset of tensors
# create a dataloader
# define the model
# train the model
# save the model

# Jakob De Vreese - 2/10/2023

print("""

:::::::::      :::      ::::::::  :::    :::      ::::    ::::   ::::::::  :::::::::  :::::::::: :::             ::::::::::: :::::::::      :::     ::::::::::: ::::    ::: :::::::::: :::::::::  
:+:    :+:   :+: :+:   :+:    :+: :+:    :+:      +:+:+: :+:+:+ :+:    :+: :+:    :+: :+:        :+:                 :+:     :+:    :+:   :+: :+:       :+:     :+:+:   :+: :+:        :+:    :+: 
+:+    +:+  +:+   +:+  +:+        +:+    +:+      +:+ +:+:+ +:+ +:+    +:+ +:+    +:+ +:+        +:+                 +:+     +:+    +:+  +:+   +:+      +:+     :+:+:+  +:+ +:+        +:+    +:+ 
+#++:++#+  +#++:++#++: +#+        +#++:++#++      +#+  +:+  +#+ +#+    +:+ +#+    +:+ +#++:++#   +#+                 +#+     +#++:++#:  +#++:++#++:     +#+     +#+ +:+ +#+ +#++:++#   +#++:++#:  
+#+    +#+ +#+     +#+ +#+        +#+    +#+      +#+       +#+ +#+    +#+ +#+    +#+ +#+        +#+                 +#+     +#+    +#+ +#+     +#+     +#+     +#+  +#+#+# +#+        +#+    +#+ 
#+#    #+# #+#     #+# #+#    #+# #+#    #+#      #+#       #+# #+#    #+# #+#    #+# #+#        #+#                 #+#     #+#    #+# #+#     #+#     #+#     #+#   #+#+# #+#        #+#    #+# 
#########  ###     ###  ########  ###    ###      ###       ###  ########  #########  ########## ##########          ###     ###    ### ###     ### ########### ###    #### ########## ###    ### 

      """)

# Variables
# Define the paths to the fugue and theme folders
fugue_folder = 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\data\\Fugas'
theme_folder = 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\data\\Themas'

# User Input
# Ask the user for hidden size, learning rate, batch size and number of epochs
print("Let's train a model!")
print("")
print("Enter the following parameters:")
print("")
print("hidden size: number of neurons in hidden layer, default = 25")
print("learning rate: default = 0.001")
print("batch size: default = 16")
print("number of epochs: default = 5")
print("")
print("Enter the parameters below:")
hidden_size = int(input("Enter the hidden size: "))
learning_rate = float(input("Enter the learning rate: "))
batch_size = int(input("Enter the batch size: "))
n_epochs = int(input("Enter the number of epochs: "))
print("")
print("Parameters entered")
print("")
print("choose a filename for the model")
print("")
filename = input("Enter the filename: (end with .pth)")
filename = 'C:\\Users\\Jakob\\OneDrive - Hogeschool Gent\\Try Out AI\\bach ai\\version 3\\saved_models\\' + filename
print("")

# variables for the model
input_size = 3 # pitch, length, track number
#hidden_size = 25 # number of neurons in hidden layer
output_size = 3 # pitch, length, track number
#learning_rate = 0.001
#batch_size = 35
#n_epochs = 5

print("")
print("Parameters entered:")
print("")
print("hidden size: " + str(hidden_size))
print("learning rate: " + str(learning_rate))
print("batch size: " + str(batch_size))
print("number of epochs: " + str(n_epochs))
print("")
print("input size: " + str(input_size))
print("output size: " + str(output_size))
print("")
print("Loading data")
print("")
print("fugue folder: " + fugue_folder)
print("theme folder: " + theme_folder)
print("")
print("Lets prep the data")
print("")

# dataPrep
# Function: extract notes from file and return a list of tensors containing the pitch, length and track number of each note
def extract_notes_from_track(midi_file, track_number, theme_track_number):
    
    notes = [] # list of tensors containing the pitch, length and track number of each note
    midi = mido.MidiFile(midi_file) # load the midi file
    
    for msg in midi.tracks[track_number]:
        if msg.type == 'note_on':
            if track_number == 0: # if it is a theme track
                check = [msg.note, msg.time, theme_track_number] 
                # Check if the message contains the correct information
                if all(isinstance(item, int) for item in check) and len(check) == 3:
                    notes.append(torch.tensor([msg.note, msg.time, theme_track_number], dtype=torch.int32)) # add the note to the list with the track matching the fugue voice
                else:
                    continue
            else:
                check = [msg.note, msg.time, track_number]
                # Check if the message contains the correct information
                if all(isinstance(item, int) for item in check) and len(check) == 3:
                    notes.append(torch.tensor([msg.note, msg.time, track_number], dtype=torch.int32))
                else:
                    continue
    return notes

# Initialize the lists
fugue_list = []
theme_list = []

# Initialize variable for counting the number of fugues
fugue_count = 0

# Iterate through the MIDI files in the fugue folder
for fugue_filename in os.listdir(fugue_folder):
    fugue_count += 1
    fugue_full_path = os.path.join(fugue_folder, fugue_filename)
    midi = mido.MidiFile(fugue_full_path)

    # Iterate over all tracks in the MIDI file
    for track_number, track in enumerate(midi.tracks):
        track_notes = extract_notes_from_track(fugue_full_path, track_number, track_number) # when it is a fugue and there is a track on 0 then it is certain the track number stays correct

        # Check if the track contains notes
        if len(track_notes) > 0:
            # Find the corresponding theme file in the theme folder
            theme_filename = fugue_filename.replace('.mid', '_theme.mid')
            theme_full_path = os.path.join(theme_folder, theme_filename)

            # Check if the theme file exists before trying to extract notes from it
            if os.path.exists(theme_full_path):
                theme_notes = extract_notes_from_track(theme_full_path, 0, track_number) # to give the theme track number the track_number of the fugue track
        
                fugue_list.append(track_notes)
                theme_list.append(theme_notes)
                # debug print statements
                print("Fugue and theme added to lists")
                print("name:" + fugue_filename + " track:" + str(track_number))
                print("theme:" + theme_filename + str(track_number))

print("")
print("Data processing finnished")
print("")
print("number of fugues: " + str(fugue_count))
print("")
print("number of rows fugue list: " + str(len(fugue_list)))
print("number of rows theme list: " + str(len(theme_list)))
print("")
print("padding the sequences")
print("")

# Transform data for training
# pad sequences to max length
# determine max length of fugue
max_fugue_len = max(len(fugue) for fugue in fugue_list)

# Pad sequences to max length by repeating the sequence
theme_list_padded = [row * (max_fugue_len // len(row)) + row[:max_fugue_len % len(row)] for row in theme_list]
fugue_list_padded = [row * (max_fugue_len // len(row)) + row[:max_fugue_len % len(row)] for row in fugue_list]

# Debug
print("Padded sequences fugues to max length: "+ str(max_fugue_len))
print("Padded sequences themes to max length: "+ str(max_fugue_len))
print("")
print("numer of rows theme list: " + str(len(theme_list)))
print("numer of rows fugue list: " + str(len(fugue_list)))
print("")
print("number of cells in a row theme list: " + str(len(theme_list_padded[0])))
print("number of cells in a row fugue list: " + str(len(fugue_list_padded[0])))
print("")


# Convert to PyTorch tensors
theme_tensors = torch.cat([torch.stack(theme) for theme in theme_list_padded])
fugue_tensors = torch.cat([torch.stack(fugue) for fugue in fugue_list_padded])

# Debug
#for i, (theme, fugue) in enumerate(zip(theme_tensors, fugue_tensors)):
    #print(f"Sequence {i + 1}: Theme size: {theme.size()}, Fugue size: {fugue.size()}")

# create a dataset
dataset = TensorDataset(theme_tensors, fugue_tensors)

# create a dataloader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False) # never shuffle

# Define the RNN model
class bachModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(bachModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, temperature=2.5):
        # Convert input tensor to float32
        x = x.float()
        out, _ = self.rnn(x)
        out = self.fc(out)
        # apply softmax with temperature
        out = nn.functional.softmax(out / temperature, dim=-1)
        return out
    
# Initialize the model
model = bachModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size) # input_size = 3 (pitch, length, track_number) / hidden_size = 128 (number of neurons in hidden layer) / output_size = 3 (pitch, length, track_number)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("")
print("Model initialized")
print("")
print("Start the magic!!!")
print("")
print("Training the model")
print("")
print("")
print("")

# Train the model
for epoch in range(n_epochs):
    epoch_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}", leave=False, total=len(dataloader))
    for i, batch in enumerate(dataloader):
        # Get the input and output batch
        theme_batch, fugue_batch = batch
        # Convert input tensors to float32
        theme_batch = theme_batch.float()
        fugue_batch = fugue_batch.float()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(theme_batch)

        # Calculate the loss
        loss = criterion(outputs, fugue_batch)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Update the progress bar
        epoch_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'], epoch=epoch)


    # Print output
    print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}]')

# Save the trained model
torch.save(model.state_dict(), filename)
print("")
print("Model saved")
print("")
print("path: " + filename)
print("")
print("Thank you for using this program!")
print("")
print("Jakob De Vreese - 2/10/2023")

# End of program
