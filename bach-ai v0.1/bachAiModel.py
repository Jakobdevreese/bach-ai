# Function to define the model for Bach AI v0.1

# 07/10/2023 - Jakob De Vreese

# import the necessary libraries
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# Define the model
def bach_ai_model(fugue_list_padded, theme_list_padded, n_epochs, learning_rate, batch_size, hidden_size, temperature, filename):

    # Give feedback
    print("")
    print("Defining the model...")
    print("")
    print("Convert the data to tensors...")
    print("")

    # Convert the data to tensors
    theme_tensors = torch.cat([torch.stack(theme) for theme in theme_list_padded])
    fugue_tensors = torch.cat([torch.stack(fugue) for fugue in fugue_list_padded])

    # create a dataset
    dataset = TensorDataset(theme_tensors, fugue_tensors)

    # create a dataloader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False) # shuffle = False because we want to keep the order of the notes

    # Give feedback
    print("")
    print("Data converted to tensors!")
    print("")
    print("Let's recreate bach his brain!")
    print("")
    print("")

    # Define the RNN model
    class bachModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(bachModel, self).__init__()
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
    
        def forward(self, x, temperature=temperature):
            # Convert input tensor to float32
            x = x.float()
            out, _ = self.rnn(x)
            out = self.fc(out)
            # apply softmax with temperature
            out = nn.functional.softmax(out / temperature, dim=-1)
            return out
    
    # Initialize the model
    model = bachModel(input_size=3, hidden_size=hidden_size, output_size=3) # input_size = 3 (pitch, length, track_number) / hidden_size = 128 (number of neurons in hidden layer) / output_size = 3 (pitch, length, track_number)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Give feedback
    print("")
    print("Model initialized!")
    print("")
    print("Start the magic!!!")
    print("Bach is coming back to life!!!")
    print("")
    print("")
    print("Let's start training!")
    print("")
    print("")
    print("")
    print("Bach is born and his brain has to learn everything again from scratch!")
    print("This can take a while, he has to practice a lot!")
    print("")
    print("He wil learn all his music again " + str(n_epochs) + " times!")
    print("Sit tight and enjoy the show!")
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

    # Give feedback
    print("")
    print("WOW!!!")
    print("Bach is back!!!")
    print("")
    print("That took forever!!!")
    print("")
    print("")
    print("Let's save the model so we can use it later!")
    print("")
    print("")

    # Save the model
    torch.save(model.state_dict(), filename)

    # Give feedback
    print("")
    print("Model saved!")
    print("")
    print("path: " + filename)
    print("")
    print("")