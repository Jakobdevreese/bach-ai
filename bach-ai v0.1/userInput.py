# User input functions for paramerters for Bach AI v0.1

# 07/10/2023 - Jakob De Vreese

# Ask for: Hidden size, temperature, learning rate, batch size, numer of epochs and filename for the model

# give feedback to user
# ask for input
# return input

def userInput():

    # Give feedback
    print("Please enter the following parameters:")
    print("")
    print("Hidden size: number of neurons in hidden layer (recommended: 350 - 1500)")
    print("Temperature: the higher the temperature, the more random the output (recommended: 3.5 - 9.0)")
    print("Learning rate: the higher the learning rate, the faster the model learns (recommended: 0.0001 - 0.001)")
    print("Batch size: number of fugue notes per batch (recommended: 3 - 64)")
    print("Number of epochs: number of times the model will go through the entire dataset (recommended: 10 - 10000)")
    print("Filename: name of the file for the saved model (recommended: test_model_-NUMER OF HIDDEN SIZE-_-NUMBER OF EPOCHS-.pth)")
    print("--- DONT FORGET TO ADD THE .pth EXTENSION ---")
    print("")
    print("")
    print("Enter the parameters below:")

    hidden_size = int(input("Enter the Hidden size: "))
    temperature = float(input("Enter the Temperature: "))
    learning_rate = float(input("Enter the Learning rate: "))
    batch_size = int(input("Enter the Batch size: "))
    n_epochs = int(input("Enter the Number of epochs: "))
    filename = input("Enter the Filename (end with .pth): ")
    
    print("")
    print("")
    print("")
    print("Parameters:")
    print("Hidden size: ", hidden_size)
    print("Temperature: ", temperature)
    print("Learning rate: ", learning_rate)
    print("Batch size: ", batch_size)
    print("Number of epochs: ", n_epochs)
    print("Filename: ", filename)
    print("")
    print("")
    print("")
    print("Thank you!!!")
    print("")
    print("")

    return hidden_size, temperature, learning_rate, batch_size, n_epochs, filename