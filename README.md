# Bach AI Fugue Generator

## Overview
The Bach AI Fugue Generator is a Python project that uses PyTorch to train a neural network model on a dataset of Bach fugues and their corresponding themes. 
The goal of this project is to generate new fugues by providing the AI with a random theme. The MIDI files of Bach fugues are preprocessed, 
separated into voices, and paired with their themes to create a consistent dataset for training.

## Features
- MIDI file preprocessing to extract voices and themes.
- Dataset creation by pairing themes with each voice and extending them to match the length of the longest fugue.
- Training a PyTorch neural network model to generate new fugues based on provided themes.
- Generation of new Bach-like fugues using the trained model.

## Dataset
The dataset used for training the model consists of MIDI files of Bach fugues, each paired with its corresponding theme. 
The MIDI files are processed to separate individual voices and ensure that all arrays have the same length (matching the length of the longest fugue in the dataset).

## Model
The model is implemented using PyTorch, a popular deep learning framework. It is designed to take a theme as input and generate multiple voices to create a complete fugue. 
The training process involves optimizing the model's parameters to minimize the difference between generated fugues and real Bach fugues from the dataset.

## Usage
To use this project:

1. Clone the repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Prepare your MIDI dataset and place it in the `data/` directory.
4. Run the data preprocessing script to separate voices and create the dataset.
5. Train the model using the generated dataset.
6. Once the model is trained, you can generate new fugues by providing a theme.


## Contributing
Contributions to this project are welcome. If you have ideas for improvements or new features, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Special thanks to Johann Sebastian Bach for his timeless compositions that inspire this project.
- We also thank the PyTorch community for providing a powerful framework for deep learning.

## Contact
For any questions or inquiries, please contact [Jakob De Vreese](mailto:jakobdevreese@gmail.com).
