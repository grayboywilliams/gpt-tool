
# GPT Training Tool

This repository contains a GPT Training Tool adapted from Andrej Karpathy's [MiniGPT repository](https://github.com/karpathy/minGPT).
The tool runs as an API and can build, train, test, store, and reload various GPT models.
It will also generate a graph of your model architecture and provide a summary of its size.
Note: Currently it is equipped to learn character encodings, but it can be easily updated to learn subword encodings.

## Installation

Before using the GPT Training Tool, make sure to install the required packages:

```shell
pip install -r requirements.txt
```

## Usage

To run the GPT Training Tool, follow these steps:

### Start the Application

Run the main script:

```shell
python app.py
```

### Endpoints

The application exposes the following endpoints:
- `\new_model`: creates a new model using provided params
- `\load_model`: loads the provided model from checkpoint
- `\save_model`: saves the current model weights, hyperparameters, and logs
- `\get_model`: gets the name of the currently loaded model
- `\view_model`: shows a summary of the model size
- `\get_params`: gets the list of the currently loaded params
- `\update_params`: patches the params file, currently only accepts training params
- `\train`: trains the currently loaded model
- `\evaluate`: evaluates the current loss on the test data set
- `\generate`: returns generation
- `\complete`: returns completion from prompt

A postman collection with all of these endpoints set up is stored in configs.
