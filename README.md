
# GPT Training Tool

This repository contains a GPT training tool adapted from Andrej Karpathy's [MiniGPT repository](https://github.com/karpathy/minGPT).
The tool runs as an API and can build, train, test, store, and reload various GPT src.

## Installation

Before using the GPT Training Tool, make sure to install the required packages by running the following commands:

```shell
pip install torch
pip install flask
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
- `\get_params`: gets the list of the currently loaded params
- `\update_params`: patches the params file, currently only accepts training params
- `\train`: trains the currently loaded model
- `\evaluate`: evaluates the current loss on the test data set
- `\generate`: returns generation
- `\complete`: returns completion from prompt
