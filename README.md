
# GPT Training Tool

This repository contains a GPT training tool adapted from Andrej Karpathy's [MiniGPT repository](https://github.com/karpathy/minGPT).
The tool runs as an API and can build, train, test, store, and reload various GPT models.

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
- `\train`: trains the currently loaded model
- `\update_config`: patches the config file (re-inits model)
- `\generate`: returns generation
- `\complete`: returns completion from prompt
- `\evaluate`: evaluates the current loss on the test data set
- `\save_parameters`: saves the current model weights, hyperparameters, and logs
- `\load_parameters`: loads the model weights from checkpoint
