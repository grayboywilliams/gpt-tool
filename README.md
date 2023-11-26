
# GPT Training Tool

This repository contains a GPT architecture adapted from Andrej Karpathy's [MiniGPT repository](https://github.com/karpathy/minGPT).

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

The application exposes the following endpoints...
- `\generate`: returns generation
- `\complete`: returns completion from prompt
- `\train`: trains the model
- `\evaluate`: evaluates the current loss on the test dataset
- `\save_parameters`: saves the current model weights
- `\load_parameters`: loads the model weights from checkpoint name
