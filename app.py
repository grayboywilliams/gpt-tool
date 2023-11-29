from flask import Flask, request, jsonify
from src.params import *
from src.dataset import *
from src.gpt import *
from src.logger import *
from src.util import *
from constants.constants import *

params: Hyperparameters
dataset: Dataset
model: GPTLanguageModel

script_dir = os.path.dirname(os.path.realpath(__file__))
logger = configure_logger()
app = Flask(app_name)

# New model
@app.route('/new_model', methods=[POST])
def new_model_route():
    global params, dataset, model

    if name_arg not in request.json:
        return jsonify({'status': f'Error: Missing required parameter {name_arg}.'})
    model_name = request.json[name_arg]

    if os.path.exists(os.path.join(script_dir, checkpoints, model_name)):
        return jsonify({'status': f'Error: Model {model_name} already exists.'})

    status, ok = setup_params(request.json)
    if not ok:
        return jsonify({'status': status})

    status, params, dataset, model = load_model(logger, model_name)
    logger.info(status)
    return jsonify({'status': f'Model {model_name} is ready to use.'})

# Load model
@app.route('/load_model', methods=[POST])
def load_model_route():
    global params, dataset, model

    if name_arg not in request.json:
        return jsonify({'status': 'Error: Missing required parameter "name".'})
    name = request.json[name_arg]

    status, params, dataset, model = load_model(logger, name, None)
    logger.info(status)
    return jsonify({'status': status})

# Save model
@app.route('/save_model', methods=[POST])
def save_model_route():
    status = save_model(model, params)
    logger.info(status)
    return jsonify({'status': status})

# Get model
@app.route('/get_model', methods=[GET])
def get_model():
    return jsonify({'model': params.name})

# Get params
@app.route('/get_params', methods=[GET])
def get_params():
    return jsonify({'params': params.file_params})

# Update params
@app.route('/update_params', methods=[POST])
def update_params():
    for key in request.json:
        if key not in params.__dict__:
            return jsonify({'status': f'Error: Invalid parameter: {key}.'})
        if not is_training_param(key):
            return jsonify({'status': f'Error: Cannot change non-training parameters: {key}.'})
        
    for key in request.json:
        if key in params.__dict__:
            params.__dict__[key] = request.json[key]

    return jsonify({'status': 'Parameters updated.'})

# Train
@app.route('/train', methods=[POST])
def train_model():
    if num_batch in request.json:
        params.num_batch = request.json[num_batch]
    if eval_interval in request.json:
        params.eval_interval = request.json[eval_interval]
    if eval_size in request.json:
        params.eval_size = request.json[eval_size]

    logger.log(SUMMARY, 'Model training in progress...')
    model.begin_train()
    logger.log(SUMMARY, 'Model training complete.')
    return jsonify({'status': 'Complete.'})

# Evaluate
@app.route('/evaluate', methods=[GET])
def evaluate_model():
    losses = model.estimate_loss()
    return jsonify({'loss': float(f"{losses[test]:.4f}")})

# Generate
@app.route('/generate', methods=[GET])
def generate_text():
    length = 500
    temp = 1.0

    if length in request.json:
        length = request.json[length]   
    if temp in request.json:
        temp = request.json[temp]     

    output = model.generate(length)
    return jsonify({'generation': output})

# Complete
@app.route('/complete', methods=[GET])
def complete_prompt():
    prompt = ''
    length = 500
    temp = 1.0

    if prompt_arg in request.json:
        if len(request.json[prompt_arg]) > params.ctx_length:
            logger.info('Prompt exceeds context window. Only the last ' +
                        f"{params.ctx_length} characters will be used...")
        prompt = request.json[prompt_arg][-params.ctx_length:]
    if length_arg in request.json:
        length = request.json[length_arg]
    if temp_arg in request.json:
        temp = request.json[temp_arg]

    output = model.complete(prompt, length, temp)
    return jsonify({'completion': output})

if __name__ == '__main__':
    app.run(host=host, port=port)
