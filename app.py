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

logger = configure_logger()
app = Flask(app_name)

# Load model
@app.route('/load_model', methods=[GET])
def load_model_route():
    global params, dataset, model

    if name not in request.json:
        return jsonify({'status': 'Error: Missing required parameter "name".'})
    model_name = request.json[name]

    status, params, dataset, model = load_model(logger, model_name, True)
    logger.info(status)
    return jsonify({'status': status})

# Save model
@app.route('/save_model', methods=[GET])
def save_model_route():
    model_name = params.name
    if name in request.json:
        model_name = request.json[name]

    status = save_model(model, params, model_name)
    logger.info(status)
    return jsonify({'status': status})

# Update config
@app.route('/update_config', methods=[POST])
def update_config():
    for key in request.json:
        if key not in params.__dict__:
            return jsonify({'status': 'Error: Invalid parameter.'})
        
    for key in request.json:
        if key in params.__dict__:
            params.__dict__[key] = request.json[key]
    
    init_model(True) # re-initialize model
    return jsonify({'status': 'Complete.'})

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
    if length in request.json:
        length = request.json[length]        

    output = model.generate(length)
    return jsonify({'generation': output})

# Complete
@app.route('/complete', methods=[GET])
def complete_prompt():
    prompt = ''
    length = 500
    temp = 1.0

    if prompt in request.json:
        if len(request.json[prompt]) > params.ctx_length:
            logger.info('Prompt exceeds context window. Only the last ' +
                        f"{params.ctx_length} characters will be used...")
        prompt = request.json[prompt][-params.ctx_length:]
    
    if length in request.json:
        length = request.json[length]
        
    if temp in request.json:
        temp = request.json[temp]

    output = model.complete(prompt, length, temp)
    return jsonify({'completion': output})

if __name__ == '__main__':
    _, params, dataset, model = load_model(logger)
    app.run(host=host, port=port)
