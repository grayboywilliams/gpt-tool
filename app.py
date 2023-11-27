from flask import Flask, request, jsonify
from models.params import *
from models.dataset import *
from models.gpt import *

params: Hyperparameters
dataset: Dataset
model: GPTLanguageModel

app = Flask(__name__)

# Initialize GPT model
def init_model(get_data=False):
    global params, dataset, model
    params = Hyperparameters()
    dataset = Dataset(params, get_data)
    model = GPTLanguageModel(params, dataset)

# Train
@app.route('/train', methods=['POST'])
def train_model():
    if 'num_batch' in request.json:
        params.num_batch = request.json['num_batch']
        print('num_batch: ', params.num_batch)
    if 'eval_interval' in request.json:
        params.eval_interval = request.json['eval_interval']
        print('eval_interval: ', params.eval_interval)
    if 'eval_iters' in request.json:
        params.eval_iters = request.json['eval_iters']
        print('eval_iters: ', params.eval_iters)

    print('Model training in progress...')
    model.begin_train()
    return jsonify({'status': 'Complete.'})

# Update config
@app.route('/update_config', methods=['POST'])
def update_config():
    for key in request.json:
        if key not in params.__dict__:
            return jsonify({'status': 'Error: Invalid parameter.'})
        
    for key in request.json:
        if key in params.__dict__:
            params.__dict__[key] = request.json[key]
    
    init_model(True) # re-initialize model
    return jsonify({'status': 'Complete.'})

# Generate
@app.route('/generate', methods=['GET'])
def generate_text():
    length = 500
    if 'length' in request.json:
        length = request.json['length']        

    output = model.generate(length)
    return jsonify({'generation': output})

# Complete
@app.route('/complete', methods=['GET'])
def complete_prompt():
    prompt = ''
    length = 500
    temp = 1.0

    if 'prompt' in request.json:
        if len(request.json['prompt']) > params.ctx_length:
            print('Prompt exceeds context window. Only the last ' +
                  str(params.ctx_length) + ' characters will be used...')
        prompt = request.json['prompt'][-params.ctx_length:]
    
    if 'length' in request.json:
        length = request.json['length']
        
    if 'temp' in request.json:
        temp = request.json['temp']

    output = model.complete(prompt, length, temp)
    return jsonify({'completion': output})

# Evaluate
@app.route('/evaluate', methods=['GET'])
def evaluate_model():
    losses = model.estimate_loss(True)
    return jsonify({'loss': float(f"{losses['test']:.4f}")})

# Save parameters
@app.route('/save_parameters', methods=['GET'])
def save_parameters_route():
    name = None
    if 'name' in request.json:
        name = request.json['name']

    return model.save_parameters(name)

# Load parameters
@app.route('/load_parameters', methods=['GET'])
def load_parameters_route():
    name = None
    if 'name' in request.json:
        name = request.json['name']

    return model.load_parameters(name)

if __name__ == '__main__':
    init_model()
    app.run(host='0.0.0.0', port=5000)
