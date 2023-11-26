from flask import Flask, request, jsonify
from models.params import *
from models.dataset import *
from models.gpt import *

app = Flask(__name__)

# Initialize GPT model
params = Hyperparameters()
dataset = Dataset(params)
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

# Generate
@app.route('/generate', methods=['GET'])
def generate_text():
    if 'length' in request.args:
        length = request.args['length']
    else:
        length = 500

    output = model.generate(length)
    return jsonify({'generation': output})

# Complete
@app.route('/complete', methods=['GET'])
def complete_prompt():
    if 'prompt' in request.args:
        prompt = request.args['prompt']
    else:
        prompt = 'ROMEO: '
    
    if 'length' in request.args:
        length = request.args['length']
    else:
        length = 500

    output = model.complete(prompt, length)
    return jsonify({'completion': output})

# Evaluate
@app.route('/evaluate', methods=['GET'])
def evaluate_model():
    losses = model.estimate_loss()
    return float(f"{losses['test']:.4f}")

# Save parameters
@app.route('/save_parameters', methods=['GET'])
def save_parameters_route():
    name = 'cp'
    if 'name' in request.args:
        name = request.args['name']

    return model.save_parameters(name)

# Load parameters
@app.route('/load_parameters', methods=['GET'])
def load_parameters_route():
    name = 'cp'
    if 'name' in request.args:
        name = request.args['name']

    return model.load_parameters(name)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
