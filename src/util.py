import json
from src.params import *
from src.dataset import *
from src.gpt import *
from src.logger import *
from constants.constants import *

script_dir = os.path.dirname(os.path.realpath(__file__))

def load_model(logger: Logger, cpt_name: str, params: Hyperparameters=None):
    reset_summary_log()

    try:
        # init model
        params = Hyperparameters(logger, cpt_name)
        dataset = Dataset(logger, params)
        model = GPTLanguageModel(logger, params, dataset)

        # load weights
        weights_path = os.path.join(script_dir, checkpoints, cpt_name, 'weights.pth')
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path))

        return f'Model "{cpt_name}" state loaded successfully.', params, dataset, model
    except Exception as e:
        return f'Model "{cpt_name}" failed to load: {e}', None, None, None

def save_model(model: GPTLanguageModel, params: Hyperparameters):
    try:
        cpt_name = params.name
        checkpoint_path = os.path.join(script_dir, checkpoints, cpt_name)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        # save params
        params_path = os.path.join(checkpoint_path, 'params.json')
        with open(params_path, 'w') as f:
            json.dump(params.file_params, f, indent=4)

        # save summary
        summary_path = os.path.join(checkpoint_path, 'summary.log')
        save_summary_log(summary_path)

        # save weights
        weights_path = os.path.join(checkpoint_path, 'weights.pth')
        torch.save(model.state_dict(), weights_path)

        return f'Model "{cpt_name}" state saved successfully.'
    except Exception as e:
        return f'Model "{cpt_name}" failed to save: {e}'

def setup_params(params: dict):
    # Load base params
    new_model_params = load_params()

    for key in params:
        if key not in new_model_params:
            return 'Error: Invalid parameter.', False
        new_model_params[key] = params[key]

    # Setup new params
    os.makedirs(os.path.join(script_dir, checkpoints, params[name_arg]), exist_ok=True)
    params_path = os.path.join(script_dir, checkpoints, params[name_arg], 'params.json')
    with open(params_path, 'w') as f:
        json.dump(new_model_params, f, indent=4)
    return 'Setup params successfully.', True
