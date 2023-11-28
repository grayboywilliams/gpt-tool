from src.params import *
from src.dataset import *
from src.gpt import *
from src.logger import *
from constants.constants import *

def load_model(logger: Logger, name=None, download_data=False):
    if name == None:
        name = 'temp'
    
    reset_summary_log()

    try:
        # init model
        params = Hyperparameters(logger, name)
        dataset = Dataset(logger, params, download_data)
        model = GPTLanguageModel(logger, params, dataset)

        # load weights
        weights_path = os.path.join(script_dir, '../checkpoints', name, 'weights.pth')
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path))

        return f'Model "{name}" state loaded successfully.', params, dataset, model
    except Exception as e:
        return f'Model "{name}" failed to load: {e}', None, None, None

def save_model(model: GPTLanguageModel, params: Hyperparameters, name=None):
    if name == None:
        name = params.name

    try:
        checkpoint_path = os.path.join(script_dir, '../checkpoints', name)

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

        return f'Model "{name}" state saved successfully.'
    except Exception as e:
        return f'Model "{name}" failed to save: {e}'
