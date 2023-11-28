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
        model.load_state_dict(torch.load(weights_path))

        return 'Model loaded successfully.', params, dataset, model
    except Exception as e:
        return f'Model failed to load: {e}', None, None, None

def save_model(model: GPTLanguageModel, params: Hyperparameters, name=None):
    if name == None:
        name = params.name

    checkpoint_path = os.path.join(script_dir, '../checkpoints', name)

    # save weights
    weights_path = os.path.join(checkpoint_path, 'weights.pth')
    torch.save(model.state_dict(), weights_path)

    # save params
    params_path = os.path.join(checkpoint_path, 'params.json')
    with open(params_path, 'w') as f:
        json.dump(params.config, f, indent=4)

    # save summary log
    summary_path = os.path.join(checkpoint_path, 'summary.log')
    save_summary_log(summary_path)

    return 'Model state saved successfully.'
