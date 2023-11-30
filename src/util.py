import json
from graphviz import Digraph
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

def model_summary(model: nn.Module, depth=0, parent_name='', full=False):
    hierarchy = []
    total_params = 0
    total_trainable_params = 0
    total_bytes = 0
    summary = {}

    for name, module in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        num_parameters = sum(p.numel() for p in module.parameters(recurse=False))
        num_trainable_parameters = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        bytes_used = sum(p.numel() * p.element_size() for p in module.parameters(recurse=False) if p.requires_grad)

        layer_info = {
            "name": full_name,
            "type": module.__class__.__name__,
            "depth": depth,
            "num_parameters": num_parameters,
            "num_trainable_parameters": num_trainable_parameters,
            "bytes": bytes_used
        }

        if full:
            hierarchy.append(layer_info)
        total_params += num_parameters
        total_trainable_params += num_trainable_parameters
        total_bytes += bytes_used

        # Recursive call to process all submodules
        children = model_summary(module, depth=depth+1, parent_name=full_name, full=full)
        if full:
            hierarchy.extend(children['hierarchy'])
        total_params += children['total_parameters']
        total_trainable_params += children['total_trainable_parameters']
        total_bytes += children['total_bytes']

    if full:
        summary["hierarchy"] = hierarchy
    summary["total_parameters"] = total_params
    summary["total_trainable_parameters"] = total_trainable_params
    summary["total_bytes"] = total_bytes

    return summary

def draw_model(model: GPTLanguageModel):
    dot = Digraph(model.params.name, format='png')
    dot.attr(rankdir='LR') # Layout from Left to Right

    def add_nodes(module: nn.Module, parent_name: str):
        total_params = sum(p.numel() for p in module.parameters(recurse=False))
        bytes_used = sum(p.numel() * p.element_size() for p in module.parameters(recurse=False))

        for n, m in module.named_children():
            child_name = f"{parent_name}.{n}" if parent_name else n
            child_params, child_bytes = add_nodes(m, child_name)
            total_params += child_params
            bytes_used += child_bytes

            if parent_name: # Connect nodes only if they are not the root
                dot.edge(parent_name, child_name)

        if parent_name: # Avoid adding the root model as a node
            label = f"{parent_name.split('.')[-1]}\n({module.__class__.__name__})\nParams: {total_params}"
            dot.node(name=parent_name, label=label, shape='box')

        return total_params, bytes_used

    total_params, total_bytes = add_nodes(model, model.params.name)

    # Adding summary info to root node
    root_label = (f"{model.params.name}\n({model.__class__.__name__})\n" +
        f"Total Modules: {len(list(model.modules()))}\n" +
        f"Total Parameters: {total_params}\n" +
        f"Total Bytes: {total_bytes}")
    dot.node(name=model.params.name, label=root_label, shape='box')

    file_path = os.path.join(script_dir, checkpoints, model.params.name, 'model')
    dot.render(file_path, cleanup=True)
