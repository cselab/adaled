import torch.nn

def any_in(list, name):
    """Returns whether any element of `list` is part of `name`."""
    return any(element in name for element in list)


def init_params(name, params):
    """Apply custom initialiations to various networks parameters.

    The network and parameter types are automatically determined from the
    parameter name.
    """
    if 'sigma' in name:
        if 'weight' in name:
            if params.data.ndim > 1:
                msg = "xavier_uniform_"
                torch.nn.init.xavier_uniform_(params.data)
            else:
                msg = "default"
        elif 'bias' in name:
            msg = "fill_(0.0001)"
            params.data.fill_(0.0001)
    elif 'weight_ih' in name:
        msg = "xavier_uniform_"
        torch.nn.init.xavier_uniform_(params.data)
    elif 'weight_hh' in name:
        msg ="orthogonal_"
        torch.nn.init.orthogonal_(params.data)
    elif any_in([
            'Wxi.weight',
            'Wxf.weight',
            'Wxc.weight',
            'Wxo.weight',
            'Bci',
            'Bcf',
            'Bco',
            'Bcc',
            ], name):
        msg = "xavier_uniform_"
        torch.nn.init.xavier_uniform_(params.data)
    elif any_in([
            'Wco', 'Wcf', 'Wci',
            'Whi.weight', 'Whf.weight',
            'Whc.weight', 'Who.weight',
            ], name):
        msg = "orthogonal_"
        torch.nn.init.orthogonal_(params.data)
    elif any_in([
            'Whi.bias', 'Wxi.bias', 'Wxf.bias', 'Whf.bias',
            'Wxc.bias', 'Whc.bias', 'Wxo.bias', 'Who.bias',
            ], name):
        msg = "fill_(0.001)"
        params.data.fill_(0.001)
    elif 'weight' in name:
        if params.data.ndim > 1:
            msg = "xavier_uniform_"
            torch.nn.init.xavier_uniform_(params.data)
        else:
            msg = "default"
            pass
    elif 'bias' in name:
        msg = "fill_(0.001)"
        params.data.fill_(0.001)
    elif 'initial_hidden_state' in name:
        msg = "fill_(0.00001)"
        params.data.fill_(0.00001)
    else:
        raise ValueError(f"parameter name '{name}' not recognized")

    # print(f"    {name}: {msg}")


def apply_custom_initialization(model: torch.nn.Module):
    # print(f"Initializing `{model.__class__.__name__}` model parameters:")
    for name, params in model.named_parameters():
        try:
            init_params(name, params)
        except:
            print(f"Error while initializing parameters '{name}'.")
            raise
