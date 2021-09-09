import torch

from typing import Sequence, Tuple

__all__ = [
    'describe_layers', 'estimate_model_num_params', 'get_layer_dimensions',
    'print_model_info',
]

def describe_layers(obj, layers, input_shape):
    shapes = get_layer_dimensions(layers, input_shape)
    print(f"{obj.__class__.__name__} layers and tensor shapes:")
    print("ID  SHAPE              #PARAMS  LAYER")
    print("   ", shapes[0])
    total_params = 0
    for i, (layer, shape) in enumerate(zip(layers, shapes[1:])):
        shape = tuple(shape)
        num_params = sum(params.numel() for params in layer.parameters())
        total_params += num_params
        print(f"#{i:<2} {str(shape):<18} {num_params:>6}  {layer}")
    print("Total params:", total_params)
    print()


def estimate_model_num_params(model: torch.nn.Module):
    """Estimate the number of parameter of a neural network."""
    # https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    unique = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            unique[p.data_ptr()] = p.numel()
    total = sum(unique.values())
    return total


def get_layer_dimensions(layers, input_shape) -> Sequence[Tuple[int, ...]]:
    """Evaluate the layers on a random input to determine the output size after
    each layer.

    Useful for determining the state size after a convoluational network.

    Returns:
        list of shapes
    """
    x = torch.randn(1, *input_shape)
    sizes = [input_shape]
    for layer in layers:
        x = layer(x)
        sizes.append(x.shape[1:])
    return sizes


def print_model_info(model: torch.nn.Module):
    """Print network parameter names and shapes, and the total number of
    parameters."""
    # https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    print(f"Model info: {model}")
    print("Parameters:")
    unique = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        unique[p.data_ptr()] = p.numel()
        print(f"    {name + ':':<40} {str(tuple(p.shape)):<15} {p.numel():>6}")
    total = sum(unique.values())
    print(f"Total unique parameters: {total}")
