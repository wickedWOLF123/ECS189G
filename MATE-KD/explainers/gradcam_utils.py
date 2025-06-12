import torch.nn as nn

def set_inplace(model, inplace):
    """
    Recursively sets the 'inplace' property of all ReLU modules in a model.

    Args:
        model (nn.Module): The model to modify.
        inplace (bool): The value to set for the 'inplace' property.
    """
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = inplace 