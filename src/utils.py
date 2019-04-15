import torch

def torch_to_numpy(tensor):
    """
    input: torch tensor
    output: numpy array
    """
    
    return tensor.detach().cpu().numpy().squeeze().transpose(1,2,0)