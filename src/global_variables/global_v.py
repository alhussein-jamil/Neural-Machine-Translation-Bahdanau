from pathlib import Path
import torch

DATA_DIR = Path("data/local_data")
EXT_DATA_DIR = Path("data/external_data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STATS = {}
WEIGHTS_STATS = {}

def stylish_stat_print(stats):
    """
    Prints the statistics of the tensors in the STATS dictionary in a stylish way. 
    where every value in STATS represents a dictionary of different statistics.
    """
    for key, value in stats.items():
        print(f"{key}:")
        for stat, val in value.items():
            print(f"\t{stat}: {val}")
def tensor_statistics(tensor):
    """
    Computes the mean and standard deviation of a tensor.

    Parameters:
        tensor (torch.Tensor): Input tensor.

    Returns:
        mean (float): Mean of the tensor.
        std (float): Standard deviation of the tensor.
    """
    if tensor is None:
        return None
    tensor = tensor.float()
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    minimum = torch.min(tensor)
    maximum = torch.max(tensor)
    return {"mean": mean, "std": std, "min": minimum, "max": maximum
    }


def weights_statistics(model : torch.nn.Module):
    """
    Computes the mean and standard deviation of the weights of a model.

    Parameters:
        model (torch.nn.Module): Input model.

    Returns:
        mean (float): Mean of the weights.
        std (float): Standard deviation of the weights.
    """
    mean = 0
    std = 0
    minimum = 0
    maximum = 0
    for param in model.parameters():
        mean += torch.mean(param.data)
        std += torch.std(param.data)
        minimum += torch.min(param.data)
        maximum += torch.max(param.data)
    mean /= len(list(model.parameters()))
    std /= len(list(model.parameters()))
    minimum /= len(list(model.parameters()))
    maximum /= len(list(model.parameters()))
    return {"mean": mean, "std": std, "min": minimum, "max": maximum}


def checkNans(name, input):
    """
    Checks if there are any NaNs in a tensor.

    Parameters:
        name (str): Name of the tensor.
        input: a tensor or a model.
    """
    if input is None:
        return False
    if isinstance(input, torch.nn.Module):
        for param in input.parameters():
            if torch.isnan(param).any():
                print(f"NaNs in model {name}")
                return True
    else:
        tensor = input
        if torch.isnan(tensor).any():
            print(f"NaNs in tensor {name}") 
            return True