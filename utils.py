import torch

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_to_device(X, y, device=None):
    if device is None:
        device = get_device()
    return X.to(device), y.to(device)

def device_data_loader(device: torch.device, dataloader: torch.utils.data.DataLoader) -> list:
    new_dataloader = []
    for X, y in dataloader:
        new_dataloader.append(load_to_device(X, y, device))
    return new_dataloader