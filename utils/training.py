import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_epoch(model: nn.Module, optimizer: optim.Optimizer,
        train_loader: DataLoader, loss_fn: nn.Module, batch_size: int):

    model.train()
    total_loss = 0
    current_loss = 0
    batch_perc = len(train_loader) // 10
    correct = 0
    for i, (context, target) in enumerate(train_loader):
        # context, target = load_to_device(context, target, device)
        optimizer.zero_grad()
        pred = model(context)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        
        ls = loss.item()
        current_loss += ls
        total_loss += ls
        correct += torch.sum(torch.argmax(pred, dim=1) == target).item()
        if (i + 1) % batch_perc == 0:
            print(f"\t [{int((i + 1) / len(train_loader) * 100)}%] Loss: {current_loss / batch_perc:.4f}")
            current_loss = 0
    accuracy = correct / len(train_loader) / batch_size
    return accuracy, total_loss / len(train_loader)

def test_epoch(model: nn.Module, test_loader: DataLoader, loss_fn: nn.Module, batch_size: int):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for context, target in test_loader:
            # context, target = load_to_device(context, target, device)
            pred = model(context)
            loss = loss_fn(pred, target)
            total_loss += loss.item()
            correct += torch.sum(torch.argmax(pred, dim=1) == target).item()
    accuracy = correct / len(test_loader) / batch_size
    return accuracy, total_loss / len(test_loader)

def train(model: nn.Module, optimizer: optim.Optimizer, 
          train_loader: DataLoader, test_loader: DataLoader, loss_fn: nn.Module,
          batch_size: int, epochs: int, scheduler=None):
    
    curr_lr = optimizer.param_groups[0]['lr'] if optimizer else 0
    for epoch in range(epochs):
        print(f"============== Epoch {epoch + 1} ==============")
        train_acc, train_loss = train_epoch(model, optimizer, train_loader, loss_fn, batch_size)
        test_acc, test_loss = test_epoch(model, test_loader, loss_fn, batch_size)
        print(f'Train Loss: {train_loss:.4f} \tTest Loss: {test_loss:.4f}')
        print(f'Train Accuracy: {(train_acc * 100):.4f}% \tTest Accuracy: {(test_acc * 100):.4f}%')
        if scheduler:
            scheduler.step()
            after_lr = optimizer.param_groups[0]['lr']
            if after_lr != curr_lr:
                print(f'Learning rate: {curr_lr} -> {after_lr}')
            else:
                print(f'Learning rate: {curr_lr}')
            curr_lr = after_lr

def extract_embeddings(model: nn.Module):
    if torch.cuda.is_available():
        model = model.cpu()
    return model.embeddings.weight.detach().numpy()