import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from networkx import Graph
from sklearn.cluster import KMeans
from utils.EarlyStopping import EarlyStopping
from utils.parse_args import parse_args

args = parse_args()

def _print(*pargs):
    if args['verbose']:
        print(*pargs)

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
            _print(f"\t [{int((i + 1) / len(train_loader) * 100)}%] Loss: {current_loss / batch_perc:.4f}")
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

    estop = EarlyStopping(patience=5, delta=0.0001)
    curr_lr = optimizer.param_groups[0]['lr'] if optimizer else 0
    for epoch in range(epochs):
        _print(f"============== Epoch {epoch + 1} ==============")
        train_acc, train_loss = train_epoch(model, optimizer, train_loader, loss_fn, batch_size)
        test_acc, test_loss = test_epoch(model, test_loader, loss_fn, batch_size)
        _print(f'Train Loss: {train_loss:.4f} \tTest Loss: {test_loss:.4f}')
        _print(f'Train Accuracy: {(train_acc * 100):.4f}% \tTest Accuracy: {(test_acc * 100):.4f}%')
        if scheduler:
            scheduler.step()
            after_lr = optimizer.param_groups[0]['lr']
            if after_lr != curr_lr:
                _print(f'Learning rate: {curr_lr} -> {after_lr}')
            else:
                _print(f'Learning rate: {curr_lr}')
            curr_lr = after_lr

        if epoch == epochs - 1:
            print(f'Final Test Accuracy: {(test_acc * 100):.4f}% \n')
            print(f'Final Train Accuracy: {(train_acc * 100):.4f}% \n')
        if estop(train_loss):
            print(f'Final Test Accuracy: {(test_acc * 100):.4f}%')
            print(f'Final Train Accuracy: {(train_acc * 100):.4f}%')
            print("Early stopping - Last epoch:", epoch + 1)
            break

def extract_embeddings(model: nn.Module):
    if torch.cuda.is_available():
        model = model.cpu()
    return model.embeddings.weight.detach().numpy()

def KMeans_Accuracy(reduced_embeddings, karate_graph: Graph):
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(reduced_embeddings)
    labels = kmeans.labels_

    nodes = list(karate_graph.nodes(data=True))
    correct = 0
    correct_flip = 0
    for i, (_, data) in enumerate(nodes):
        actual = 0 if data["club"] == "Mr. Hi" else 1
        predicted = labels[i]
        if predicted == actual:
            correct += 1
        if (1 - predicted) == actual:
            correct_flip += 1
    correct = max(correct, correct_flip)
    acc = correct / len(nodes)

    print(f'Accuracy w/ KMeans: {acc * 100:.2f}%')    
    return acc
