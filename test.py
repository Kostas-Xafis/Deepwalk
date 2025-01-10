import torch
import networkx as nx
from CBOW import CBOW
from dataset import build_vocab, generate_dataset
from visualize import visualize_embeddings

def train_epoch(model, optimizer, train_loader, loss_fn):
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
    accuracy = correct / len(train_loader)
    return accuracy, total_loss / len(train_loader)

def test_epoch(model, test_loader, loss_fn):
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
    accuracy = correct / len(test_loader)
    return accuracy, total_loss / len(test_loader)

def train(model, optimizer, train_loader, test_loader, loss_fn, scheduler = None, epochs = 1):
    curr_lr = optimizer.param_groups[0]['lr'] if optimizer else 0
    for epoch in range(epochs):
        print(f"============== Epoch {epoch + 1} ==============")
        train_acc, train_loss = train_epoch(model, optimizer, train_loader, loss_fn)
        test_acc, test_loss = test_epoch(model, test_loader, loss_fn)
        print(f'Train Loss: {train_loss:.4f} \tTest Loss: {test_loss:.4f}')
        print(f'Train Accuracy: {train_acc:.4f} \tTest Accuracy: {test_acc:.4f}')
        if scheduler:
            scheduler.step()
            after_lr = optimizer.param_groups[0]['lr']
            print(f'Learning rate: {curr_lr} -> {after_lr}')
            curr_lr = after_lr

# Generate dataset
G = nx.karate_club_graph()
vocab = build_vocab(G)
train_loader, test_loader = generate_dataset(graph=G, window_size=3, walk_length=4, num_walks=1000, batch_size=64)
model = CBOW(len(vocab), 128)
optimizer = torch.optim.Adam(model.parameters(), lr=10e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

train(model, optimizer, train_loader, test_loader, scheduler=scheduler,
      epochs=100, loss_fn=torch.nn.CrossEntropyLoss())

# Λήψη των embeddings από το μοντέλο
embeddings = model.embeddings.weight.detach().numpy()
labels = range(len(vocab))

# Οπτικοποίηση
visualize_embeddings(embeddings, labels)
