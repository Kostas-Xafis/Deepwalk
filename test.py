import torch
import networkx as nx
import torch.optim as optim
from CBOW import CBOW
from dataset import build_vocab, generate_dataset
from utils.training import extract_embeddings, train
from utils.visualize import visualize_embeddings, visualize_graph
from utils.device import get_device

batch_size = 8192

# Generate dataset
G = nx.karate_club_graph()
# visualize_graph(G)

vocab = build_vocab(G)
train_loader, test_loader = generate_dataset(graph=G, window_size=3, walk_length=4, num_walks=4000, batch_size=batch_size)
model = CBOW(len(vocab), embedding_dim=12).to(get_device())

optimizer = optim.Adam(model.parameters(), lr=10e-3)
scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=10e-5, max_lr=10e-2, step_size_up=100, cycle_momentum=True)

train(model, optimizer, train_loader, test_loader,
      batch_size=batch_size, epochs=100, loss_fn=torch.nn.NLLLoss())

# Extract embeddings and visualize
embeddings = extract_embeddings(model)
labels = range(len(vocab))

visualize_embeddings(embeddings, labels)

