import json
import torch
import itertools
import networkx as nx
import torch.optim as optim
from CBOW import CBOW
from dataset import build_vocab, generate_dataset
from sklearn.manifold import TSNE
from utils.parse_args import cbow_args, parse_args
from utils.training_utils import extract_embeddings, train, KMeans_Accuracy
from utils.visualize import visualize_embeddings
from utils.device import get_device

full_args = parse_args()
args = cbow_args()

# Generate dataset
G = nx.karate_club_graph()
vocab = build_vocab(G)

param_grid = args if args is not None else {
      "window_size": [3, 5, 7],
      "walk_length": [0, 2, 4],
      "num_walks": [1000, 2000],
      "embedding_dim": [12, 16, 24, 32],
      "batch_size": [256],
}
if args is None:
      print("Total combinations:", len(param_grid["window_size"]) * len(param_grid["walk_length"]) * len(param_grid["num_walks"]) * len(param_grid["embedding_dim"]))

accs = []
best_acc = 0.0
best_params = None

for w_size, w_len, n_walks, emb_dim, b_size in itertools.product(
      param_grid["window_size"],
      param_grid["walk_length"],
      param_grid["num_walks"],
      param_grid["embedding_dim"],
      param_grid["batch_size"],
):
      print("Current running configuration:")
      print(json.dumps({
            "window_size": w_size,
            "walk_length": w_size + w_len,
            "num_walks": n_walks,
            "embedding_dim": emb_dim,
            "batch_size": b_size,
            "epochs": full_args["epochs"],
      }))
      train_loader, test_loader = generate_dataset(
            graph=G,
            window_size=w_size,
            walk_length=w_size + w_len,
            num_walks=n_walks,
            batch_size=b_size
      )
      model = CBOW(len(vocab), embedding_dim=emb_dim).to(get_device())
      optimizer = optim.Adam(model.parameters(), lr=1e-4)

      train(
            model,
            optimizer,
            train_loader,
            test_loader,
            batch_size=b_size,
            epochs=full_args["epochs"],
            loss_fn=torch.nn.CrossEntropyLoss(),
      )

      reduced_embeds = TSNE(n_components=2, random_state=42).fit_transform(
            extract_embeddings(model)
      )
      acc = KMeans_Accuracy(reduced_embeds, G)
      accs.append((acc, w_size, w_len, n_walks, emb_dim))
      if acc > best_acc:
            best_acc = acc
            best_params = (w_size, w_len, n_walks, emb_dim)

      if args is not None:
            visualize_embeddings(reduced_embeds, G.nodes())
      
if args is None:
      # Sort the accuracies in descending order based on the first element of the tuple
      print(sorted(accs, key=lambda x: x[0], reverse=True))
      print(f"Best configuration: {best_params} with KMeans accuracy {best_acc * 100:.2f}%")
      