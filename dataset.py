import torch
import networkx as nx
import pandas as pd
from os.path import exists
from torch.utils.data import Dataset, DataLoader, random_split

from utils import get_device, load_to_device

class GraphDataset(Dataset):
    def __init__(self, graph:nx.Graph, num_walks: int, walk_length: int, window_size: int):
        self.window_size = window_size
        self.graph = graph
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.dataset = self.generate_dataset()
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def _get_dataset_name(self):
        return f"datasets/{self.graph.name}_{self.walk_length}_{self.window_size}_{self.num_walks}.csv"

    def check_if_exists(self):        
        return exists(self._get_dataset_name())

    def generate_dataset(self):
        if self.check_if_exists():
            print("Reading from stored dataset")
            dataset = pd.read_csv(self._get_dataset_name())
            return [(torch.tensor(list(map(int, x[0][1:-1].split(",")))), torch.tensor(x[1])) for x in dataset.values]
        walks = self.generate_walks()
        dataset = []
        walk_length = len(walks[0])
        half_window = (self.window_size - 1) // 2
        for walk in walks:
            for i in range(half_window, walk_length - half_window):
                target = walk[i]
                context = walk[i - half_window:i] + walk[i + 1:i + half_window + 1]
                dataset.append((context, target))

        pd.DataFrame(columns=["context", "target"], data=dataset)\
            .to_csv(self._get_dataset_name(), index=False)

        dataset = [(torch.tensor(context), torch.tensor(target)) for context, target in dataset]        

        return dataset

    def generate_walks(self) -> list[list[int]]:
        walks = []
        for _ in range(self.num_walks):
            for node in self.graph.nodes():
                walk = [node]
                for _ in range(self.walk_length - 1):
                    neighbors = list(self.graph.neighbors(node))
                    node = neighbors[torch.randint(len(neighbors), (1,)).item()]
                    walk.append(node)
                walks.append(walk)
        print("Walks generated")
        return walks

def build_vocab(G: nx.Graph) -> dict:
    vocab = {}
    for i, node in enumerate(G.nodes()):
        vocab[node] = i
    return vocab

def generate_dataset(graph: nx.Graph, walk_length: int, window_size: int, num_walks: int, batch_size=32):    
    dataset = GraphDataset(graph, num_walks, walk_length, window_size)
    # Write dataset in the dataset folder

    print(f"Dataset size: {len(dataset)}")
    generator = torch.Generator().manual_seed(42)
    train_ds, test_ds = random_split(dataset, [0.8, 0.2], generator=generator)

    train_ds = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_ds = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    device = get_device()
    train_ds = [load_to_device(context, target, device) for context, target in train_ds]
    test_ds = [load_to_device(context, target, device) for context, target in test_ds]
    
    return train_ds, test_ds