import os
import torch
import networkx as nx
import pandas as pd
from os.path import exists
from torch.utils.data import Dataset, DataLoader, random_split
from utils.device import get_device, device_data_loader

class GraphDataset(Dataset):
    def __init__(self, graph:nx.Graph, gamma: int, walk_length: int, window_size: int):
        self.window_size = window_size
        self.graph = graph
        self.gamma = gamma
        self.walk_length = walk_length
        
        if not exists("./datasets"):
            os.mkdir("./datasets")
        self.dataset = self.generate_dataset()
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return self.dataset[idx]

    def _get_dataset_name(self):
        return f"datasets/{self.graph.name}_{self.walk_length}_{self.window_size}_{self.gamma}.csv"

    def check_if_exists(self):
        return exists(self._get_dataset_name())

    def generate_dataset(self):
        if self.check_if_exists():
            print("Reading from stored dataset")
            dataset = pd.read_csv(self._get_dataset_name())
            return [(torch.tensor(list(map(int, x[0][1:-1].split(",")))), torch.tensor(x[1])) for x in dataset.values]
        walks = self.generate_walks()
        dataset = []

        # half window size
        hw_size = (self.window_size - 1) // 2
        for walk in walks:
            for i in range(hw_size, self.walk_length - hw_size):
                target = walk[i]
                context = walk[i - hw_size:i] + walk[i + 1:i + hw_size + 1]
                dataset.append((context, target))

        pd.DataFrame(columns=["context", "target"], data=dataset)\
            .to_csv(self._get_dataset_name(), index=False)

        dataset = [(torch.tensor(context), torch.tensor(target)) for context, target in dataset]

        return dataset

    def generate_walks(self) -> list[list[int]]:
        walks = []
        for _ in range(self.gamma):
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

    print(f"Dataset size: {len(dataset)} walks")
    generator = torch.Generator().manual_seed(42)
    train_ds, test_ds = random_split(dataset, [0.8, 0.2], generator=generator)

    train_ds = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_ds = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = get_device()
    train_ds = device_data_loader(device, train_ds)
    test_ds = device_data_loader(device, test_ds)

    return train_ds, test_ds