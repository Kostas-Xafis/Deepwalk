import torch.nn as nn

class CBOW(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.sf = nn.LogSoftmax(dim=1)

    def forward(self, w):
        w = self.embeddings(w)
        w = w.mean(dim=1)
        w = self.linear(w)
        w = self.sf(w)
        return w