import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def visualize_embeddings(embeddings, labels):
    tsne = TSNE(n_components=2, random_state=0)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i, label in enumerate(labels):
        x, y = reduced_embeddings[i, :]
        plt.scatter(x, y)
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
        
    # Draw a line from a LinearRegression model for the embeddings
    x = reduced_embeddings[:, 0]
    y = reduced_embeddings[:, 1]
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x + b, color='red')
    
    plt.show()

def visualize_graph(G):
    nx.draw(G, with_labels=True)
    plt.show()