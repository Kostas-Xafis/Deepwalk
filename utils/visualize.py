import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def visualize_embeddings(reduced_embeddings, labels):
    # Perform k-means clustering with 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=0).fit(reduced_embeddings)
    clusters = kmeans.labels_

    plt.figure(figsize=(10, 10))
    for i, label in enumerate(labels):
        x, y = reduced_embeddings[i, :]
        color = 'red' if clusters[i] == 0 else 'blue'
        plt.scatter(x, y, c=color)
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.show()

def visualize_graph(G):
    nx.draw(G, with_labels=True)
    plt.show()
