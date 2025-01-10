from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_embeddings(embeddings, labels):
    tsne = TSNE(n_components=2, random_state=0)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i, label in enumerate(labels):
        x, y = reduced_embeddings[i, :]
        plt.scatter(x, y)
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.show()
