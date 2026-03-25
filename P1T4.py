from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import numpy as np
import os

def pca_2d(vectors):
    # PCA: center, cov, top-2 eigvecs
    mean = np.mean(vectors, axis=0)
    centered = vectors - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # sort by eigenvalues desc
    idx = np.argsort(eigenvalues)[::-1]
    top2 = eigenvectors[:, idx[:2]]
    projected = centered @ top2
    return projected

def tsne_2d(vectors, perplexity=10, lr=100.0, n_iter=1000, seed=42):
    # simple t-SNE
    np.random.seed(seed)
    n = vectors.shape[0]

    # pairwise sq distances
    def pairwise_dist(X):
        sum_sq = np.sum(X ** 2, axis=1)
        D = sum_sq[:, None] + sum_sq[None, :] - 2 * X @ X.T
        np.fill_diagonal(D, 0)
        return np.maximum(D, 0)

    # compute P via binary search
    D = pairwise_dist(vectors)
    P = np.zeros((n, n))
    target_entropy = np.log(perplexity)

    for i in range(n):
        lo, hi = 1e-10, 1e4
        for _ in range(50):
            sigma = (lo + hi) / 2
            p_row = np.exp(-D[i] / (2 * sigma ** 2))
            p_row[i] = 0
            sum_p = np.sum(p_row)
            if sum_p == 0:
                lo = sigma
                continue
            p_row /= sum_p
            entropy = -np.sum(p_row[p_row > 0] * np.log(p_row[p_row > 0]))
            if entropy > target_entropy:
                hi = sigma
            else:
                lo = sigma
        P[i] = p_row

    # symmetrize
    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12)

    # init Y
    Y = np.random.randn(n, 2) * 0.01
    velocity = np.zeros_like(Y)
    momentum = 0.5

    for it in range(n_iter):
        # compute Q
        D_y = pairwise_dist(Y)
        Q = 1.0 / (1.0 + D_y)
        np.fill_diagonal(Q, 0)
        sum_Q = np.sum(Q)
        Q = Q / max(sum_Q, 1e-12)
        Q = np.maximum(Q, 1e-12)

        # gradient
        PQ_diff = P - Q
        grad = np.zeros_like(Y)
        for i in range(n):
            diff = Y[i] - Y
            grad[i] = 4 * np.sum((PQ_diff[i] * Q[i])[:, None] * diff * (1 + D_y[i])[:, None], axis=0)

        if it > 250:
            momentum = 0.8
        velocity = momentum * velocity - lr * grad
        Y = Y + velocity

    return Y

def get_word_vectors(model, words):
    valid_words = [w for w in words if w in model.wv]
    vectors = np.array([model.wv[w] for w in valid_words])
    return valid_words, vectors


if __name__ == '__main__':
    cbow_path = 'e:/NLU2/w2v_models/cbow_d100_w5_n5.model'
    sg_path = 'e:/NLU2/w2v_models/sg_d100_w5_n5.model'

    if not os.path.exists(cbow_path):
        print("Models not found. Run P1T2.py first.")
        exit(1)

    cbow_model = Word2Vec.load(cbow_path)
    sg_model = Word2Vec.load(sg_path)

    # word groups
    word_groups = {
        'Academic': ['student', 'students', 'course', 'courses', 'semester',
                     'grade', 'credits', 'lecture', 'lectures'],
        'Research': ['research', 'phd', 'thesis', 'paper', 'journal',
                     'conference', 'publication'],
        'Department': ['computer', 'science', 'engineering', 'department',
                       'electrical', 'mechanical', 'technology'],
        'Institute': ['iit', 'jodhpur', 'institute', 'campus', 'faculty',
                      'director', 'education'],
    }

    all_words = []
    word_labels = []
    for group, words in word_groups.items():
        for w in words:
            all_words.append(w)
            word_labels.append(group)

    colors = {'Academic': 'blue', 'Research': 'red',
              'Department': 'green', 'Institute': 'orange'}

    models = {'CBOW': cbow_model, 'Skip-gram': sg_model}

    # PCA
    print("Generating PCA plots...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for idx, (model_name, model) in enumerate(models.items()):
        valid_words, vectors = get_word_vectors(model, all_words)
        coords = pca_2d(vectors)
        ax = axes[idx]
        for i, word in enumerate(valid_words):
            group = word_labels[all_words.index(word)]
            ax.scatter(coords[i, 0], coords[i, 1],
                      color=colors[group], s=60, alpha=0.7)
            ax.annotate(word, (coords[i, 0], coords[i, 1]),
                       fontsize=8, ha='center', va='bottom')
        ax.set_title(f'{model_name} - PCA', fontsize=14)
        ax.grid(True, alpha=0.3)
    for group, color in colors.items():
        axes[0].scatter([], [], color=color, label=group, s=60)
    axes[0].legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig('e:/NLU2/pca_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved pca_visualization.png")

    # t-SNE
    print("Generating t-SNE plots...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for idx, (model_name, model) in enumerate(models.items()):
        valid_words, vectors = get_word_vectors(model, all_words)
        perp = min(8, len(valid_words) - 1)
        coords = tsne_2d(vectors, perplexity=perp)
        ax = axes[idx]
        for i, word in enumerate(valid_words):
            group = word_labels[all_words.index(word)]
            ax.scatter(coords[i, 0], coords[i, 1],
                      color=colors[group], s=60, alpha=0.7)
            ax.annotate(word, (coords[i, 0], coords[i, 1]),
                       fontsize=8, ha='center', va='bottom')
        ax.set_title(f'{model_name} - t-SNE', fontsize=14)
        ax.grid(True, alpha=0.3)
    for group, color in colors.items():
        axes[0].scatter([], [], color=color, label=group, s=60)
    axes[0].legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig('e:/NLU2/tsne_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved tsne_visualization.png")

    print("\nDone.")