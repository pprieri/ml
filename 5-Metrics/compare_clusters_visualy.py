from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

results = #contains the clusters

def compare_clusterings(y1, y2, title=''):
    """Show the adjusted rand score and plot the two clusterings in color"""
    ars = adjusted_rand_score(y1, y2)
    n1 = y1.max() + 1
    n2 = y2.max() + 1
    argsort = np.argsort(y1*100 + y2) if n1 >= n2 else np.argsort(y2*100 + y1)
    plt.figure(figsize=(16, 0.5))
    for i in range(6, 11):
        plt.scatter(np.arange(len(y1)), np.full_like(y1, i), c=y1[argsort], s=1, cmap='tab10')
    for i in range(5):
        plt.scatter(np.arange(len(y2)), np.full_like(y2, i), c=y2[argsort], s=1, cmap='tab10')
    plt.gca().axis('off')
    plt.title(f'{title}\nAdjusted Rand score: {ars:.5f}')
    plt.savefig(title + '.png', bbox_inches='tight')
    plt.show()

compare_clusterings(results.loc[8, 0].y, 7 - results.loc[8, 0].y, '8 clusters vs. relabeling')
