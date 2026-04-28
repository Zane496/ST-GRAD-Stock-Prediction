import pandas as pd
import numpy as np
import os

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def build_adjacency_matrix(cluster_file, output_adj_path, output_order_path):
    """Build and save normalized adjacency matrix from trend clustering results."""
    if not os.path.exists(cluster_file):
        raise FileNotFoundError(f"Cluster file not found: {cluster_file}")

    df = pd.read_csv(cluster_file)
    stocks = df['Stock'].values
    trend_clusters = df['Trend'].values
    num_stocks = len(stocks)

    adj_matrix = np.zeros((num_stocks, num_stocks))
    for i in range(num_stocks):
        for j in range(num_stocks):
            if i == j or trend_clusters[i] == trend_clusters[j]:
                adj_matrix[i, j] = 1

    normalized_adj = normalize_adj(adj_matrix)

    os.makedirs(os.path.dirname(output_adj_path), exist_ok=True)
    np.save(output_adj_path, normalized_adj)
    np.save(output_order_path, stocks, allow_pickle=True)

    return normalized_adj, stocks