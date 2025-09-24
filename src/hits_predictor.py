
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from networkx.exception import PowerIterationFailedConvergence

SEED = 42

def make_clusters(Z_train, y_train, k_min=8, k_max=40):
    km_k = int(np.clip(np.sqrt(len(Z_train) / 200), k_min, k_max))
    print(f"[kmeans] using n_clusters={km_k}")
    km = KMeans(n_clusters=km_k, random_state=SEED, n_init=10)
    labels = km.fit_predict(Z_train)

    y_arr = np.asarray(y_train).astype(int)
    clusters = sorted(np.unique(labels))
    cluster_classes, cluster_mal_ratio = {}, {}

    print("\n=== Cluster Stats (KMeans) ===")
    print(f"{'Cluster':<8} {'Samples':<8} {'Percent':<8} {'Majority':<10} {'Mal.Ratio':<10}")
    N = len(labels)
    for c in clusters:
        idx = (labels == c)
        cnt = int(idx.sum())
        if cnt == 0:
            maj, ratio = 0, 0.0
        else:
            lbls = y_arr[idx]
            maj = np.bincount(lbls).argmax()
            ratio = float(lbls.mean())
        cluster_classes[c] = int(maj)
        cluster_mal_ratio[c] = float(ratio)
        print(f"{c:<8} {cnt:<8} {100.0*cnt/max(1,N):7.2f}% {maj:<10} {ratio:<10.3f}")

    return km, labels, cluster_classes, cluster_mal_ratio

def hits_predict_with_proba(
    z_sample, Z_train, y_train_arr, cluster_labels, cluster_classes, cluster_mal_ratio,
    kmeans, k=12, sim_edge=0.80
):
    sample_cluster = int(kmeans.predict(z_sample.reshape(1, -1))[0])

    idx = np.where(cluster_labels == sample_cluster)[0]
    if len(idx) == 0:
        p = float(cluster_mal_ratio.get(sample_cluster, 0.0))
        return (1 if p >= 0.5 else 0), p

    Z_cluster = Z_train[idx]
    sims = cosine_similarity(z_sample.reshape(1, -1), Z_cluster)[0]
    k_eff = int(min(k, len(idx)))
    top_k_idx = sims.argsort()[-k_eff:]
    neighbor_idx = idx[top_k_idx]
    neighbor_sims = sims[top_k_idx]

    if len(neighbor_idx) == 0:
        p = float(cluster_mal_ratio.get(sample_cluster, 0.0))
        return (1 if p >= 0.5 else 0), p

    G = nx.DiGraph()
    G.add_node("sample")
    for i, ni in enumerate(neighbor_idx):
        w = float(neighbor_sims[i])
        G.add_edge("sample", int(ni), weight=w)
        G.add_edge(int(ni), "sample", weight=w)

    for i in range(len(neighbor_idx)):
        zi = Z_train[neighbor_idx[i]].reshape(1, -1)
        for j in range(i + 1, len(neighbor_idx)):
            zj = Z_train[neighbor_idx[j]].reshape(1, -1)
            sim_ij = float(cosine_similarity(zi, zj)[0, 0])
            if sim_ij >= sim_edge:
                ni, nj = int(neighbor_idx[i]), int(neighbor_idx[j])
                G.add_edge(ni, nj, weight=sim_ij)
                G.add_edge(nj, ni, weight=sim_ij)

    try:
        hubs, authorities = nx.hits(G, max_iter=200, normalized=True)
        auth_neighbors = np.array([authorities.get(int(ni), 0.0) for ni in neighbor_idx], dtype=float)
        if auth_neighbors.sum() == 0:
            p = float(cluster_mal_ratio.get(sample_cluster, 0.0))
        else:
            y_neighbors = np.array([int(y_train_arr[ni]) for ni in neighbor_idx], dtype=int)
            p = float(auth_neighbors[y_neighbors == 1].sum() / auth_neighbors.sum())
    except PowerIterationFailedConvergence:
        weights = np.maximum(neighbor_sims, 0.0)
        if weights.sum() == 0:
            p = float(cluster_mal_ratio.get(sample_cluster, 0.0))
        else:
            y_neighbors = np.array([int(y_train_arr[ni]) for ni in neighbor_idx], dtype=int)
            p = float(weights[y_neighbors == 1].sum() / weights.sum())

    return (1 if p >= 0.5 else 0), p
