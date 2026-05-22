"""Activation Verbalizer.

Pipeline:
    1. KMeans on activations → cluster ids.
    2. For each cluster, find top-K texts closest to its centroid.
    3. Run TF-IDF over those texts to surface distinctive keywords.
    4. Format a short natural-language explanation per cluster.
    5. Assign each activation the explanation of its cluster.

This is a simplification of Anthropic's learned verbalizer, but produces
genuinely text-form, content-bearing explanations and is fully reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class ClusterInfo:
    cluster_id: int
    size: int
    keywords: list[str]
    representative_texts: list[str]   # closest-to-centroid samples
    explanation: str                  # the natural-language verbalization


def fit_kmeans(
    activations: np.ndarray,
    num_clusters: int,
    n_init: int,
    seed: int,
) -> tuple[KMeans, np.ndarray]:
    """Fit KMeans; return the model and an int array of cluster assignments."""
    km = KMeans(
        n_clusters=num_clusters,
        random_state=seed,
        n_init=n_init,
    )
    labels = km.fit_predict(activations)
    return km, labels


def top_k_per_cluster(
    activations: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    k: int,
) -> dict[int, list[int]]:
    """For each cluster, return indices of the k closest samples to its centroid."""
    out: dict[int, list[int]] = {}
    for c in range(centroids.shape[0]):
        idxs = np.where(labels == c)[0]
        if len(idxs) == 0:
            out[c] = []
            continue
        # Euclidean distance to centroid (KMeans optimizes squared L2)
        d = np.linalg.norm(activations[idxs] - centroids[c], axis=1)
        order = np.argsort(d)[: min(k, len(idxs))]
        out[c] = idxs[order].tolist()
    return out


def _keywords_for_cluster(
    cluster_texts: list[str],
    all_texts: list[str],
    top_n: int = 6,
) -> list[str]:
    """Use TF-IDF (cluster vs whole corpus) to pick distinctive keywords."""
    if not cluster_texts:
        return []
    # Fit on the whole corpus so IDF reflects what's common globally.
    vec = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    try:
        vec.fit(all_texts)
    except ValueError:
        return []  # too few documents

    tfidf = vec.transform(cluster_texts)
    # Mean TF-IDF per term across the cluster's representative texts.
    mean_scores = np.asarray(tfidf.mean(axis=0)).ravel()
    vocab = np.array(vec.get_feature_names_out())
    order = np.argsort(-mean_scores)
    picks: list[str] = []
    for i in order:
        term = vocab[i]
        if mean_scores[i] <= 0:
            break
        # Avoid near-duplicates (e.g. "king" and "king of")
        if any(term in p or p in term for p in picks):
            continue
        picks.append(term)
        if len(picks) >= top_n:
            break
    return picks


def _format_explanation(keywords: list[str]) -> str:
    """Turn keywords into a short English sentence."""
    if not keywords:
        return "This activation is associated with miscellaneous, low-signal content."
    head = ", ".join(keywords[:-1])
    tail = keywords[-1]
    if head:
        body = f"{head}, and {tail}"
    else:
        body = tail
    return f"This activation is related to: {body}."


def build_cluster_explanations(
    activations: np.ndarray,
    texts: list[str],
    num_clusters: int,
    n_init: int,
    top_k_examples: int,
    seed: int,
) -> tuple[np.ndarray, list[ClusterInfo]]:
    """End-to-end: cluster, find reps, build explanations.

    Returns
    -------
    labels : (N,) int array
    cluster_infos : list of ClusterInfo (one per cluster, ordered by id)
    """
    if len(texts) != activations.shape[0]:
        raise ValueError(
            f"texts ({len(texts)}) and activations ({activations.shape[0]}) "
            "must have the same length."
        )

    km, labels = fit_kmeans(activations, num_clusters, n_init, seed)
    reps = top_k_per_cluster(activations, labels, km.cluster_centers_, top_k_examples)

    infos: list[ClusterInfo] = []
    for c in range(num_clusters):
        rep_idxs = reps[c]
        rep_texts = [texts[i] for i in rep_idxs]
        keywords = _keywords_for_cluster(rep_texts, texts)
        explanation = _format_explanation(keywords)
        infos.append(
            ClusterInfo(
                cluster_id=c,
                size=int((labels == c).sum()),
                keywords=keywords,
                representative_texts=rep_texts,
                explanation=explanation,
            )
        )
    return labels, infos
