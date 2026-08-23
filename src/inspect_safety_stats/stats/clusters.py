from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClusterBootstrapResult:
    estimate: float
    lower: float
    upper: float
    cluster_count: int
    minimum_cluster_size: int
    maximum_cluster_size: int
    cluster_size_ratio: float
    maximum_leave_one_cluster_out_shift: float
    replicates: int
    seed: int


def paired_cluster_bootstrap(
    differences: Sequence[float],
    clusters: Sequence[Hashable],
    *,
    weights: Sequence[float] | None = None,
    alpha: float = 0.05,
    replicates: int = 2000,
    seed: int = 0,
) -> ClusterBootstrapResult:
    values = np.asarray(differences, dtype=float)
    cluster_values = np.asarray(clusters, dtype=object)
    if values.ndim != 1 or cluster_values.shape != values.shape or len(values) == 0:
        raise ValueError("differences and clusters must be nonempty and equal length")
    if not np.all(np.isfinite(values)):
        raise ValueError("differences must be finite")
    if not 0 < alpha < 1 or replicates < 100:
        raise ValueError("invalid alpha or too few bootstrap replicates")
    if weights is None:
        row_weights = np.full(len(values), 1.0 / len(values))
    else:
        row_weights = np.asarray(weights, dtype=float)
        if (
            row_weights.shape != values.shape
            or np.any(row_weights < 0)
            or not np.any(row_weights > 0)
        ):
            raise ValueError("invalid weights")
        row_weights = row_weights / np.sum(row_weights)
    unique = list(dict.fromkeys(cluster_values.tolist()))
    if len(unique) < 2:
        raise ValueError("at least two clusters are required")
    indices = {cluster: np.flatnonzero(cluster_values == cluster) for cluster in unique}
    estimate = float(np.dot(row_weights, values))
    cluster_sizes = [len(indices[cluster]) for cluster in unique]
    leave_one_out = []
    for cluster in unique:
        retained = cluster_values != cluster
        retained_weights = row_weights[retained]
        if np.sum(retained_weights) <= 0:
            raise ValueError("weights must assign positive mass outside every cluster")
        retained_weights = retained_weights / np.sum(retained_weights)
        leave_one_out.append(float(np.dot(retained_weights, values[retained])))
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for b in range(replicates):
        selected = rng.integers(0, len(unique), size=len(unique))
        sampled_indices = np.concatenate([indices[unique[i]] for i in selected])
        sampled_values = values[sampled_indices]
        sampled_weights = row_weights[sampled_indices]
        sampled_weights = sampled_weights / np.sum(sampled_weights)
        draws[b] = np.dot(sampled_weights, sampled_values)
    return ClusterBootstrapResult(
        estimate=estimate,
        lower=float(np.quantile(draws, alpha / 2)),
        upper=float(np.quantile(draws, 1 - alpha / 2)),
        cluster_count=len(unique),
        minimum_cluster_size=min(cluster_sizes),
        maximum_cluster_size=max(cluster_sizes),
        cluster_size_ratio=max(cluster_sizes) / min(cluster_sizes),
        maximum_leave_one_cluster_out_shift=max(abs(value - estimate) for value in leave_one_out),
        replicates=replicates,
        seed=seed,
    )
