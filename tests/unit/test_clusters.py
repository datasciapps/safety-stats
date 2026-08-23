import pytest

from inspect_safety_stats.stats.clusters import paired_cluster_bootstrap


def test_cluster_bootstrap_is_seeded_and_preserves_clusters() -> None:
    differences = [1.0, 1.0, -1.0, -1.0, 0.5, 0.5]
    clusters = ["a", "a", "b", "b", "c", "c"]
    first = paired_cluster_bootstrap(differences, clusters, replicates=500, seed=42)
    second = paired_cluster_bootstrap(differences, clusters, replicates=500, seed=42)
    assert first == second
    assert first.cluster_count == 3
    assert first.minimum_cluster_size == 2
    assert first.maximum_cluster_size == 2
    assert first.cluster_size_ratio == 1
    assert first.maximum_leave_one_cluster_out_shift > 0
    assert first.estimate == pytest.approx(sum(differences) / len(differences))
    assert first.lower <= first.estimate <= first.upper


def test_cluster_bootstrap_requires_multiple_clusters() -> None:
    with pytest.raises(ValueError, match="at least two clusters"):
        paired_cluster_bootstrap([0.0, 1.0], ["only", "only"])
