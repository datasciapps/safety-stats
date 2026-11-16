"""The numeric core: interval methods and risk measures over arrays and plain values.

The public names are imported on first use, so that importing this package does not load
pandas and SciPy.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .binomial import (
        binomial_bounds,
        binomial_lower_bound,
        binomial_upper_bound,
        missingness_robust_binomial_bounds,
    )
    from .comparisons import (
        DirectComparisonBounds,
        RatioComparisonBounds,
        direct_ratio_bounds,
        direct_risk_difference_bounds,
    )
    from .design_based import (
        DesignBasedBounds,
        InclusionDesign,
        horvitz_thompson_mean_bounds,
    )
    from .direct_mean import (
        DirectMeanBounds,
        weighted_bounded_empirical_bernstein_mean_bounds,
        weighted_bounded_hoeffding_mean_bounds,
        weighted_bounded_kl_chernoff_mean_bounds,
        weighted_empirical_bernstein_mean_bounds,
        weighted_hoeffding_mean_bounds,
        weighted_kl_chernoff_mean_bounds,
    )
    from .hierarchy_region import (
        HierarchyRegionProjection,
        LinearIntervalConstraint,
        RatioHierarchyRegionProjection,
        project_contrast_hierarchy_confidence_region,
        project_hierarchy_confidence_region,
        project_ratio_hierarchy_confidence_region,
        project_two_arm_difference_hierarchy_confidence_region,
    )
    from .independence import BoundedBlocks
    from .judges import (
        AgreementSummary,
        IdentifiedRiskSet,
        JudgeAccuracyBox,
        compose_verdicts,
        composite_accuracy_box,
        effective_votes,
        identified_risk_set,
        pairwise_agreement,
    )
    from .misclassification import (
        CorrectedRisk,
        DeltaCorrectedRisk,
        JudgeCalibration,
        correct_judge_risk,
        delta_method_corrected_risk,
    )
    from .multiplicity import bonferroni_allocate
    from .paired import (
        exact_mcnemar,
        exact_mcnemar_partial,
        newcombe_paired_interval,
        newcombe_paired_interval_partial,
    )
    from .planning import zero_failure_sample_size
    from .risk import (
        entropic_risk,
        maximum_risk,
        threshold_exceedance,
        threshold_excess,
        top_k_mean,
        weighted_cvar,
        weighted_mean,
        worst_of_k_estimate,
        worst_of_k_risk,
    )
    from .tree_calibration import (
        FrontierTargetCalibration,
        IndependentFrontier,
        calibrate_independent_frontiers,
        sidak_local_alpha,
    )

__all__ = [
    "binomial_bounds",
    "binomial_lower_bound",
    "binomial_upper_bound",
    "BoundedBlocks",
    "missingness_robust_binomial_bounds",
    "bonferroni_allocate",
    "correct_judge_risk",
    "CorrectedRisk",
    "JudgeCalibration",
    "DirectComparisonBounds",
    "DesignBasedBounds",
    "DirectMeanBounds",
    "horvitz_thompson_mean_bounds",
    "InclusionDesign",
    "direct_ratio_bounds",
    "direct_risk_difference_bounds",
    "entropic_risk",
    "exact_mcnemar",
    "exact_mcnemar_partial",
    "FrontierTargetCalibration",
    "newcombe_paired_interval",
    "newcombe_paired_interval_partial",
    "HierarchyRegionProjection",
    "IndependentFrontier",
    "LinearIntervalConstraint",
    "maximum_risk",
    "project_contrast_hierarchy_confidence_region",
    "project_hierarchy_confidence_region",
    "project_ratio_hierarchy_confidence_region",
    "project_two_arm_difference_hierarchy_confidence_region",
    "RatioHierarchyRegionProjection",
    "RatioComparisonBounds",
    "threshold_exceedance",
    "threshold_excess",
    "top_k_mean",
    "weighted_cvar",
    "weighted_bounded_empirical_bernstein_mean_bounds",
    "weighted_bounded_hoeffding_mean_bounds",
    "weighted_bounded_kl_chernoff_mean_bounds",
    "weighted_empirical_bernstein_mean_bounds",
    "weighted_hoeffding_mean_bounds",
    "weighted_kl_chernoff_mean_bounds",
    "weighted_mean",
    "worst_of_k_estimate",
    "worst_of_k_risk",
    "zero_failure_sample_size",
    "AgreementSummary",
    "IdentifiedRiskSet",
    "JudgeAccuracyBox",
    "compose_verdicts",
    "composite_accuracy_box",
    "effective_votes",
    "identified_risk_set",
    "pairwise_agreement",
    "DeltaCorrectedRisk",
    "delta_method_corrected_risk",
    "calibrate_independent_frontiers",
    "sidak_local_alpha",
]

_EXPORTS = {
    "AgreementSummary": "inspect_safety_stats.stats.judges",
    "BoundedBlocks": "inspect_safety_stats.stats.independence",
    "CorrectedRisk": "inspect_safety_stats.stats.misclassification",
    "DeltaCorrectedRisk": "inspect_safety_stats.stats.misclassification",
    "DesignBasedBounds": "inspect_safety_stats.stats.design_based",
    "DirectComparisonBounds": "inspect_safety_stats.stats.comparisons",
    "DirectMeanBounds": "inspect_safety_stats.stats.direct_mean",
    "FrontierTargetCalibration": "inspect_safety_stats.stats.tree_calibration",
    "HierarchyRegionProjection": "inspect_safety_stats.stats.hierarchy_region",
    "IdentifiedRiskSet": "inspect_safety_stats.stats.judges",
    "InclusionDesign": "inspect_safety_stats.stats.design_based",
    "IndependentFrontier": "inspect_safety_stats.stats.tree_calibration",
    "JudgeAccuracyBox": "inspect_safety_stats.stats.judges",
    "JudgeCalibration": "inspect_safety_stats.stats.misclassification",
    "LinearIntervalConstraint": "inspect_safety_stats.stats.hierarchy_region",
    "RatioComparisonBounds": "inspect_safety_stats.stats.comparisons",
    "RatioHierarchyRegionProjection": "inspect_safety_stats.stats.hierarchy_region",
    "binomial_bounds": "inspect_safety_stats.stats.binomial",
    "binomial_lower_bound": "inspect_safety_stats.stats.binomial",
    "binomial_upper_bound": "inspect_safety_stats.stats.binomial",
    "bonferroni_allocate": "inspect_safety_stats.stats.multiplicity",
    "calibrate_independent_frontiers": "inspect_safety_stats.stats.tree_calibration",
    "compose_verdicts": "inspect_safety_stats.stats.judges",
    "composite_accuracy_box": "inspect_safety_stats.stats.judges",
    "correct_judge_risk": "inspect_safety_stats.stats.misclassification",
    "delta_method_corrected_risk": "inspect_safety_stats.stats.misclassification",
    "direct_ratio_bounds": "inspect_safety_stats.stats.comparisons",
    "direct_risk_difference_bounds": "inspect_safety_stats.stats.comparisons",
    "effective_votes": "inspect_safety_stats.stats.judges",
    "entropic_risk": "inspect_safety_stats.stats.risk",
    "exact_mcnemar": "inspect_safety_stats.stats.paired",
    "exact_mcnemar_partial": "inspect_safety_stats.stats.paired",
    "horvitz_thompson_mean_bounds": "inspect_safety_stats.stats.design_based",
    "identified_risk_set": "inspect_safety_stats.stats.judges",
    "maximum_risk": "inspect_safety_stats.stats.risk",
    "missingness_robust_binomial_bounds": "inspect_safety_stats.stats.binomial",
    "newcombe_paired_interval": "inspect_safety_stats.stats.paired",
    "newcombe_paired_interval_partial": "inspect_safety_stats.stats.paired",
    "pairwise_agreement": "inspect_safety_stats.stats.judges",
    "project_contrast_hierarchy_confidence_region": "inspect_safety_stats.stats.hierarchy_region",
    "project_hierarchy_confidence_region": "inspect_safety_stats.stats.hierarchy_region",
    "project_ratio_hierarchy_confidence_region": "inspect_safety_stats.stats.hierarchy_region",
    "project_two_arm_difference_hierarchy_confidence_region": (
        "inspect_safety_stats.stats.hierarchy_region"
    ),
    "sidak_local_alpha": "inspect_safety_stats.stats.tree_calibration",
    "threshold_exceedance": "inspect_safety_stats.stats.risk",
    "threshold_excess": "inspect_safety_stats.stats.risk",
    "top_k_mean": "inspect_safety_stats.stats.risk",
    "weighted_bounded_empirical_bernstein_mean_bounds": "inspect_safety_stats.stats.direct_mean",
    "weighted_bounded_hoeffding_mean_bounds": "inspect_safety_stats.stats.direct_mean",
    "weighted_bounded_kl_chernoff_mean_bounds": "inspect_safety_stats.stats.direct_mean",
    "weighted_cvar": "inspect_safety_stats.stats.risk",
    "weighted_empirical_bernstein_mean_bounds": "inspect_safety_stats.stats.direct_mean",
    "weighted_hoeffding_mean_bounds": "inspect_safety_stats.stats.direct_mean",
    "weighted_kl_chernoff_mean_bounds": "inspect_safety_stats.stats.direct_mean",
    "weighted_mean": "inspect_safety_stats.stats.risk",
    "worst_of_k_estimate": "inspect_safety_stats.stats.risk",
    "worst_of_k_risk": "inspect_safety_stats.stats.risk",
    "zero_failure_sample_size": "inspect_safety_stats.stats.planning",
}

# Submodules stay reachable as attributes, as they were when this module imported them.
_SUBMODULES = {
    "binomial",
    "clusters",
    "comparisons",
    "design_based",
    "direct_mean",
    "hierarchy_region",
    "independence",
    "judges",
    "misclassification",
    "multiplicity",
    "paired",
    "planning",
    "risk",
    "tree_calibration",
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        value = getattr(importlib.import_module(_EXPORTS[name]), name)
    elif name in _SUBMODULES:
        value = importlib.import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
