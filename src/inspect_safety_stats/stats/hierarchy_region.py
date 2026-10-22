from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import OptimizeResult, linprog
from scipy.sparse import csr_array


@dataclass(frozen=True)
class LinearIntervalConstraint:
    target_id: str
    coefficients: np.ndarray
    lower: float
    upper: float


@dataclass(frozen=True)
class HierarchyRegionProjection:
    feasible: bool
    intervals: dict[str, tuple[float, float]]
    constraint_count: int
    message: str | None = None


@dataclass(frozen=True)
class RatioHierarchyRegionProjection:
    feasible: bool
    intervals: dict[str, tuple[float | None, float | None]]
    constraint_count: int
    message: str | None = None


def _objective(result: OptimizeResult) -> float:
    """The optimal value of a linear program that reported success.

    Unlike an assert, this check survives python -O.
    """
    if result.fun is None:
        raise RuntimeError(f"the linear program reported success without a value: {result.message}")
    return float(result.fun)


def _region_solver(
    matrix: np.ndarray, limits: np.ndarray, bounds: list[tuple[float, float]]
) -> Callable[[np.ndarray], OptimizeResult]:
    """Solve linear programs over one fixed system of inequality constraints.

    linprog validates a dense constraint matrix and converts it to the same sparse matrix
    on every call, so it is handed that sparse matrix once instead. An objective that was
    already solved returns the earlier result: HiGHS is deterministic, and many projection
    targets share a coefficient vector. Neither changes the solver's input or any endpoint.
    """
    # linprog accepts sparse input for HiGHS; scipy-stubs only types the dense form.
    sparse: Any = csr_array(matrix)
    solved: dict[bytes, OptimizeResult] = {}

    def solve(objective: np.ndarray) -> OptimizeResult:
        key = objective.tobytes()
        if key not in solved:
            solved[key] = linprog(
                objective, A_ub=sparse, b_ub=limits, bounds=bounds, method="highs"
            )
        return solved[key]

    return solve


def _coerce_coefficients(values: ArrayLike, variable_count: int) -> np.ndarray:
    coefficients = np.asarray(values, dtype=float)
    if coefficients.ndim != 1 or len(coefficients) != variable_count:
        raise ValueError("every coefficient vector must match the variable count")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("coefficient vectors must be finite")
    if np.any(coefficients < 0) or not np.isclose(
        float(np.sum(coefficients)), 1.0, rtol=1e-10, atol=1e-12
    ):
        raise ValueError("coefficient vectors must be nonnegative and sum to one")
    return coefficients


def project_hierarchy_confidence_region(
    constraints: list[LinearIntervalConstraint],
    targets: dict[str, ArrayLike],
    *,
    variable_count: int,
) -> HierarchyRegionProjection:
    """Project a simultaneous linear confidence region onto requested functionals.

    The primitive parameters are bounded to ``[0, 1]``. Each supplied constraint has
    the form ``lower <= coefficients @ theta <= upper``. The returned interval for a
    target is its exact coordinate projection under these inequalities, up to the
    numerical tolerance of HiGHS linear programming.
    """

    if variable_count <= 0:
        raise ValueError("variable_count must be positive")
    if not constraints:
        raise ValueError("at least one interval constraint is required")
    if not targets:
        raise ValueError("at least one projection target is required")

    upper_rows: list[np.ndarray] = []
    upper_limits: list[float] = []
    for constraint in constraints:
        coefficients = _coerce_coefficients(constraint.coefficients, variable_count)
        if not np.isfinite(constraint.lower) or not np.isfinite(constraint.upper):
            raise ValueError("constraint endpoints must be finite")
        if not 0 <= constraint.lower <= constraint.upper <= 1:
            raise ValueError("constraint endpoints must satisfy 0 <= lower <= upper <= 1")
        upper_rows.extend([coefficients, -coefficients])
        upper_limits.extend([constraint.upper, -constraint.lower])

    matrix = np.vstack(upper_rows)
    limits = np.asarray(upper_limits, dtype=float)
    bounds = [(0.0, 1.0)] * variable_count
    solve = _region_solver(matrix, limits, bounds)
    feasibility = solve(np.zeros(variable_count))
    if not feasibility.success:
        return HierarchyRegionProjection(
            feasible=False,
            intervals={},
            constraint_count=len(constraints),
            message=feasibility.message,
        )

    intervals: dict[str, tuple[float, float]] = {}
    for target_id, raw_coefficients in targets.items():
        coefficients = _coerce_coefficients(raw_coefficients, variable_count)
        minimum = solve(coefficients)
        maximum = solve(-coefficients)
        if not minimum.success or not maximum.success:
            message = minimum.message if not minimum.success else maximum.message
            return HierarchyRegionProjection(
                feasible=False,
                intervals={},
                constraint_count=len(constraints),
                message=message,
            )
        lower = min(1.0, max(0.0, _objective(minimum)))
        upper = min(1.0, max(0.0, -_objective(maximum)))
        if lower > upper:
            if lower - upper > 1e-9:
                return HierarchyRegionProjection(
                    feasible=False,
                    intervals={},
                    constraint_count=len(constraints),
                    message="projected lower endpoint exceeds upper endpoint",
                )
            lower = upper = (lower + upper) / 2
        intervals[target_id] = (lower, upper)

    return HierarchyRegionProjection(
        feasible=True,
        intervals=intervals,
        constraint_count=len(constraints),
    )


def project_contrast_hierarchy_confidence_region(
    constraints: list[LinearIntervalConstraint],
    targets: dict[str, ArrayLike],
    *,
    variable_count: int,
) -> HierarchyRegionProjection:
    """Project a hierarchy region for prompt-level contrasts in ``[-1, 1]``."""

    if variable_count <= 0:
        raise ValueError("variable_count must be positive")
    if not constraints or not targets:
        raise ValueError("constraints and targets must be nonempty")
    upper_rows: list[np.ndarray] = []
    upper_limits: list[float] = []
    for constraint in constraints:
        coefficients = _coerce_coefficients(constraint.coefficients, variable_count)
        if not np.isfinite(constraint.lower) or not np.isfinite(constraint.upper):
            raise ValueError("constraint endpoints must be finite")
        if not -1 <= constraint.lower <= constraint.upper <= 1:
            raise ValueError("contrast endpoints must satisfy -1 <= lower <= upper <= 1")
        upper_rows.extend([coefficients, -coefficients])
        upper_limits.extend([constraint.upper, -constraint.lower])

    matrix = np.vstack(upper_rows)
    limits = np.asarray(upper_limits, dtype=float)
    bounds = [(-1.0, 1.0)] * variable_count
    solve = _region_solver(matrix, limits, bounds)
    feasibility = solve(np.zeros(variable_count))
    if not feasibility.success:
        return HierarchyRegionProjection(
            feasible=False,
            intervals={},
            constraint_count=len(constraints),
            message=feasibility.message,
        )

    intervals: dict[str, tuple[float, float]] = {}
    for target_id, raw_coefficients in targets.items():
        coefficients = _coerce_coefficients(raw_coefficients, variable_count)
        minimum = solve(coefficients)
        maximum = solve(-coefficients)
        if not minimum.success or not maximum.success:
            message = minimum.message if not minimum.success else maximum.message
            return HierarchyRegionProjection(
                feasible=False,
                intervals={},
                constraint_count=len(constraints),
                message=message,
            )
        lower = max(-1.0, min(1.0, _objective(minimum)))
        upper = max(-1.0, min(1.0, -_objective(maximum)))
        intervals[target_id] = (lower, upper)
    return HierarchyRegionProjection(
        feasible=True,
        intervals=intervals,
        constraint_count=len(constraints),
    )


def project_two_arm_difference_hierarchy_confidence_region(
    candidate_constraints: list[LinearIntervalConstraint],
    baseline_constraints: list[LinearIntervalConstraint],
    targets: dict[str, ArrayLike],
    *,
    variable_count: int,
) -> HierarchyRegionProjection:
    """Project weighted arm differences from one common two-arm linear region."""

    if variable_count <= 0:
        raise ValueError("variable_count must be positive")
    if not candidate_constraints or not baseline_constraints or not targets:
        raise ValueError("candidate constraints, baseline constraints, and targets are required")

    rows: list[np.ndarray] = []
    limits: list[float] = []
    for arm_index, constraints in enumerate((candidate_constraints, baseline_constraints)):
        offset = arm_index * variable_count
        for constraint in constraints:
            coefficients = _coerce_coefficients(constraint.coefficients, variable_count)
            if not 0 <= constraint.lower <= constraint.upper <= 1:
                raise ValueError("arm endpoints must satisfy 0 <= lower <= upper <= 1")
            arm_vector = np.zeros(2 * variable_count, dtype=float)
            arm_vector[offset : offset + variable_count] = coefficients
            rows.extend([arm_vector, -arm_vector])
            limits.extend([constraint.upper, -constraint.lower])

    matrix = np.vstack(rows)
    vector = np.asarray(limits, dtype=float)
    bounds = [(0.0, 1.0)] * (2 * variable_count)
    solve = _region_solver(matrix, vector, bounds)
    feasibility = solve(np.zeros(2 * variable_count))
    constraint_count = len(candidate_constraints) + len(baseline_constraints)
    if not feasibility.success:
        return HierarchyRegionProjection(
            feasible=False,
            intervals={},
            constraint_count=constraint_count,
            message=feasibility.message,
        )

    intervals: dict[str, tuple[float, float]] = {}
    for target_id, raw_coefficients in targets.items():
        coefficients = _coerce_coefficients(raw_coefficients, variable_count)
        objective = np.concatenate([coefficients, -coefficients])
        minimum = solve(objective)
        maximum = solve(-objective)
        if not minimum.success or not maximum.success:
            message = minimum.message if not minimum.success else maximum.message
            return HierarchyRegionProjection(
                feasible=False,
                intervals={},
                constraint_count=constraint_count,
                message=message,
            )
        lower = max(-1.0, min(1.0, _objective(minimum)))
        upper = max(-1.0, min(1.0, -_objective(maximum)))
        intervals[target_id] = (lower, upper)

    return HierarchyRegionProjection(
        feasible=True,
        intervals=intervals,
        constraint_count=constraint_count,
    )


def project_ratio_hierarchy_confidence_region(
    candidate_constraints: list[LinearIntervalConstraint],
    baseline_constraints: list[LinearIntervalConstraint],
    targets: dict[str, ArrayLike],
    *,
    variable_count: int,
) -> RatioHierarchyRegionProjection:
    """Project ratios of two weighted means over a common two-arm linear region.

    Linear-fractional objectives are solved exactly (up to HiGHS tolerance) with
    the Charnes-Cooper transformation. A missing upper endpoint means the region
    permits the baseline denominator to approach zero while candidate risk remains
    positive.
    """

    if variable_count <= 0:
        raise ValueError("variable_count must be positive")
    if not candidate_constraints or not baseline_constraints or not targets:
        raise ValueError("candidate constraints, baseline constraints, and targets are required")

    rows: list[np.ndarray] = []
    limits: list[float] = []
    for arm_index, constraints in enumerate((candidate_constraints, baseline_constraints)):
        offset = arm_index * variable_count
        for constraint in constraints:
            coefficients = _coerce_coefficients(constraint.coefficients, variable_count)
            if not 0 <= constraint.lower <= constraint.upper <= 1:
                raise ValueError("arm endpoints must satisfy 0 <= lower <= upper <= 1")
            arm_vector = np.zeros(2 * variable_count, dtype=float)
            arm_vector[offset : offset + variable_count] = coefficients
            rows.extend([arm_vector, -arm_vector])
            limits.extend([constraint.upper, -constraint.lower])

    matrix = np.vstack(rows)
    vector = np.asarray(limits, dtype=float)
    base_bounds = [(0.0, 1.0)] * (2 * variable_count)
    feasibility = _region_solver(matrix, vector, base_bounds)(np.zeros(2 * variable_count))
    constraint_count = len(candidate_constraints) + len(baseline_constraints)
    if not feasibility.success:
        return RatioHierarchyRegionProjection(
            feasible=False,
            intervals={},
            constraint_count=constraint_count,
            message=feasibility.message,
        )

    # z=t*x, t=1/(a' baseline). Original A*x<=b becomes
    # A*z-b*t<=0, with 0<=z_i<=t and a'z_baseline=1.
    transformed_rows = np.column_stack([matrix, -vector])
    upper_bound_rows = []
    for index in range(2 * variable_count):
        row = np.zeros(2 * variable_count + 1, dtype=float)
        row[index] = 1.0
        row[-1] = -1.0
        upper_bound_rows.append(row)
    # Sparse for HiGHS, as in _region_solver; scipy-stubs only types the dense form.
    transformed_matrix: Any = csr_array(np.vstack([transformed_rows, *upper_bound_rows]))
    transformed_limits = np.zeros(transformed_matrix.shape[0], dtype=float)
    solved: dict[bytes, tuple[float | None, float | None]] = {}

    intervals: dict[str, tuple[float | None, float | None]] = {}
    for target_id, raw_coefficients in targets.items():
        coefficients = _coerce_coefficients(raw_coefficients, variable_count)
        key = coefficients.tobytes()
        if key in solved:
            intervals[target_id] = solved[key]
            continue
        equality = np.zeros((1, 2 * variable_count + 1), dtype=float)
        equality[0, variable_count : 2 * variable_count] = coefficients
        objective = np.zeros(2 * variable_count + 1, dtype=float)
        objective[:variable_count] = coefficients
        transformed_bounds = [(0.0, None)] * (2 * variable_count + 1)
        minimum = linprog(
            objective,
            A_ub=transformed_matrix,
            b_ub=transformed_limits,
            A_eq=equality,
            b_eq=np.ones(1),
            bounds=transformed_bounds,
            method="highs",
        )
        maximum = linprog(
            -objective,
            A_ub=transformed_matrix,
            b_ub=transformed_limits,
            A_eq=equality,
            b_eq=np.ones(1),
            bounds=transformed_bounds,
            method="highs",
        )
        lower = (
            max(0.0, float(minimum.fun)) if minimum.success and minimum.fun is not None else None
        )
        upper = (
            max(0.0, float(-maximum.fun)) if maximum.success and maximum.fun is not None else None
        )
        intervals[target_id] = solved[key] = (lower, upper)

    return RatioHierarchyRegionProjection(
        feasible=True,
        intervals=intervals,
        constraint_count=constraint_count,
    )
