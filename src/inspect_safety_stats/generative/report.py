"""Result artifacts for a generative evaluation.

The generative report is kept visibly distinct from the fixed-design report, as section 2
of the plan requires. Its estimand is a probability under a declared generative design, not
over a materialized prompt set, and every artifact names the taxonomy and kernel
fingerprints it was produced under so two runs cannot be pooled by accident.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from inspect_safety_stats.generative.comparisons import ComparisonResult
from inspect_safety_stats.generative.contrasts import SliceFamilyResult
from inspect_safety_stats.generative.execution import AcceptanceRow
from inspect_safety_stats.generative.inference import (
    NodeEstimate,
    ReweightedEstimate,
    TreeEstimate,
)
from inspect_safety_stats.generative.judge import CorrectedEstimate
from inspect_safety_stats.generative.taxonomy import CompiledTaxonomy
from inspect_safety_stats.generative.visualization import write_generative_report
from inspect_safety_stats.schema.reports import Diagnostic
from inspect_safety_stats.version import __version__

ARTIFACTS = (
    "report.json",
    "report.html",
    "nodes.csv",
    "taxonomy.csv",
    "diagnostics.csv",
    "comparisons.csv",
    "acceptance.csv",
)
# Written only when the plan declares inference.slice_family.
FAMILY_ARTIFACT = "family.csv"
FAMILY_COLUMNS = (
    "kind", "member_id", "level", "value", "candidate_value", "baseline_value",
    "shared_seeds", "draws", "estimate", "lower", "upper", "method", "frontier_id",
    "frontier_size", "endpoint_alpha", "candidate_draws", "candidate_estimate",
    "candidate_lower", "candidate_upper", "baseline_draws", "baseline_estimate",
    "baseline_lower", "baseline_upper",
)


def plan_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class GenerativeReport:
    evaluation: dict[str, Any]
    design: dict[str, Any]
    tree: TreeEstimate
    taxonomy: CompiledTaxonomy
    slices: list[NodeEstimate] = field(default_factory=list)
    comparisons: list[ComparisonResult] = field(default_factory=list)
    corrected: list[CorrectedEstimate] = field(default_factory=list)
    acceptance: list[AcceptanceRow] = field(default_factory=list)
    reweightings: list[ReweightedEstimate] = field(default_factory=list)
    slice_family: SliceFamilyResult | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(item.severity == "ERROR" for item in self.diagnostics)

    def _nodes_frame(self) -> pd.DataFrame:
        rows = [asdict(node) for node in (*self.tree.nodes, *self.slices)]
        for row, node in zip(rows, (*self.tree.nodes, *self.slices), strict=True):
            row["kind"] = "slice" if node.node_id.startswith("slice:") else "node"
        return pd.DataFrame(rows)

    def _comparisons_frame(self) -> pd.DataFrame:
        rows = []
        for item in self.comparisons:
            row = {
                key: value
                for key, value in asdict(item).items()
                if key not in {"candidate", "baseline"}
            }
            row["candidate_lower"] = item.candidate.lower
            row["candidate_upper"] = item.candidate.upper
            row["candidate_estimate"] = item.candidate.estimate
            row["baseline_lower"] = item.baseline.lower
            row["baseline_upper"] = item.baseline.upper
            row["baseline_estimate"] = item.baseline.estimate
            rows.append(row)
        return pd.DataFrame(rows)

    def _family_frame(self) -> pd.DataFrame:
        family = self.slice_family
        rows: list[dict[str, Any]] = []
        if family is not None:
            for item in family.slices:
                rows.append(
                    {
                        "kind": "slice",
                        "member_id": item.member_id,
                        "level": item.level,
                        "value": item.value,
                        "shared_seeds": item.arm.seeds,
                        "draws": item.arm.draws,
                        "estimate": item.arm.estimate,
                        "lower": item.arm.lower,
                        "upper": item.arm.upper,
                        "method": item.arm.method,
                        "frontier_id": item.frontier_id,
                        "frontier_size": item.frontier_size,
                        "endpoint_alpha": item.endpoint_alpha,
                    }
                )
            for contrast in family.contrasts:
                row: dict[str, Any] = {
                    "kind": "contrast",
                    "member_id": contrast.id,
                    "level": contrast.level,
                    "candidate_value": contrast.candidate_value,
                    "baseline_value": contrast.baseline_value,
                    "shared_seeds": contrast.shared_seeds,
                    "draws": contrast.candidate.draws + contrast.baseline.draws,
                    "estimate": contrast.estimate,
                    "lower": contrast.lower,
                    "upper": contrast.upper,
                    "method": f"{contrast.candidate.method}|{contrast.baseline.method}",
                    "frontier_id": contrast.frontier_id,
                    "frontier_size": contrast.frontier_size,
                    "endpoint_alpha": contrast.endpoint_alpha,
                }
                sides = (("candidate", contrast.candidate), ("baseline", contrast.baseline))
                for side, arm in sides:
                    row[f"{side}_draws"] = arm.draws
                    row[f"{side}_estimate"] = arm.estimate
                    row[f"{side}_lower"] = arm.lower
                    row[f"{side}_upper"] = arm.upper
                rows.append(row)
        return pd.DataFrame(rows, columns=list(FAMILY_COLUMNS))

    def to_dict(self) -> dict[str, Any]:
        document = self._base_dict()
        if self.slice_family is not None:
            # Only when declared, so a plan without a family keeps its report unchanged.
            document["slice_family"] = {
                **asdict(self.slice_family),
                "simultaneous": True,
                "judge_scale": "reported",
            }
        return document

    def _base_dict(self) -> dict[str, Any]:
        return {
            "schema": "generative-report/1.0",
            "package_version": __version__,
            "evaluation": self.evaluation,
            "design": self.design,
            "estimand": (
                "failure probability under the declared generative design; not a "
                "probability over all possible prompts, attacks, or a deployment "
                "distribution"
            ),
            "root": asdict(self.tree.root),
            "nodes": [asdict(node) for node in self.tree.nodes],
            "slices": [asdict(node) for node in self.slices],
            "comparisons": [
                {
                    key: (asdict(value) if isinstance(value, NodeEstimate) else value)
                    for key, value in asdict(item).items()
                }
                for item in self.comparisons
            ],
            "corrected": [asdict(item) for item in self.corrected],
            "acceptance": [asdict(item) for item in self.acceptance],
            "design_sensitivity": [asdict(item) for item in self.reweightings],
            "unsampled_nodes": list(self.tree.unsampled_nodes),
            "coverage": {
                "alpha": self.tree.alpha,
                "mode": self.tree.coverage_mode,
                "simultaneous_tree": self.tree.simultaneous,
                "slices_are_pointwise": True,
            },
            "diagnostics": [item.model_dump(mode="json") for item in self.diagnostics],
        }

    def write(self, output: str | Path, *, overwrite: bool = False) -> Path:
        directory = Path(output)
        if directory.exists() and any(directory.iterdir()) and not overwrite:
            raise FileExistsError(
                f"{directory} is not empty; pass --overwrite to replace its contents"
            )
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "report.json").write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        self._nodes_frame().to_csv(directory / "nodes.csv", index=False, lineterminator="\n")
        self.taxonomy.to_frame().to_csv(
            directory / "taxonomy.csv", index=False, lineterminator="\n"
        )
        pd.DataFrame(
            [item.model_dump(mode="json") for item in self.diagnostics],
            columns=["code", "severity", "message", "scope", "details"],
        ).to_csv(directory / "diagnostics.csv", index=False, lineterminator="\n")
        self._comparisons_frame().to_csv(
            directory / "comparisons.csv", index=False, lineterminator="\n"
        )
        pd.DataFrame([asdict(item) for item in self.acceptance]).to_csv(
            directory / "acceptance.csv", index=False, lineterminator="\n"
        )
        if self.slice_family is not None:
            self._family_frame().to_csv(
                directory / FAMILY_ARTIFACT, index=False, lineterminator="\n"
            )
        write_generative_report(self, directory / "report.html")
        return directory
