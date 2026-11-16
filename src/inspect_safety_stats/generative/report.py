"""Result artifacts for a generative evaluation.

The generative report is kept visibly distinct from the fixed-design report, as section 2
of the plan requires. Its estimand is a probability under a declared generative design, not
over a materialized prompt set, and every artifact names the taxonomy and kernel
fingerprints it was produced under so two runs cannot be pooled by accident.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from inspect_safety_stats.generative.comparisons import ComparisonResult
from inspect_safety_stats.generative.execution import AcceptanceRow
from inspect_safety_stats.generative.inference import (
    NodeEstimate,
    ReweightedEstimate,
    TreeEstimate,
)
from inspect_safety_stats.generative.judge import CorrectedEstimate
from inspect_safety_stats.generative.taxonomy import CompiledTaxonomy
from inspect_safety_stats.generative.visualization import write_generative_report
from inspect_safety_stats.report.artifacts import (
    CHECKSUMS,
    write_checksums,
    write_frame,
    write_json,
)
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
    CHECKSUMS,
)


def plan_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class GenerativeReport:
    """Everything :func:`analyze_run` estimated, and the writer for its artifacts.

    ``evaluation`` names the plan hash and the taxonomy and kernel fingerprints; ``design``
    records the resolved sampling design. :meth:`write` produces the files in ``ARTIFACTS``.
    """

    evaluation: dict[str, Any]
    design: dict[str, Any]
    tree: TreeEstimate
    taxonomy: CompiledTaxonomy
    slices: list[NodeEstimate] = field(default_factory=list)
    comparisons: list[ComparisonResult] = field(default_factory=list)
    corrected: list[CorrectedEstimate] = field(default_factory=list)
    acceptance: list[AcceptanceRow] = field(default_factory=list)
    reweightings: list[ReweightedEstimate] = field(default_factory=list)
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

    def to_dict(self) -> dict[str, Any]:
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
        # Unlike the fixed-design report.json, this one escapes non-ASCII text and allows NaN.
        write_json(directory / "report.json", self.to_dict(), ensure_ascii=True, allow_nan=True)
        write_frame(directory / "nodes.csv", self._nodes_frame())
        write_frame(directory / "taxonomy.csv", self.taxonomy.to_frame())
        write_frame(
            directory / "diagnostics.csv",
            pd.DataFrame(
                [item.model_dump(mode="json") for item in self.diagnostics],
                columns=["code", "severity", "message", "scope", "details"],
            ),
        )
        write_frame(directory / "comparisons.csv", self._comparisons_frame())
        write_frame(
            directory / "acceptance.csv",
            pd.DataFrame([asdict(item) for item in self.acceptance]),
        )
        write_generative_report(self, directory / "report.html")
        # ORDER MATTERS: last, once every file it covers is written.
        write_checksums(directory, [name for name in ARTIFACTS if name != CHECKSUMS])
        return directory
