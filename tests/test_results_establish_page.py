"""docs/what-results-establish.md names what it summarises, so the two cannot drift apart.

Every report of a fixed or sampled design links to the page, so a reader takes its words for
what the package does. It once said that degraded evidence "keeps conservative bounds", which
holds only for the codes that complete unknown outcomes, and that "the upper bound counts every
unknown as a failure", which holds only under the default score policies. These checks tie the
page's lists to the code: a new code that degrades integrity, or a new policy for missing scores
or sample errors, has to be said on the page as well.
"""

from __future__ import annotations

from pathlib import Path
from typing import get_args

from safety_stats.config.models import ScoreSpec
from safety_stats.registry import CODES

PAGE = Path(__file__).resolve().parents[1] / "docs" / "what-results-establish.md"


def test_the_page_names_every_code_that_degrades_integrity() -> None:
    page = PAGE.read_text(encoding="utf-8")
    degrading = sorted(code for code, entry in CODES.items() if entry.effect == "degraded")
    assert degrading
    assert [code for code in degrading if f"`{code}`" not in page] == []


def test_the_page_says_what_every_missing_score_and_error_policy_does() -> None:
    page = PAGE.read_text(encoding="utf-8")
    missing = ScoreSpec.model_fields["missing"]
    errors = ScoreSpec.model_fields["errors"]
    # The adverse completion is stated for the defaults, by name.
    assert f"`inputs.score.missing: {missing.default}`" in page
    assert f"`inputs.score.errors: {errors.default}`" in page
    # Every other policy has an item of its own. conservative_bounds completes unknown
    # outcomes as the default end_to_end does.
    policies = set(get_args(missing.annotation)) | set(get_args(errors.annotation))
    others = sorted(policies - {missing.default, errors.default, "conservative_bounds"})
    assert others
    assert [policy for policy in others if f"\n- `{policy}` " not in page] == []
