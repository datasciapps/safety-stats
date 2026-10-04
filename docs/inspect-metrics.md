# Bounds as Inspect metrics

The package's main path analyzes finished logs. For a quick view while you iterate, three
Inspect metrics put the failure rate and its interval in the eval log itself, next to
`accuracy` and the rest, so they show in `inspect view` and in `eval()` results.

```python
from inspect_ai import Epochs, Task, task
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import generate

from inspect_safety_stats.metrics import failure_rate, failure_rate_lower, failure_rate_upper

FAIL = ["I"]


@task
def harmful_requests() -> Task:
    return Task(
        dataset=...,
        solver=generate(),
        scorer=model_graded_qa(),
        metrics=[
            failure_rate(failure_values=FAIL),
            failure_rate_lower(failure_values=FAIL),
            failure_rate_upper(failure_values=FAIL),
        ],
        epochs=Epochs(8),
    )
```

The log then reports, for example:

```text
failure_rate        0.150
failure_rate_lower  0.076
failure_rate_upper  0.252
```

## The score mapping

As everywhere in the package, the mapping is explicit; the metrics refuse to be created
without one.

- `failure_values=[...]`: the score values that are failures. Every other value is safe unless
  `safe_values=[...]` is also given, in which case anything in neither list is unclassified.
- `harm_scale=[minimum, maximum]`: a graded score mapped linearly onto harm in [0, 1], with
  `lower_is_harmful=True` when the minimum is the worst grade. Non-numeric or off-scale values
  are unclassified.
- `key="..."`: for a scorer whose value is a dict, the entry to read.

Unclassified values widen the interval: they count as harmless for `failure_rate` and
`failure_rate_lower` and as failures for `failure_rate_upper`.

## What the interval means

The metrics bound the same quantity as `analyze`: the mean, over the samples you ran with equal
weight, of each sample's per-rollout failure probability, assuming the rollouts are
independent. They see every epoch's score, not one reduced score per sample (they are declared
with Inspect's `scores="unreduced"`), so an epoch reducer such as `"max"` or `"mode"` does not
change them.

The interval is the weighted KL/Chernoff interval: valid at any sample size, including zero
observed failures, and for graded harm. `alpha` (default 0.05) is two-sided, so
`failure_rate_lower` and `failure_rate_upper` with the same `alpha` form one 95% interval.
For one model and one task this is exactly the primary interval `analyze` reports for the same
log with a starter plan.

## What they leave out

A metric sees one scorer on one task for one model. It cannot:

- share an error budget across models, tasks, groups or other metrics: running several models
  gives several separate 95% intervals, not simultaneous ones;
- compare models, break results down by category, or apply judge-error correction;
- report the prompt-level sensitivity that drops the independence assumption.

For any of those, or for a result you will act on, run `inspect-safety-stats init` and
`analyze` on the logs.
