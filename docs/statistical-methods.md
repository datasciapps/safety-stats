# Statistical Methods for Safety-Oriented AI Evaluation

## Mathematical foundations, decision guarantees, and precedents in AI safety research

**Status:** design specification; fixed-design core release with deferred stretch goals  
**Target package:** `inspect-safety-stats` (working name)  
**Target framework:** [Inspect AI](https://inspect.aisi.org.uk/)  
**Last updated:** 18 August 2026

**Version 1 engineering design:** [v1-design.md](archive/v1-design.md) (archived)

---

## 1. Purpose and scope

This document specifies the mathematical foundations of a statistical extension for safety-oriented AI evaluations. The library is intended to turn binary safety outcomes—such as whether a model disclosed restricted information, executed a harmful action, escaped a sandbox, or evaded a monitor—into defensible statements about risk.

The initial release is deliberately restricted to a **fixed, hand-curated evaluation design**. Prompts, questions, tasks, groups, weights, rollout counts, thresholds, and the analysis time are declared before outcomes are examined. The core release consists of six capabilities:

1. Exact one-sided bounds for a failure probability.
2. Sample-size planning when zero failures are expected or required.
3. Risk-sensitive aggregation over the fixed design, including worst-case, CVaR, entropic risk, power means, and threshold-exceedance measures.
4. Paired and cluster-aware model comparisons that respect the declared design.
5. Multiplicity correction across safety claims where the decision semantics require it.
6. A pass/fail/inconclusive decision and reporting system.

Bernoulli confidence sequences, sequential monitoring, and adaptive rollout allocation are retained in this document as a **stretch goal**, not as requirements for the initial release. They become materially useful when evaluators want to choose sample sizes or rollout allocation using observed outcomes. The core workflow instead uses predeclared fixed horizons, for which exact methods are simpler, more auditable, and usually tighter.

The release priorities are therefore:

| Priority | Included |
|---|---|
| Core | Fixed design schema and validation; exact one-sided bounds; zero-failure planning; declared mean and risk-sensitive fixed-design aggregators; paired and cluster-aware comparisons; multiplicity; pass/fail/inconclusive reports |
| Later extension | Anytime-valid confidence sequences; adaptive rollout allocation; live stopping; sequential multiplicity integration |
| Out of scope for now | Generalization to an unseen prompt population; LLM-generated prompt rewriting; semantic-cluster inference; adversarial prompt-distribution estimation; deployment-prevalence claims |

The library is not intended to prove that an AI system is safe in every deployment context. Statistical inference can quantify sampling uncertainty only relative to a defined population, outcome, data-generating process, and evaluation procedure. It cannot repair a benchmark that omits an important threat, a scorer that misses harmful behavior, an elicitation procedure that fails to expose a capability, or a deployment distribution that differs materially from the evaluation distribution.

The intended claim is therefore narrower:

> Under a predeclared evaluation design and its stated assumptions, the observed evidence does or does not rule out a specified rate of a defined failure event.

This distinction is especially important in safety research. A mathematically correct interval around the wrong estimand can create more confidence without creating more safety.

---

## 2. Statistical object: what probability are we estimating?

Let

\[
X_i =
\begin{cases}
1, & \text{if evaluation unit } i \text{ exhibits the defined safety failure},\\
0, & \text{otherwise.}
\end{cases}
\]

The most basic estimand is a failure probability

\[
p = \Pr(X_i=1).
\]

This notation is deceptively simple. The probability is meaningless until the evaluation specifies what is random. At least four interpretations occur in AI evaluation.

### 2.1 Model-sampling probability on a fixed prompt

The prompt and system configuration are fixed, while decoding randomness, tool responses, or environment randomness vary. Here, \(p\) is the probability of failure on that prompt under a precisely specified generation policy.

### 2.2 A possible average over a fixed benchmark

If the benchmark contains \(N\) fixed items, one possible estimand is the equal-item average

\[
p_B = \frac{1}{N}\sum_{j=1}^N p_j,
\]

where \(p_j\) is the model-sampling failure probability on item \(j\). Repeated trials on an item help estimate \(p_j\), but do not create new benchmark items.

This quantity has a precise but limited interpretation: it is the expected failure probability if an item is selected uniformly from these exact \(N\) items and a rollout is then generated. It is not automatically the right safety estimand. It gives every item equal policy weight, can be changed by duplicating or subdividing items, and can conceal an unacceptable result on one critical item behind successes on many others.

The initial library therefore has **no universal default aggregate estimand**. The evaluation creator must choose and justify the aggregation semantics before outcomes are examined. Depending on the safety case, the appropriate primary claim may instead be:

- a hierarchy-weighted average that gives each question or task equal weight;
- a worst-prompt or worst-question bound;
- separate thresholds for every mandatory group;
- an all-mandatory-items conjunction;
- a zero-tolerance rule for specified critical events.

Arithmetic averaging should be treated as a declared policy choice and ordinarily as a secondary summary, not as the default definition of safety. The benchmark remains a design set, not a random sample from an implicit population of prompts. Accordingly, every report must state:

> Results apply to the declared fixed evaluation set and specified stochastic rollout process. They do not estimate failure prevalence over an external population of prompts.

### 2.3 Declared hierarchy within the fixed design

The creator may declare a hierarchy such as

\[
\text{task}
\rightarrow
\text{question}
\rightarrow
\text{prompt}
\rightarrow
\text{rollout}.
\]

The task, question, prompt, and group identities are fixed design elements. Only the rollout is treated as stochastic unless the analysis plan explicitly declares another randomization mechanism. The library should validate creator-supplied identifiers and relationships, but must not infer semantic groupings automatically.

For prompt \(j\), define

\[
p_j=\Pr(\text{failure on a rollout}\mid\text{fixed prompt }j).
\]

For a declared group \(g\), define a fixed weighted estimand

\[
\mu_g=\sum_{j\in g}w_{gj}p_j,
\qquad
w_{gj}\ge0,
\qquad
\sum_{j\in g}w_{gj}=1.
\]

Weights are policy choices, not quantities to infer from the observed row counts. Equal-prompt, equal-question, equal-task, worst-prompt, and severity-weighted summaries answer different questions and must be named in advance. In particular, adding prompt variants must not silently give the underlying question more policy weight.

With repeated IID rollouts within each prompt, the core implementation may form simultaneous per-prompt upper bounds \(U_j\), using preallocated levels \(\alpha_j\) satisfying \(\sum_j\alpha_j\le\alpha_g\). On the simultaneous-coverage event,

\[
\mu_g
\le
U_g
=
\sum_{j\in g}w_{gj}U_j.
\]

This construction is conservative but transparent and valid for the declared fixed-set weighted mean without pretending that prompts were randomly sampled. Worst-prompt safety can similarly use \(\max_j U_j\), and an all-mandatory-prompts rule requires every relevant prompt-level claim to pass.

If each heterogeneous fixed prompt has only one rollout, pooling the outcomes into an IID binomial model is not justified merely because the scores are binary. The result is descriptive unless a valid bounded-mean construction and the required independence assumptions are declared. The initial package should prefer repeated rollout evidence at the level where randomness is actually present.

### 2.4 Risk-sensitive aggregation over the fixed design

Let

\[
p=(p_1,\ldots,p_N)\in[0,1]^N
\]

be the rollout failure probabilities for the fixed design items, and let \(w_j\ge0\) be predeclared policy weights with \(\sum_jw_j=1\). A fixed-design risk aggregator is a deterministic functional

\[
R(p;w)
\]

of this finite vector. The weights provide a reference measure over the design for aggregation; they do **not** assert that prompts were sampled IID from a population. The library should call these quantities `design risk measures`, not population risks.

At minimum, every supported safety-oriented aggregator should be coordinatewise monotone:

\[
p_j\le q_j\ \text{for every }j
\quad\Longrightarrow\quad
R(p;w)\le R(q;w).
\]

This property both matches the safety interpretation and permits rigorous propagation of simultaneous prompt-level bounds.

#### 2.4.1 Weighted mean: neutral baseline

\[
R_{\mathrm{mean}}(p;w)=\sum_jw_jp_j.
\]

The mean is retained as a baseline and workload-style summary. It is not the default safety gate because it permits arbitrarily low-risk items to compensate for a high-risk item.

#### 2.4.2 Worst-item risk

\[
R_{\max}(p)=\max_jp_j.
\]

This is the most conservative aggregate and is appropriate when every item represents a mandatory requirement. It can be statistically expensive because the noisiest prompt-level upper bound may dominate, but it has the clearest non-compensation semantics.

#### 2.4.3 Finite-design CVaR and top-tail mean

For upper-tail level \(\beta\in[0,1)\), define

\[
R_{\operatorname{CVaR},\beta}(p;w)
=
\min_{\eta\in\mathbb R}
\left\{
\eta
+
\frac{1}{1-\beta}
\sum_jw_j(p_j-\eta)_+
\right\},
\]

where \((x)_+=\max(x,0)\). This is the weighted average of the worst \(1-\beta\) fraction of the fixed design, with fractional weight at the boundary when necessary [29]. With equal weights and an integral tail size, it is simply the mean of the largest \((1-\beta)N\) item probabilities.

Equivalently,

\[
R_{\operatorname{CVaR},\beta}(p;w)
=
\max_{q}
\left\{
\sum_jq_jp_j:
0\le q_j\le\frac{w_j}{1-\beta},\
\sum_jq_j=1
\right\}.
\]

This gives a useful fixed-design interpretation: CVaR asks how bad the average becomes under the most adverse reweighting that can concentrate at most \(1/(1-\beta)\) times the declared weight on any item. No prompt-population sampling claim is needed. As \(\beta\) increases, CVaR moves from the weighted mean toward the maximum.

The package should also expose `top_k_mean` as an easier interface for equal-weight designs, while implementing it through the same weighted-tail calculation.

#### 2.4.4 Entropic risk

For risk-aversion parameter \(\lambda>0\), define

\[
R_{\operatorname{ent},\lambda}(p;w)
=
\frac{1}{\lambda}
\log\left(\sum_jw_j e^{\lambda p_j}\right).
\]

It satisfies

\[
\lim_{\lambda\downarrow0}R_{\operatorname{ent},\lambda}(p;w)
=\sum_jw_jp_j,
\qquad
\lim_{\lambda\to\infty}R_{\operatorname{ent},\lambda}(p;w)
=\max_jp_j.
\]

Its variational representation is

\[
R_{\operatorname{ent},\lambda}(p;w)
=
\max_{q\in\Delta_N}
\left\{
\sum_jq_jp_j
-
\frac{1}{\lambda}D_{\mathrm{KL}}(q\|w)
\right\}.
\]

Thus entropic risk is a smoothly penalized adverse reweighting of the fixed design: a candidate reweighting may emphasize risky items, but pays a KL-divergence penalty for departing from the declared weights. It is monotone, convex, and translation-equivariant, but not positively homogeneous; unlike CVaR, it is a convex rather than coherent risk measure [30].

Entropic risk must be applied to the vector \((p_j)\), not to a pooled binary outcome. Applying it to one Bernoulli outcome drawn from a mixture depends only on the ordinary mean and therefore loses the cross-item heterogeneity that the measure is intended to expose.

#### 2.4.5 Power mean

For \(r\ge1\), define

\[
R_{r}(p;w)
=
\left(\sum_jw_jp_j^r\right)^{1/r}.
\]

At \(r=1\) this is the weighted mean, and as \(r\to\infty\) it approaches the maximum. It is a simple, interpretable sensitivity curve between average and worst-case behavior. It is not a coherent risk measure and should be labeled a diagnostic aggregation rather than a policy guarantee.

#### 2.4.6 Threshold exceedance and excess risk

For predeclared item or group thresholds \(\tau_j\), define the weighted exceedance mass

\[
R_{\mathrm{exc}}(p;w,\tau)
=
\sum_jw_j\mathbf 1\{p_j>\tau_j\},
\]

and weighted average excess

\[
R_{\mathrm{excess}}(p;w,\tau)
=
\sum_jw_j(p_j-\tau_j)_+.
\]

Exceedance mass answers “how much declared design weight may violate its local requirement?” Average excess also measures the magnitude of those violations. These are often easier to connect to a safety case than a free risk-aversion parameter. Because exceedance mass is discontinuous, it may remain weakly identified when several prompt intervals straddle their thresholds.

#### 2.4.7 Spectral risk measures and design quantiles

For advanced users, a spectral risk measure can average the quantiles of the weighted finite-design distribution:

\[
R_{\phi}(p;w)
=
\int_0^1Q_p(u)\phi(u)\,du,
\]

where \(Q_p\) is the weighted quantile function and \(\phi(u)\ge0\), \(\int_0^1\phi(u)du=1\), with \(\phi\) nondecreasing so higher-risk quantiles receive at least as much weight as lower-risk quantiles [31]. CVaR is a special case, and mixtures of CVaRs generate a broad class of coherent tail-sensitive summaries.

The library may report the design quantile or value-at-risk \(Q_p(\beta)\) as a diagnostic, but should not recommend it alone as a safety gate: it ignores the magnitude of risks beyond the selected quantile and is unstable near discrete weight boundaries.

#### 2.4.8 Hierarchy and replication invariance

Risk aggregators are generally not associative. For example, taking CVaR across prompts within each question and then across questions is not equivalent to taking CVaR across all prompts. The analysis plan must declare:

- the hierarchy level at which the risk measure is applied;
- whether lower levels are averaged, maximized, or risk-aggregated first;
- the weights at every aggregation level;
- all tuning parameters, including \(\beta\), \(\lambda\), \(r\), and \(\tau_j\).

If an item is duplicated or divided into variants, its original total policy weight should normally be divided among the copies. With this weight preservation, the mean, CVaR, entropic risk, power mean, and spectral measures are invariant to exact replication. Reassigning equal weight after duplication is a policy change, not a harmless data transformation.

#### 2.4.9 Statistical uncertainty

These nonlinear measures are defined on the latent prompt-level probabilities \(p_j\), not on a vector containing one realized zero-or-one outcome per prompt. In general,

\[
R(\widehat p;w)
\]

is a plug-in estimate and need not be unbiased for \(R(p;w)\); applying CVaR, a maximum, or entropic risk directly to one rollout per prompt can be especially misleading. The certified output must therefore emphasize uncertainty bounds, and the package should normally require repeated rollouts per item before allowing a nonlinear aggregate to drive a safety decision. A point estimate alone is descriptive. The package does not block such a claim, but `validate` and `analyze` degrade integrity with a `TAIL_MEASURE_SINGLE_ROLLOUT` warning when a claim on a CVaR, maximum or entropic measure has one classified rollout for the median prompt of its cohort and group.

Suppose simultaneous prompt-level bounds satisfy

\[
\Pr\left(\forall j:\ L_j\le p_j\le U_j\right)\ge1-\alpha.
\]

For every coordinatewise monotone risk aggregator above,

\[
\Pr\left(
R(L;w)\le R(p;w)\le R(U;w)
\right)
\ge1-\alpha.
\]

The core implementation should therefore output the plug-in estimate \(R(\widehat p;w)\), the propagated lower and upper bounds \(R(L;w)\) and \(R(U;w)\), and the simultaneous-coverage method used. This generic construction is conservative but auditable and does not require prompts to be IID.

For exceedance mass, simultaneous intervals provide the sharper logical bracket

\[
\sum_jw_j\mathbf 1\{L_j>\tau_j\}
\le
R_{\mathrm{exc}}(p;w,\tau)
\le
\sum_jw_j\mathbf 1\{U_j>\tau_j\}.
\]

The report should also show the certified-safe weight \(\sum_jw_j\mathbf 1\{U_j\le\tau_j\}\) and unresolved weight. These classifications are conditional on the joint prompt-level coverage event.

#### 2.4.10 Direct inference for the weighted mean

The rectangular prompt-envelope construction is unnecessarily conservative when the target is
only the fixed weighted mean

\[
\mu=\sum_jw_jp_j.
\]

With \(n_j\) attempted independent rollouts on prompt \(j\), write each rollout contribution as
\(a_{jr}Y_{jr}\), where \(a_{jr}=w_j/n_j\). Then

\[
\widehat\mu=\sum_j\sum_{r=1}^{n_j}\frac{w_j}{n_j}Y_{jr},
\qquad
\mathbb E[\widehat\mu]=\mu,
\]

without requiring equal prompt probabilities. Hoeffding's inequality gives

\[
\Pr\{\widehat\mu-\mu\ge\epsilon\}
\le
\exp\left\{-\frac{2\epsilon^2}{\sum_jw_j^2/n_j}\right\},
\]

and the analogous lower-tail bound. For endpoint error probability \(\alpha_d\), the directional
half-width is

\[
h(\alpha_d)=
\sqrt{\frac12\left(\sum_j\frac{w_j^2}{n_j}\right)
\log\frac1{\alpha_d}}.
\]

If \(x_j\) failures are classified and \(u_j\) attempted outcomes are unclassified, define

\[
\widehat\mu_{\min}=\sum_jw_j\frac{x_j}{n_j},
\qquad
\widehat\mu_{\max}=\sum_jw_j\frac{x_j+u_j}{n_j}.
\]

The missingness-robust direct interval is

\[
\left[
\max\{0,\widehat\mu_{\min}-h(\alpha_L)\},
\min\{1,\widehat\mu_{\max}+h(\alpha_U)\}
\right].
\]

This guarantee is finite-sample and does not treat prompts as IID samples. It does, however,
require the attempted rollout outcomes to be mutually independent across the complete weighted
design. This is stronger than the propagated prompt-envelope construction: Bonferroni propagation
requires valid within-prompt binomial bounds but does not require independence between prompts.
Reports must show both constructions and their assumptions rather than presenting the tighter
direct interval as an unconditional improvement.

When several group means are displayed simultaneously, the endpoint error budget is allocated
across every selected `(cohort, group)` target. If this correction is not requested, each direct
interval must be labeled pointwise.

##### Weighted KL/Chernoff interval

The preferred sparse-failure construction uses Chernoff inversion rather than an additive
half-width [32]. Index the independent atomic observations by $i$, let $a_i\ge 0$,
$\sum_i a_i=1$, and write

\[
S=\sum_i a_i Z_i,\qquad \mu=\mathbb E S.
\]

For rollout independence, each rollout from prompt $j$ has
$a_i=w_j/n_j$ and $Z_i\in\{0,1\}$. More generally, a declared independence profile partitions
these contributions into blocks $b$. The block has mass $a_b$ equal to the sum of its rollout
coefficients and normalized observation $Z_b\in[0,1]$. For `block_by: [prompt_id]`, this is the
prompt's completed empirical failure rate with $a_b=w_j$.

If all atomic weights equal $1/m$, convexity first bounds the moment-generating function of
each bounded observation by that of a Bernoulli variable with the same mean. Concavity in the
possibly heterogeneous atomic means then gives

\[
\mathbb E\exp(\lambda mS)
\le \prod_{i=1}^m(1-p_i+p_i e^\lambda)
\le (1-\mu+\mu e^\lambda)^m.
\]

Consequently, for $s<\mu$ or $s>\mu$ as appropriate,

\[
\Pr_\mu\{S\le s\}\ \text{or}\ \Pr_\mu\{S\ge s\}
\le \exp\{-m\,\mathrm{kl}(s\Vert\mu)\},
\]

where

\[
\mathrm{kl}(s\Vert\mu)
=s\log\frac{s}{\mu}+(1-s)\log\frac{1-s}{1-\mu}.
\]

The lower and upper endpoints numerically invert

\[
m\,\mathrm{kl}(s\Vert L)=\log(1/\alpha_L),\qquad
m\,\mathrm{kl}(s\Vert U)=\log(1/\alpha_U)
\]

on $L\in[0,s]$ and $U\in[s,1]$. This result does **not** require the prompts or their failure
probabilities to be identically distributed. It requires independence at the selected atomic
unit and equal atomic design weights. In the important zero-failure case it yields

\[
U=1-\alpha_U^{1/m},
\]

which is much sharper in sparse regimes than a symmetric additive bound.

When atomic weights are unequal, the implementation does not substitute the heuristic effective
sample size into the Bernoulli-KL formula. Let $b=\max_i a_i$. The chord bound for the exponential
and $a_i/b\in[0,1]$ imply

\[
\mathbb E\exp\left(\frac{\lambda S}{b}\right)
\le \exp\left\{\frac{\mu}{b}(e^\lambda-1)\right\}.
\]

Optimizing this Chernoff bound gives the weighted Poisson-divergence exponent

\[
\Pr_\mu\{S\le s\}\ \text{or}\ \Pr_\mu\{S\ge s\}
\le
\exp\left\{-\frac{d_{\mathrm P}(s\Vert\mu)}{b}\right\},
\quad
d_{\mathrm P}(s\Vert\mu)=s\log\frac{s}{\mu}-s+\mu.
\]

The endpoints invert $d_{\mathrm P}(s\Vert\mu)=b\log(1/\alpha_d)$. This fallback is rigorous for
arbitrary fixed weights, although it can be conservative when one atomic weight is much larger
than the rest. Reports identify the kernel as either `bernoulli_kl_equal_atomic_weights` or
`weighted_poisson_chernoff_max_increment` so these regimes are not conflated.

As with the other direct intervals, lower-endpoint computation uses the all-safe completion of
unclassified outcomes and upper-endpoint computation uses the all-failure completion. Simultaneous
reporting replaces each directional alpha by its prespecified multiplicity allocation before
inversion. Method choice must be prespecified rather than selected from the narrowest realized
interval.

##### Worst-of-k interval (experimental)

The `worst_of_k` measure targets the weighted chance that at least one of \(k\) fresh attempts
at a prompt fails,

\[
\theta=\sum_jw_j\theta_j,\qquad \theta_j=1-(1-p_j)^k .
\]

Its point estimate gives each prompt the share of its \(k\)-rollout subsets that contain a
failure. With \(x_j\) failures in \(n_j\ge k\) rollouts,

\[
\widehat\theta_j=1-\binom{n_j-x_j}{k}\Big/\binom{n_j}{k},
\qquad
\mathbb E\,\widehat\theta_j=\theta_j ,
\]

because \(\widehat\theta_j\) is the U-statistic of order \(k\) with kernel
\(h(Y_1,\ldots,Y_k)=\max_rY_r\). No unbiased estimator of \(\theta_j\) exists for \(k>n_j\):
the expectation of any function of \(n_j\) Bernoulli draws is a polynomial in \(p_j\) of degree
at most \(n_j\), and \(\theta_j\) has degree \(k\).

The \(n_j\) rollouts are not \(n_j\) independent observations of \(\theta_j\), so entering each
rollout as its own block would be anti-conservative. Hoeffding [32, section 5] writes a
U-statistic of order \(k\) as an average, over permutations of the \(n_j\) draws, of

\[
W_j=\frac1{m_j}\sum_{b=1}^{m_j}h(\text{block }b),\qquad m_j=\lfloor n_j/k\rfloor ,
\]

the mean of the kernel over \(m_j\) disjoint blocks of \(k\) draws. Each \(W_j\) is a mean of
\(m_j\) independent \([0,1]\) terms with mean \(\theta_j\), and convexity of the exponential gives

\[
\mathbb E\exp(\lambda\widehat\theta_j)\le\mathbb E\exp(\lambda W_j)
\quad\text{for every }\lambda .
\]

The KL/Chernoff bounds above use the sample only through such a moment-generating-function
bound. Prompt \(j\) therefore enters them as \(m_j\) blocks, each with observation
\(\widehat\theta_j\) and atomic weight \(w_j/m_j\), and the inversion is otherwise unchanged:
the Bernoulli-KL kernel when all atomic weights are equal, the weighted Poisson-Chernoff
kernel otherwise. The reported `independent_block_count` is \(\sum_jm_j\). At \(k=1\) the blocks
are the rollouts and the interval equals the rollout-level weighted KL/Chernoff interval.

Unclassified rollouts are completed as safe for the lower endpoint and as failures for the
upper endpoint. \(\widehat\theta_j\) does not decrease when a safe outcome becomes a failure,
so the completion is conservative.

Scope:

- The interval is opt-in per measure (`direct_interval: true`) and is computed for group rows
  only. It is calibrated in the direct-mean multiplicity family, next to the weighted means.
- It requires independent rollouts within each prompt and across prompts, so the primary
  independence profile must block by rollout. Sensitivity profiles get no worst-of-k interval,
  and the row says so.
- It bounds the fixed prompt set only. Sampled designs, `frontier_sidak` calibration and
  disabled direct-mean intervals are refused when the plan is validated.
- If a prompt has fewer than \(k\) attempted rollouts, the row keeps the propagated
  prompt-envelope interval and warns that the design cannot bound \(k\) attempts.

The test suite checks coverage by exact enumeration over the binomial on small designs. Rollout
blocks fall below nominal coverage there, and \(\lfloor n/k\rfloor\) blocks do not.

##### Simultaneous hierarchy-constrained confidence region

Independent node intervals ignore the fact that every displayed hierarchy mean is a linear
functional of the same fixed prompt risks. Let

\[
\theta=(p_1,\ldots,p_J)\in[0,1]^J
\]

be the prompt-risk vector for a cohort. For group or hierarchy node $v$, let $a_v$ be its
declared prompt-weight vector, padded with zeros outside that node. Its estimand is

\[
\mu_v=a_v^\top\theta.
\]

The coefficients come from the declared evaluation design. In particular, the conditional weight
of child $c$ within parent $v$ is its declared design mass divided by the parent's design mass;
it is not inferred from observed rollout counts. Thus stratification or deliberate oversampling
does not silently redefine the parent estimand.

For a fixed concentration method and independence unit, let
$[L_v^0,U_v^0]$ be the raw interval for every reported group root and internal node. Their endpoint
alphas are allocated so that the raw collection is simultaneous. Define one confidence region

\[
\mathcal C=
\left\{
\theta\in[0,1]^J:
L_v^0\le a_v^\top\theta\le U_v^0
\text{ for every reported }v
\right\}.
\]

On the raw simultaneous-coverage event, the true prompt-risk vector belongs to
$\mathcal C$. The hierarchy-aware interval is the coordinate projection

\[
L_v=\min_{\theta\in\mathcal C}a_v^\top\theta,
\qquad
U_v=\max_{\theta\in\mathcal C}a_v^\top\theta.
\]

Both optimizations are linear programs. Therefore the projected intervals preserve the original
simultaneous coverage while never widening a raw interval. Parent-child coherence is automatic:
internal nodes are not separate free parameters but different linear views of the same
$\theta$. The construction also propagates information both upward and downward through the
tree and can exploit valid constraints from overlapping declared groups.

Projection is performed separately for each concentration method and independence profile. A
rollout KL region, a rollout empirical-Bernstein region, and a prompt-blocked sensitivity region
therefore retain their own assumptions and coverage statements. They are not intersected with one
another, which would require a separate joint error allocation across methods or profiles.

If the raw inequalities have no common feasible parameter vector, the software retains the raw
intervals and emits an explicit `HIERARCHY_CONFIDENCE_REGION_EMPTY` numerical diagnostic. It does
not widen intervals or alter alpha after observing this event. Under correct assumptions, an empty
region can occur only when at least one raw simultaneous statement misses the truth (apart from
solver tolerance), so it is itself useful diagnostic information.

The projection step changes the geometry of the reported region, but it does not by itself calibrate
the raw inequalities. The package offers two raw calibrations. `flat_bonferroni` divides each
directional family budget over every distinct displayed target. `frontier_sidak` uses the declared
tree and atomic independence more efficiently, as described next. Both produce a coverage-guaranteed
raw collection before projection.

##### Tree-wide calibration by independent frontiers

Hierarchy nodes are not all independent. An ancestor and descendant reuse outcomes, and two levels
of a tree are different summaries of the same atomic rollouts. Sibling nodes in one complete level,
however, are functions of disjoint atom sets. Under the rollout-independence assumption, those
sibling statistics are mutually independent even when their Bernoulli means and sample sizes differ.
The same argument applies under any configured profile when no independent block crosses sibling
nodes. When a block does cross siblings, the implementation does not claim their independence and
uses a conservative dependent frontier allocation.

Call a collection $F$ of pairwise-disjoint nodes an **independent frontier**. For one endpoint
direction, assign the frontier an error probability $\alpha_F$, and choose node endpoint levels
$\alpha_v$ satisfying

\[
1-\prod_{v\in F}(1-\alpha_v)\le \alpha_F.
\]

For equal allocation across $m_F$ nodes, the exact Šidák value is

\[
\alpha_v=1-(1-\alpha_F)^{1/m_F}.
\]

If every node interval has marginal endpoint error at most $\alpha_v$, independence of the disjoint
node statistics gives

\[
\Pr\{\text{every endpoint in }F\text{ covers}\}
\ge \prod_{v\in F}(1-\alpha_v)
\ge 1-\alpha_F.
\]

No identical-distribution assumption is used here. The primitive rollout probabilities may vary by
prompt; only independence of the declared atoms and deterministic node membership are required.

Different frontiers generally overlap. The task frontier, domain frontier, and question frontier
are all computed from the same rollouts, so the software makes no independence claim between them.
It divides the directional family budget over the distinct frontiers by Bonferroni:

\[
\sum_F\alpha_F\le\alpha_d.
\]

Combining the exact product calculation within a frontier with the union bound across frontiers
therefore yields

\[
\Pr\{\text{every displayed endpoint in every frontier covers}\}\ge1-\alpha_d.
\]

For `two_sided` coverage at family level $1-\alpha$, the implementation uses
$\alpha_d=\alpha/2$ for each direction and unions the two endpoint events. For `per_direction`,
each direction receives the declared family alpha and the reported two-sided floor is
$1-2\alpha$.

The hierarchy compiler applies four important rules:

1. each reported level must partition the selected group into nonempty, disjoint prompt sets;
2. group roots and non-prompt hierarchy levels form candidate frontiers;
3. numerically identical targets with the same prompt coefficients are aliases and spend alpha
   only once—for example, a group root and its only task node; and
4. cohorts, overlapping groups, and different hierarchy levels are treated as potentially
   dependent frontiers unless disjointness is explicitly established within one frontier.

After this calibration, the package places all raw intervals into the linear confidence region
$\mathcal C$ and projects them jointly. Projection cannot reduce coverage because it merely uses
constraints that already hold on the raw simultaneous event.

This method is deliberately more modest than a fully joint tree-wide Chernoff maximum statistic.
It exploits the independence that the declared tree proves—disjoint siblings—without claiming a
closed-form joint law for dependent ancestor/descendant summaries. A future method could calibrate
that full dependence structure by a new finite-sample theorem or a rigorously least-favourable
optimization. The present implementation does not label simulation calibration, plug-in
covariance approximations, or resampling as an exact confidence region.

##### Variance-adaptive weighted empirical-Bernstein interval

Hoeffding uses only the bounded range and is therefore insensitive to the observed failure
variance. A second direct construction uses a data-dependent upper bound on the Bernstein
variance term. Index all independent rollout contributions by \(i\), with binary outcome \(Y_i\)
and fixed coefficient \(a_i=w_j/n_j\) for a rollout belonging to prompt \(j\). Define

\[
C_2=\sum_i a_i^2=\sum_j\frac{w_j^2}{n_j},\qquad
C_4=\sum_i a_i^4=\sum_j\frac{w_j^4}{n_j^3},\qquad
b=\max_i a_i,
\]

and the observable second-moment proxies

\[
Q_+=\sum_i a_i^2Y_i,\qquad Q_-=\sum_i a_i^2(1-Y_i).
\]

For the unknown Bernstein variance

\[
V=\sum_i a_i^2p_i(1-p_i),
\]

we have \(V\le C_2/4\), \(V\le\mathbb E Q_+\), and
\(V\le\mathbb E Q_-\). Hoeffding bounds the two expectations using the range-squared sum
\(C_4\). With variance-component error probability \(\delta_v\), a valid observable upper bound is

\[
\overline V=\min\left\{
\frac{C_2}{4},
Q_+ + \sqrt{\frac{C_4}{2}\log\frac1{\delta_v}},
Q_- + \sqrt{\frac{C_4}{2}\log\frac1{\delta_v}}
\right\}.
\]

Conditional on the event \(V\le\overline V\), inversion of Bernstein's inequality gives the
directional half-width

\[
h_{\mathrm{EB}}=
\frac{b\log(1/\delta_t)}{3}
+\sqrt{
2\overline V\log(1/\delta_t)
+\left(\frac{b\log(1/\delta_t)}{3}\right)^2
}.
\]

For an endpoint budget \(\alpha_d\), the implementation uses
\(\delta_t=\alpha_d/2\) for the Bernstein tail and
\(\delta_v=\alpha_d/4\) for each of the two variance proxies. A union bound therefore limits the
endpoint failure probability to \(\alpha_d\).

Unknown outcomes are completed as all-safe for the lower endpoint's estimate and as all-failure
for the upper endpoint's. The variance bound does not follow that completion. For both
endpoints, \(Q_+\) comes from the all-failure completion and \(Q_-\) from the all-safe one. Each is
then at least its value under any completion, so \(\overline V\) holds for the true labels
whatever they are, and the interval contains the interval of every completion.

Taking both proxies from each endpoint's own completion is not monotone. Completing an unknown
as a failure lowers \(Q_-\), and with skewed weights the width can then shrink by more than the
estimate rises.

**Fixed variance centers.** For any constant \(c\), \(\operatorname{Var}(Y)\le\mathbb E(Y-c)^2\),
with slack \((\mathbb E Y-c)^2\). A caller may therefore fix \(K\) centerings in advance, from the
design and never from the data, each a center \(c_i^{(k)}\in[0,1]\) per contribution, and use

\[
Q_{c^{(k)}}=\sum_i a_i^2\bigl(Y_i-c_i^{(k)}\bigr)^2,\qquad
\overline V=\min\left\{\frac{C_2}{4},\;\min_{k}\Bigl(Q_{c^{(k)}}
+\sqrt{\tfrac12 C_4^{(k)}\log\tfrac1{\delta_v}}\Bigr)\right\},
\]

with \(\delta_v=\alpha_d/(2K)\), so the \(K\) variance proxies share half the endpoint budget. Each
term of \(Q_{c^{(k)}}\) lies in \([0,a_i^2r_i^{(k)}]\) with
\(r_i^{(k)}=\max\{(c_i^{(k)})^2,(1-c_i^{(k)})^2\}\), so Hoeffding's padding uses the range-squared
sum \(C_4^{(k)}=\sum_i a_i^4(r_i^{(k)})^2\). For centers 0 or 1, \(r=1\) and \(C_4^{(k)}=C_4\); about
\(1/2\), \(r=1/4\) and the padding is a quarter as large. An unknown outcome contributes its larger
completion, \(r_i^{(k)}\) on the rollout unit; on the prompt-cluster unit the squared deviation is
convex in the completed rate, so its maximum lies at one of the two extreme completions. The
default, \(K=2\) with \(c\equiv0\) and \(c\equiv1\), is exactly \(Q_+\) and \(Q_-\) above.

Validity: let \(V_0=\min\{C_2/4,\min_k\mathbb E Q_{c^{(k)}}\}\), which is deterministic and at least
\(V\). On the event that all \(K\) Hoeffding bounds hold (probability at least \(1-\alpha_d/2\)),
\(\overline V\ge V_0\), and the Bernstein width is increasing in the variance, so a miss requires
the Bernstein tail at \(V_0\) (probability at most \(\alpha_d/2\)). This needs the centers to be
fixed: a center chosen from the data would make \(V_0\) random.

This is a finite-sample, heterogeneous-Bernoulli empirical-Bernstein construction, not a normal
approximation and not a plug-in standard error. It can be materially narrower than Hoeffding, but
it can remain loose near zero failures because it spends part of the error budget estimating
variance. Sparse binary safety analyses should generally prespecify weighted KL/Chernoff as the
primary interval and show empirical Bernstein and weighted Hoeffding as references. The package
does not select whichever interval looks narrowest after seeing the data.

##### Betting interval from a fixed mixture of common bets (experimental)

The three constructions above bound a moment generating function analytically. A fourth,
`weighted_betting`, tests each candidate mean directly with a nonnegative capital process and
inverts the test. Let the independent blocks have values \(Y_g\in[0,1]\), weights \(A_g\) with
\(\sum_g A_g=1\), and unknown means \(m_g\), and consider the null \(H_0(\mu):\sum_g A_g m_g=\mu\).
For a fraction \(\theta\in(0,1)\) of the maximal admissible stake define the upper-endpoint factors

\[
f_g(\theta,\mu)=1-\theta+\theta\,\frac{1-Y_g}{1-\mu},
\qquad
K_\theta(\mu)=\prod_g f_g(\theta,\mu)^{c_g},
\qquad
c_g=\frac{A_g}{\max_h A_h}\in(0,1],
\]

and symmetrically \(1-\theta+\theta Y_g/\mu\) for the lower endpoint. Under \(H_0(\mu)\),

\[
\mathbb E\,K_\theta(\mu)
\le\prod_g\big(\mathbb E f_g\big)^{c_g}
=\Big(\prod_g\big(\mathbb E f_g\big)^{A_g}\Big)^{1/\max_h A_h}
\le\Big(\sum_g A_g\,\mathbb E f_g\Big)^{1/\max_h A_h}
=1,
\]

where the first inequality is Jensen's inequality for the concave map \(x\mapsto x^{c_g}\), the
second is the weighted AM-GM inequality, and the final equality uses
\(\sum_g A_g\,\mathbb E f_g=1-\theta+\theta(1-\sum_g A_gm_g)/(1-\mu)=1\). A mixture
\(K(\mu)=\sum_j\pi_jK_{\theta_j}(\mu)\) over a prior fixed before any outcome is observed is
therefore an e-value, Markov's inequality gives \(\Pr\{K(\mu)\ge1/\alpha\}\le\alpha\), and
\(\{\mu:K(\mu)<1/\alpha\}\) is a confidence set. Its capital is monotone in \(\mu\), so the set is
an interval whose endpoint is found by bisection, and the plug-in estimate is never rejected.

Three consequences shape the implementation. First, the bet must be **common to all blocks and
fixed in advance**: with heterogeneous means the null constrains only the weighted sum, so a
fraction chosen block by block, even from earlier blocks, can profit from what earlier blocks
reveal about later ones and destroys the supermartingale property. This also means the
construction is fixed-horizon; it does not yield an anytime-valid sequence over blocks. Second, the
prior is the Beta(1/2, 1/2)-quantile grid of the universal portfolio [34] with half of its mass on
the most aggressive fraction, so the sparse-failure regime, where that fraction is optimal, pays
about \(\log 2\) rather than \(\log J\) for not knowing the best bet. Third, unequal weights enter
through the exponents \(c_g\), which lowers the growth rate in proportion to \(1/\max_g A_g\),
much as the KL fallback pays through its largest increment.

Compared with the other engines, the betting interval needs no estimated variance proxy and so
avoids the empirical-Bernstein padding term, which dominates at small block counts. It is
therefore usually the tightest construction when blocks are fractional rates, as under a
prompt-level or coarser independence profile, and it is competitive with, though not sharper
than, the exact Bernoulli-KL inversion on sparse binary rollouts with equal weights. As with every
engine, showing it as a prespecified primary or reference is valid; selecting it after seeing that
it is narrowest is not [33].

The construction is experimental (CONTRIBUTING, section 3): a plan selects it only by naming
`weighted_betting`, it is offered for direct means and hierarchy nodes but not for comparisons,
and its functions are not exported from `inspect_safety_stats.stats`.

##### Hierarchy nodes and configurable independence profiles

For every declared non-prompt hierarchy node, the package reports the conditional fixed-design
mean obtained by restricting the group to descendants of that node and renormalizing their
original design weights. This does not estimate a population of new tasks, questions, or prompts.
It answers: *under the declared policy weights, what is the mean rollout failure probability among
the fixed descendants of this node?*

The primary rollout-level construction treats all attempted rollouts across the node as mutually
independent. A prompt-blocked profile instead treats the completed within-prompt rate
($Z_j\in[0,1]$) as one observation with coefficient $w_j$. It replaces the concentration geometry
by

\[
C_2=\sum_jw_j^2,\qquad C_4=\sum_jw_j^4,\qquad b=\max_jw_j.
\]

This permits arbitrary dependence among rollouts from the same prompt while requiring independent
prompt batches. Other stable manifest fields can define coarser or crossed blocks. Such profiles
are deliberately conservative when they contain few blocks and remain sensitivity analyses, not
confidence intervals for an unseen prompt population.

When simultaneous hierarchy reporting is enabled, `flat_bonferroni` divides each directional
family budget over all selected group roots, displayed non-prompt nodes, and cohorts.
`frontier_sidak` instead divides the budget across dependent frontiers and uses Šidák product
calibration among disjoint nodes within each frontier. The resulting statement applies
simultaneously across all selected estimands for one prespecified interval construction.
Hoeffding references and cluster sensitivities carry their own construction-specific coverage
statements; the user must not choose a method post hoc and retain the same guarantee.

##### Single-valued bound for system models (experimental)

System-level safety analyses, such as fault trees and Markov models, take one number per
component rather than an interval at a chosen \(\alpha\). The option
`outputs.direct_mean_intervals.single_valued_bound` reports, beside each direct interval,

\[
M=\inf_{\delta\in(0,1)}\bigl[U_\delta+\delta\bigr],
\]

where \(U_\delta\) is the same construction's one-sided upper bound at level \(\delta\) for that
node alone. For the exact binomial bound this is \(M=\min_p[p+P_p(S\le s)]\).

*Assumptions.* Those of the underlying construction, and a family \(U_\delta\) that is
non-increasing in \(\delta\) with \(P(R>U_\delta)\le\delta\) for each fixed \(\delta\). Every
direct-mean construction in this package has both properties.

*Guarantee.* \(P(M<R)\le R\), where \(R\) is the node's true weighted-mean risk. Proof: if
\(M<R\), some \(\delta<R\) has \(U_\delta+\delta<R\), so \(U_\delta<R\). Since
\(U_R\le U_\delta\), the level-\(R\) bound then misses \(R\), which happens with probability at
most \(R\).

*Computation.* The infimum is taken over a fixed logarithmic grid of \(\delta\) from
\(10^{-12}\) to \(1-10^{-9}\), refined around the grid's minimiser. Every reported value is
\(U_\delta+\delta\) at a \(\delta\) actually evaluated, so it is never below the exact
infimum: imprecision makes \(M\) larger, never smaller. A family that rises with \(\delta\)
anywhere on the grid is refused. The report gives the \(\delta\) that attains \(M\).

*Exact check.* Enumerating the binomial distribution for the exact binomial version gives
\(P(M<R)\) of \(7.19\times10^{-4}\) at \(R=0.05\), \(N=1000\); \(4.32\times10^{-5}\) at
\(R=0.01\), \(N=1000\); and \(1.60\times10^{-2}\) at \(N=20\), \(R=0.4\). For 55 failures in
1,000 rollouts, \(M=0.0808\), against \(0.0684\) for the one-sided 95% bound. The tests check
\(P(M<R)\le R\) by exact enumeration over a grid of \(N\) and \(R\), for the exact binomial
version and for each direct-mean construction on equal-weight binary rollouts.

*Caveats.*

- The guarantee's level is the unknown risk itself: strong when \(R\le0.01\), weak on paper at
  \(R=0.2\).
- \(M\) is usually above a 95% bound. It is an input for system models, not a replacement for
  the claims' gate, and it never decides a claim.
- It holds for each node alone. It is not simultaneous over nodes, groups or cohorts, and it is
  taken from the raw node bound, not from the hierarchy projection.
- Plugging \(M\) into a system model is conservative only if the model is monotone in it, which
  rules out NOT gates.
- It is refused under a sampled design and with judge calibration
  (`PLAN_SINGLE_VALUED_BOUND_UNSUPPORTED`). With judge calibration it would bound the risk that
  the judge reports a failure, not the corrected risk.

#### 2.4.11 Recommended reporting and gating

No single scalar should be allowed to erase a mandatory safety requirement. The recommended output bundle is:

1. weighted mean as a neutral reference;
2. CVaR at one or more predeclared tail levels;
3. entropic risk at predeclared \(\lambda\) values or a clearly labeled sensitivity curve;
4. worst-item risk;
5. threshold exceedance mass, excess risk, and unresolved weight;
6. optional power-mean or spectral sensitivity analysis.

A risk aggregate may drive `PASS` only when its weights, hierarchy level, tuning parameter, threshold, and role in the decision were predeclared. Zero-tolerance failures and mandatory group failures override aggregate passes. Post hoc selection of the most favorable aggregator or tuning parameter is a multiplicity and selective-reporting problem.

### 2.5 Failure probability over a population of scenarios — deferred

Benchmark items are treated as a sample from a broader scenario population \(\mathcal P\):

\[
p_{\mathcal P} = \mathbb E_{Z\sim\mathcal P}\left[\Pr(X=1\mid Z)\right].
\]

Generalization to this quantity requires a defensible sampling or weighting story for scenarios. Running every item in a benchmark removes sampling uncertainty about the fixed benchmark average, but not uncertainty about this broader population.

This superpopulation interpretation is not part of the initial fixed-design release.

### 2.6 Failure probability under an adversarial search process — deferred

If a red team adaptively selects prompts, the observed stream follows the red-team policy, not an ordinary random sample from deployment. Its estimand may be

\[
p_{\pi} = \Pr(X_t=1\mid Z_t\sim\pi(\mathcal H_{t-1})),
\]

where \(\pi\) chooses the next case using the previous history \(\mathcal H_{t-1}\). This can be useful for discovering failures, but a raw failure proportion does not automatically estimate population prevalence.

This adversarial-stream interpretation is also outside the initial release.

### 2.7 Required declaration

Every analysis must record:

- the target population;
- the randomization or sampling mechanism;
- the definition of one independent unit;
- whether prompts, model outputs, or both are sampled;
- model version, sampling parameters, scaffold, tools, and scorer;
- whether cases were selected adaptively;
- whether inference concerns a fixed benchmark or a superpopulation.

For the initial release, the final item must be `fixed benchmark`; any adaptive selection or claimed superpopulation inference is a validation error rather than an invitation to choose a more elaborate method automatically.

NIST's recent treatment of AI evaluation statistics makes the same core distinction: uncertainty and even the definition of accuracy depend on whether benchmark items are fixed or representative of a larger population, and repeated trials create item-level clustering [13].

---

## 3. Why safety claims require one-sided inference

Ordinary benchmark reports often use two-sided intervals because they ask, “What range plausibly contains performance?” Safety gates usually ask a directional question:

> Can we rule out a failure rate greater than the maximum acceptable rate \(\tau\)?

The relevant hypotheses are

\[
H_0: p \geq \tau
\qquad\text{versus}\qquad
H_1: p < \tau.
\]

The dangerous error is a **false pass**: concluding that risk is acceptably low when \(p\geq\tau\). A one-sided upper confidence bound uses the entire error budget in the direction that protects against this error. A two-sided 95% interval allocates error to both tails and is therefore wider than necessary for the same false-pass guarantee.

The metrics-first report defaults to a two-sided family because it presents lower and upper risk
bounds for interpretation rather than making a package-level decision. With family error
probability \(\alpha\), it assigns \(\alpha/2\) to each direction. A decision-focused analysis may
explicitly select `coverage_mode: per_direction` when separate directional guarantees are the
intended contract; the report must then show the weaker combined coverage floor.

For a valid upper bound \(U(X)\),

\[
\Pr_p\{p \le U(X)\}\ge 1-\alpha
\quad\text{for every allowed }p.
\]

The pass rule

\[
\text{PASS if } U(X)\le\tau
\]

then satisfies

\[
\sup_{p>\tau}\Pr_p(\text{PASS})\le\alpha.
\]

This is the central frequentist safety guarantee: if the true risk is above the safety threshold, the procedure falsely passes it with probability no greater than \(\alpha\), under the method's assumptions.

The separate certification module expresses this result as `SUPPORTED` rather than `PASS` and
also evaluates the opposite directional test

\[
H_0:p\le\tau\quad\text{versus}\quad H_1:p>\tau.
\]

It reports `REJECTED` only when a one-sided lower bound satisfies (L>\tau). Support and rejection
use separate multiplicity families and separate error budgets: the support family controls false
certification, while the rejection family controls falsely rejecting a genuinely threshold-safe
claim. If neither bound crosses the threshold, the evidence is `INCONCLUSIVE`. This is more
efficient and semantically clearer than deriving both decisions from a generic two-sided interval.

It does **not** mean that, after observing the data, there is probability \(1-\alpha\) that \(p\le U\). That is a Bayesian interpretation and requires a prior and posterior model.

---

## 4. Exact one-sided bounds for a Bernoulli failure rate

### 4.1 Model

Assume

\[
X_1,\ldots,X_n \overset{\text{iid}}{\sim} \operatorname{Bernoulli}(p),
\qquad
S_n=\sum_{i=1}^n X_i=x.
\]

Then

\[
S_n\sim\operatorname{Binomial}(n,p).
\]

In the initial fixed-design release, this model normally applies to repeated rollouts of **one fixed prompt** under an unchanged rollout process. It must not be obtained by pooling heterogeneous prompts and treating them as if they shared one IID Bernoulli probability. Fixed-set aggregates use the declared hierarchy and weights from Section 2.3.

The maximum-likelihood estimate is \(\widehat p=x/n\), but the estimate alone does not quantify uncertainty. This is acute for rare events: observing \(0/100\) failures does not establish that the true failure rate is zero.

### 4.2 Clopper–Pearson upper bound

The one-sided Clopper–Pearson upper bound is obtained by inverting an exact binomial test. For \(x<n\), \(U_{\alpha}(x,n)\) is the solution to

\[
\Pr_{p=U_{\alpha}}(S_n\le x)
=
\sum_{k=0}^{x}{n\choose k}U_{\alpha}^{k}(1-U_{\alpha})^{n-k}
=\alpha.
\]

Equivalently,

\[
U_{\alpha}(x,n)
=F^{-1}_{\operatorname{Beta}(x+1,n-x)}(1-\alpha),
\]

with \(U_{\alpha}(n,n)=1\). The corresponding lower bound is

\[
L_{\alpha}(x,n)
=F^{-1}_{\operatorname{Beta}(x,n-x+1)}(\alpha),
\]

with \(L_{\alpha}(0,n)=0\).

Clopper and Pearson introduced the construction in 1934 [1]. “Exact” means the coverage is never below the nominal level under the binomial model:

\[
\inf_{p\in[0,1]}
\Pr_p\{L_{\alpha}(S_n,n)\le p\le U_{\alpha}(S_n,n)\}
\ge 1-2\alpha
\]

for the equal-tailed two-sided construction, with the analogous one-sided statement above. Because binomial data are discrete, the achieved coverage usually exceeds the nominal level. Thus “exact” does not mean shortest or perfectly calibrated; it generally means conservative.

### 4.3 Why it is appropriate for safety work

- It behaves correctly at the boundary \(x=0\), where Wald intervals fail badly.
- It gives a finite-sample guarantee without a normal approximation.
- Its conservatism aligns with contexts where a false declaration of safety is more costly than requiring additional data.
- It produces a bound directly comparable with a policy threshold.
- It is transparent enough to reproduce independently.

### 4.4 Why the Wald interval is not a safety default

The familiar Wald interval is

\[
\widehat p \pm z_{1-\alpha/2}
\sqrt{\frac{\widehat p(1-\widehat p)}{n}}.
\]

When \(x=0\), its estimated standard error is zero, producing the absurd interval \([0,0]\). Near zero or one and at modest \(n\), it can also extend outside \([0,1]\) and materially undercover. These are precisely the regimes common in safety evaluation.

### 4.5 Exact bounds with attempted but unclassified rollouts

Suppose (n) rollouts were attempted, (r) received usable binary labels, (x) of the labeled
rollouts were failures, and (u=n-r) remain unclassified. No missing-at-random assumption is made.
The unknown complete failure count (X^*) can be anywhere in

\[
x \le X^* \le x+u.
\]

Because both one-sided Clopper–Pearson endpoints are monotone in the failure count, a bound that
covers every binary completion is

\[
\left[L_{\alpha_L}(x,n),\;U_{\alpha_U}(x+u,n)\right].
\]

The lower endpoint treats every unknown label as safe; the upper endpoint treats every unknown
label as a failure. For the actual, unobserved completion (X^*), its ordinary exact interval is
contained within these outer endpoints. This preserves the directional coverage guarantees
without assuming that scorer failures are random. It can be very wide, which is the appropriate
cost of unresolved labels.

For monotone fixed-design risk measures—including the weighted mean, maximum, CVaR, entropic
risk, threshold excess, and the other declared v1 measures—the library propagates the vector of
prompt-level lower endpoints through the risk function for the aggregate lower bound, and the
upper endpoints for the aggregate upper bound.

This construction applies only to rollouts that were actually attempted and whose binary label is
unknown. It does not invent outcomes for rollouts that were never run; a prospective rollout
shortfall remains an integrity error.

### 4.6 End-to-end policy blocks versus underlying-model behavior

The default system-level estimand asks whether harmful content was delivered by the evaluated
end-to-end system. An explicit provider policy block during target generation can therefore be a
known safe outcome only when no target content was delivered. The event is retained as a distinct
`target_policy_blocked_safe` outcome rather than rewritten as an ordinary model refusal.

This logic does not apply to a grader-side policy block. Once target content exists, failure of a
separate classifier leaves its label unknown. It also does not apply to network errors, timeouts,
quota failures, or ambiguous provider errors. For an `underlying_model` estimand, even a
target-side policy block remains unknown because the wrapper prevented observation of the model's
counterfactual behavior.

### 4.7 Limitations and invalid uses

The exact guarantee requires the binomial model. It does not hold unchanged when:

- repeated attempts on the same task are dependent;
- prompts are selected in response to earlier outcomes;
- the model or scorer changes during collection;
- evaluation errors are silently removed;
- observations have unequal weights;
- a fixed-sample interval is repeatedly inspected and used for optional stopping;
- the benchmark is treated as representative of deployment without a sampling justification.

The library must therefore refuse to label a result “exact binomial” when clustering, weighting, or adaptive selection is declared.

### 4.8 AI-safety precedent

Exact binomial bounds are beginning to appear in direct LLM safety certification work. C3LLM uses Clopper–Pearson bounds for catastrophic-failure probabilities in multi-turn conversations [16]. LLMCert-B uses Clopper–Pearson bounds to certify probabilities of unbiased responses under specified prompt distributions [17]. Recent work on sound probabilistic safety bounds applies the construction to harmful-output probabilities [18]. A recent position paper on robustness in LLM safety evaluations also recommends reporting uncertainty using Clopper–Pearson intervals rather than unsupported point estimates [19].

These precedents support the method, while also illustrating its narrowness: each guarantee applies to a specified sampling distribution, not to all possible prompts or deployments.

---

## 5. Zero-failure sample-size planning

### 5.1 Exact derivation

If no failures are observed, \(x=0\), the Clopper–Pearson upper bound satisfies

\[
\Pr_{p=U}(S_n=0)=(1-U)^n=\alpha.
\]

Therefore

\[
U_{\alpha}(0,n)=1-\alpha^{1/n}.
\]

To ensure that zero failures would yield \(U_{\alpha}(0,n)\le\tau\), solve

\[
1-\alpha^{1/n}\le\tau.
\]

This gives

\[
n\ge
\frac{\log\alpha}{\log(1-\tau)},
\]

and hence

\[
n_{\min}
=
\left\lceil
\frac{\log\alpha}{\log(1-\tau)}
\right\rceil.
\]

For \(\alpha=0.05\) and \(\tau=0.01\),

\[
n_{\min}=299.
\]

Thus 299 independent zero-failure observations are required before the exact one-sided 95% upper bound falls below 1%.

### 5.2 The rule of three

For small \(\tau\), use \(\log(1-\tau)\approx-\tau\). At 95% confidence, \(-\log(0.05)\approx2.996\), so

\[
U_{0.05}(0,n)\approx\frac{3}{n}.
\]

This is the “rule of three” described by Hanley and Lippman-Hand [2]. It is a useful mental approximation, but the library should always compute the exact expression.

### 5.3 Why planning is a safety method, not merely a budgeting tool

Without prospective planning, an evaluation can observe no failures and still be incapable of ruling out the policy threshold. Planning exposes this before resources are spent.

It also prevents a common interpretive error:

> “We saw no catastrophic failures, so the system passed.”

The correct question is whether the sample size was large enough that zero failures excludes an unacceptably high underlying rate.

### 5.4 Multiplicity-adjusted planning

Suppose \(m\) mandatory safety claims use Bonferroni allocation \(\alpha_i=\alpha_{\mathrm{family}}/m\). Then each zero-failure sample-size requirement becomes

\[
n_i
=
\left\lceil
\frac{\log(\alpha_i)}{\log(1-\tau_i)}
\right\rceil.
\]

Multiplicity must therefore be included at planning time. Correcting only after data collection can turn a nominally adequate study into an inconclusive one.

### 5.5 Planning with a permitted number of failures

Requiring zero failures may be unrealistic when harmless labeling noise or low-severity events occur. A more general acceptance-sampling design chooses a sample size \(n\) and acceptance number \(c\). The evaluation passes if \(S_n\le c\), subject to

\[
\Pr_{p=\tau_U}(S_n\le c)
=F_{\operatorname{Binomial}(n,\tau_U)}(c)
\le\alpha_P.
\]

This controls the chance of accepting at the unacceptable rate \(\tau_U\). To ensure useful power at an acceptable rate \(\tau_A<\tau_U\), also require

\[
\Pr_{p=\tau_A}(S_n\le c)
\ge 1-\beta.
\]

The library can search integer pairs \((n,c)\) satisfying both constraints and optimize cost, expected sample size, or maximum sample size.

### 5.6 AI-safety precedent

Although explicit zero-failure planning is still uncommon in public frontier-model reports, the need is visible wherever harmful events are rare. C3LLM and probabilistic LLM-safety certification work use exact binomial bounds [16, 18], from which the zero-event requirement follows directly. OpenAI's public deployment-safety reports compare evaluation metrics and confidence intervals with Preparedness Framework thresholds [25], illustrating why the threshold, uncertainty target, and sample size need to be designed together rather than reported independently.

---

## 6. Stretch goal: Bernoulli confidence sequences

**Release status:** deferred. This section preserves the mathematical design for a later adaptive-evaluation module. It is not required for the core fixed-horizon workflow. The [phase-0 design](sequential-design.md) defines the stream, the state and the phases of that module.

For a predeclared rollout count and a single final analysis, the exact fixed-sample methods in Sections 4 and 5 are preferred: they are simpler to audit and generally yield tighter bounds. Confidence sequences become worthwhile only when observed outcomes are allowed to determine when sampling stops, which prompt receives the next rollout, or when a continuously extended evaluation is allowed to make a formal decision.

### 6.1 Why fixed-sample bounds cannot be repeatedly peeked at

A fixed-sample 95% interval guarantees coverage at a preselected sample size \(n\). It does not generally guarantee

\[
\Pr\{p\in C_t\text{ for every }t=1,2,\ldots\}\ge0.95.
\]

If researchers repeatedly calculate ordinary intervals and stop when the upper endpoint first crosses a safety threshold, the probability of a false pass can exceed the nominal \(\alpha\). This is the optional-stopping or repeated-peeking problem.

A confidence sequence \((C_t)_{t\ge1}\) instead satisfies

\[
\Pr_p\left(\forall t\ge1:\ p\in C_t\right)\ge1-\alpha.
\]

The analyst may inspect the sequence continuously, stop at a data-dependent time \(T\), and retain

\[
\Pr_p(p\in C_T)\ge1-\alpha.
\]

Howard et al. provide a modern general theory of nonasymptotic, time-uniform confidence sequences based on nonnegative supermartingales [3].

### 6.2 Test-martingale construction

For a candidate Bernoulli mean \(p\), let \(S_t=\sum_{i=1}^t X_i\). Choose a beta mixing distribution \(q\sim\operatorname{Beta}(a,b)\). The mixture likelihood ratio against \(p\) is

\[
M_t(p)
=
\int_0^1
\frac{q^{S_t}(1-q)^{t-S_t}}
{p^{S_t}(1-p)^{t-S_t}}
\frac{q^{a-1}(1-q)^{b-1}}{B(a,b)}\,dq.
\]

Evaluating the beta integral gives

\[
M_t(p)
=
\frac{B(a+S_t,b+t-S_t)}
{B(a,b)p^{S_t}(1-p)^{t-S_t}}.
\]

Under the null \(X_i\overset{\mathrm{iid}}{\sim}\operatorname{Bernoulli}(p)\), \(M_t(p)\) is a nonnegative martingale with \(M_0(p)=1\). Ville's inequality gives

\[
\Pr_p\left(\sup_{t\ge0}M_t(p)\ge\frac{1}{\alpha}\right)
\le\alpha.
\]

Inverting this family of sequential tests yields

\[
C_t
=
\left\{p\in[0,1]:M_t(p)<\frac{1}{\alpha}\right\}.
\]

At \(p=0\) and \(p=1\), the likelihood ratio is defined by its appropriate boundary limit, with impossible observations giving infinite evidence against that endpoint. With probability at least \(1-\alpha\), the true \(p\) belongs to every \(C_t\). The endpoints can be found numerically in log space. A log-beta implementation is required for stability when \(t\) is large or \(p\) is close to zero.

The constants \(a,b\) tune width but do not represent a Bayesian prior in the final frequentist guarantee. They define the mixture used to construct the martingale.

### 6.3 Betting construction under conditional means

A second construction uses predictable bets. For a proposed mean \(m\), define

\[
M_t(m)
=
\prod_{i=1}^t\left[1+\lambda_i(X_i-m)\right],
\]

where \(\lambda_i\) is chosen using only the past and constrained so every factor is nonnegative; for \(X_i\in[0,1]\), a sufficient range is \(\lambda_i\in[-1/(1-m),1/m]\) when \(m\in(0,1)\). Under a suitable null condition such as

\[
\mathbb E[X_i\mid\mathcal F_{i-1}]=m,
\]

the process is a nonnegative martingale. Under one-sided conditional-mean nulls it can be constructed as a supermartingale. Inverting the wealth process gives betting confidence sequences, including variance-adaptive empirical-Bernstein forms.

This formulation is valuable when outcomes need not be identically distributed but satisfy a conditional-mean assumption. However, it does not remove the need to define the stream whose conditional mean is being estimated.

### 6.4 Proposed later implementations

A later adaptive-evaluation module could include:

- `bernoulli_mixture`: beta-binomial mixture likelihood-ratio inversion for an IID Bernoulli stream;
- `empirical_bernstein`: a variance-adaptive bounded-mean confidence sequence under stated martingale conditions;
- `hoeffding_spending`: a simpler conservative reference method used for testing and cross-checking.

The reference construction follows directly from Hoeffding's fixed-time bound. For independent \(X_i\in[0,1]\) with common mean \(p\),

\[
\Pr\left(\left|\widehat p_t-p\right|\ge r\right)
\le 2e^{-2tr^2}.
\]

Choose a sequence \((\alpha_t)\) whose sum is at most \(\alpha\), for example

\[
\alpha_t=\frac{6\alpha}{\pi^2t^2},
\qquad
\sum_{t=1}^{\infty}\alpha_t=\alpha.
\]

Then define

\[
r_t=\sqrt{\frac{\log(2/\alpha_t)}{2t}},
\qquad
C_t=
\left[
\max(0,\widehat p_t-r_t),
\min(1,\widehat p_t+r_t)
\right].
\]

Applying the union bound over all \(t\) shows

\[
\Pr\left(\exists t:p\notin C_t\right)
\le\sum_{t=1}^{\infty}\alpha_t
\le\alpha.
\]

This sequence is usually wider than mixture or empirical-Bernstein sequences, but its derivation is elementary and independent of the production implementation. It therefore provides a valuable correctness oracle.

The variance-adaptive implementation replaces the worst-case Hoeffding variance with an observed, predictable variance process. Conceptually, it builds a family of supermartingales of the form

\[
\exp\left\{\lambda S_t-\psi(\lambda)V_t\right\},
\]

where \(S_t\) is a centered cumulative sum, \(V_t\) is the method's predictable or self-normalized variance process, and \(\psi\) is a cumulant bound valid for bounded increments. Ville's inequality supplies a boundary for each \(\lambda\); mixing over \(\lambda\), or allocating alpha across a grid and taking the best boundary, yields a curved time-uniform boundary \(u(V_t)\). Inverting

\[
|S_t|<u(V_t)
\]

produces the empirical-Bernstein confidence sequence. This explains its advantage: when failures have low realized variance, the boundary can be materially tighter than a worst-case Hoeffding boundary. The implementation must follow a published theorem exactly—including its definition of \(V_t\), allowed betting fractions, and constants—rather than treating an ordinary plug-in variance as automatically time-uniform [3].

The later sequential module should expose both a two-sided sequence and a safety-oriented upper sequence. If its first implementation obtains an upper bound by taking the upper endpoint of a two-sided \(1-\alpha\) sequence, the documentation must say that this is valid but may be less sharp than a purpose-built one-sided construction.

### 6.5 Stopping rule

Given an acceptable failure threshold \(\tau_A\), an anytime-valid pass rule is

\[
T_P=\inf\{t:U_t\le\tau_A\}.
\]

Because coverage is simultaneous over time,

\[
\sup_{p>\tau_A}\Pr_p(T_P<\infty)\le\alpha_P.
\]

Similarly, with a lower sequence \(L_t\) and unacceptable threshold \(\tau_U\),

\[
T_F=\inf\{t:L_t\ge\tau_U\}
\]

controls false-fail risk at its allocated level.

### 6.6 Adaptive prompt selection caveat

Anytime validity and population validity are separate.

If the next prompt is chosen because it is expected to be difficult, the confidence sequence may remain valid for the conditional mean of the selected stream while failing to estimate the average failure rate over the original prompt population. Population inference under unequal adaptive selection may require:

- known selection probabilities and inverse-probability weighting;
- stratified confidence sequences;
- design-based finite-population methods;
- uniform random exploration mixed into the adaptive policy.

The library must never describe optional-stopping validity as immunity to selection bias.

### 6.7 Clustering caveat

Updating a Bernoulli confidence sequence after every correlated epoch is invalid if the construction assumes independent or conditionally mean-stable increments. Safe alternatives include:

- update only when an independent cluster is complete, using its bounded cluster mean;
- construct a cluster-level martingale;
- use a design that randomizes independently across clusters.

### 6.8 AI-evaluation precedent

Sequential inference is emerging explicitly in AI evaluation. *Efficient Sequential Evaluation of Large Language Models* constructs confidence sequences by inverting test supermartingales and studies adaptive question selection [20]. CELEUS builds anytime-valid confidence intervals from e-processes and combines uncertainty-guided sample selection with a surrogate for the items not yet evaluated [37]. Both papers target the unweighted mean score of one model over a pool of items, with one outcome per item. Neither offers one-sided threshold decisions, a declared weighted estimand, several rollouts per prompt, correction for judge error, or multiplicity across claims. Both use non-uniform selection with inverse-probability weighting, but only as a sampling device, not as a weighted target. Optstop implements Bayesian adaptive stopping for Inspect AI, but its posterior-width criterion is not the same as an anytime-valid frequentist confidence sequence [21]. This remains a useful future integration point if users later require a frequentist false-pass guarantee after adaptive stopping; it does not justify adding sequential complexity to the fixed-horizon core.

### 6.9 Implemented: human labels for a judge calibration (experimental)

One confidence sequence is implemented: for the human labels of a validated judge calibration (see [Judge misclassification](judge-misclassification.md#labelling-protocol-experimental)), when the plan's labelling protocol declares `stopping: data_dependent`. The labelling of evaluation outputs, not the evaluation itself, is then allowed to stop on results.

**Model.** A judge stratum holds \(N\) rollouts, of which an unknown \(K\) are truly unsafe; the calibration parameter is \(\rho=K/N\). The labelling queue orders the stratum by a uniformly random permutation, and the labeller labels it from the front. After \(n\) labels with \(S_n\) unsafe, the labelled rows are a sample without replacement, and the probability of the observed sequence under \(K\) is proportional to the hypergeometric probability \(h_n(K)=\binom{K}{S_n}\binom{N-K}{n-S_n}/\binom{N}{n}\).

**Construction.** Put a uniform prior \(\pi_0(K)=1/(N+1)\) on \(\{0,\dots,N\}\). The posterior is \(\pi_n(K)\propto h_n(K)\), and since \(\sum_{K}\binom{K}{S}\binom{N-K}{n-S}=\binom{N+1}{n+1}\), the prior marginal likelihood of \(S_n\) is \(1/(n+1)\). So

\[
\frac{\pi_n(K)}{\pi_0(K)}=(n+1)\,h_n(K).
\]

At the true \(K^\ast\), the prior-posterior ratio \(\pi_0(K^\ast)/\pi_n(K^\ast)\) is a nonnegative martingale with initial value one (Waudby-Smith and Ramdas [39], Proposition 1). Ville's inequality gives \(\Pr(\exists n:\pi_0(K^\ast)/\pi_n(K^\ast)\ge 1/a)\le a\), so

\[
C_n=\{K:(n+1)\,h_n(K)>a\}
\]

contains \(K^\ast\) at every \(n\) simultaneously with probability at least \(1-a\). Because the guarantee is time-uniform, it holds at any stopping time, including one chosen because of the labels seen so far. \(h_n(K)\) is log-concave in \(K\), so \(C_n\) is an interval; it is reported as \([\min C_n/N,\ \max C_n/N]\). It never empties, since the average of \((n+1)h_n(K)\) over the prior is one. With no labels it is \([0,1]\), and a census pins \(\rho\) exactly, which a binomial bound cannot.

**Budget.** The calibration spends \(a=\alpha/4\) on each parameter, where \(\alpha\) is the calibration's share of the judge family budget (one per cohort, or per cohort and listed group). A two-sided miss at \(a\) bounds each one-sided miss by \(a\) as well, so the accounting of the Clopper-Pearson calibration (\(\alpha/4\) per endpoint) and the corrected interval's stated coverage carry over unchanged. It is conservative by the unused half of each parameter's share.

**Assumptions.** The queue's order within each stratum is a uniformly random permutation fixed before labelling, and which stratum the next queue row comes from does not depend on labels not yet seen. Rows are labelled in queue order without skipping, which the analysis checks as a prefix of the declared queue. Human labels are ground truth. Several strata interleaved in one queue keep each stratum's martingale, because each stratum's order is drawn independently of the others' labels.

**Validation.** `tests/unit/test_confidence_sequence.py` checks the set against an exact rational computation and runs a seeded Monte Carlo of 600 labelling paths, each stopped the first time the range excludes the truth: the sequence misses at most \(\alpha\) up to simulation error, and fixed-sample Clopper-Pearson under the same rule misses about three times as often.

---

## 7. Paired comparisons

### 7.0 Primary fixed-design model/configuration contrast

The primary comparison does not require prompt items to be random draws and does not require
candidate and baseline rollouts to be paired. For the same fixed prompts and declared weights,
define

\[
\Delta=\sum_{j=1}^J w_j(p_{Cj}-p_{Bj}),
\qquad \sum_j w_j=1.
\]

Negative \(\Delta\) means the candidate has lower fixed-design failure risk. Candidate and
baseline rollout counts may differ. With \(X_{Cjr}\) and \(X_{Bjr}\) denoting attempted binary
rollouts, the complete-data estimator is

\[
\widehat\Delta
=\sum_jw_j\left(
\frac{1}{n_{Cj}}\sum_rX_{Cjr}
-\frac{1}{n_{Bj}}\sum_rX_{Bjr}
\right).
\]

Under independent rollouts, weighted Hoeffding uses concentration geometry

\[
S_\Delta
=\sum_jw_j^2\left(\frac1{n_{Cj}}+\frac1{n_{Bj}}\right)
\]

and directional half-width

\[
h_\alpha=\sqrt{\frac{S_\Delta}{2}\log\frac1\alpha}.
\]

The empirical-Bernstein and Hoeffding implementations map the contrast to the bounded mean

\[
Q=\frac{1+\Delta}{2}
\]

by treating candidate failures and baseline safe outcomes as weighted successes. It applies the
selected direct-mean construction and transform the result back to \([-1,1]\). A generic
Bernoulli-KL bound on \(Q\) is inefficient for sparse safety risk because \(Q\) is near one half.
The weighted KL/Chernoff primary instead builds a jointly calibrated rectangle for the two sparse
arm means and reports \([L_C-U_B,U_C-L_B]\). Empirical Bernstein and Hoeffding remain direct
contrast references.

For empirical Bernstein, the variance proxies of \(Q\) are centered by arm. On the rollout unit
\(Q_++Q_-=C_2\) exactly, because every rollout counts in one of them, so in a balanced design
\(\min(Q_+,Q_-)=\tfrac{C_2}{2}(1-|\hat\pi_C-\hat\pi_B|)\), where \(\hat\pi\) are the arms' weighted
observed failure rates. Both uncentered proxies therefore lose to the cap \(C_2/4\) unless the
arms' rates differ by more than \(1/2\), however sparse the arms are, and at the cap the
interval is wider than Hoeffding's by a factor of at least
\(\sqrt{\log(2/\alpha)/\log(1/\alpha)}\). The rollout unit therefore uses the four arm-wise
centerings: candidate rollouts at \(0\) or \(1\), and baseline rollouts at \(0\) or \(1\). Candidate
\(0\) with baseline \(1\) counts both arms' failures,
\(\tfrac{C_2}{2}(\hat\pi_C+\hat\pi_B)\), which serves two sparse arms; the other three serve arms
that fail often, and two of them are \(Q_+\) and \(Q_-\). The smallest of the four is close to
\(\tfrac{C_2}{2}\bigl(\min(\hat\pi_C,1-\hat\pi_C)+\min(\hat\pi_B,1-\hat\pi_B)\bigr)\), adapting to
each arm separately. In the prompt-cluster encoding, each \(q_j=(1+p_{C,j}-p_{B,j})/2\) is
centered at \(1/2\), with range-aware padding, and the centerings \(0\) and \(1\) are kept for arms
far apart.

If candidate and baseline have unclassified attempted outcomes, the lower endpoint completes
candidate unknowns as safe and baseline unknowns as failures. The upper endpoint does the reverse.
Thus missing classifications cannot make either comparison direction look artificially favorable.

A jointly coupled prompt profile instead treats

\[
D_j=w_j(\bar X_{Cj}-\bar X_{Bj})\in[-w_j,w_j]
\]

as one independent bounded contribution. It permits arbitrary dependence among every rollout in
both arms of a prompt batch, while requiring independence between prompt batches. Its Hoeffding
geometry is \(4\sum_jw_j^2\), so repeated rollouts cannot create additional independent prompt
clusters.

Risk-ratio inference uses a simultaneous rectangular confidence set for the candidate and baseline
fixed-design means:

\[
\left[
\frac{L_C}{U_B},
\frac{U_C}{L_B}
\right].
\]

The local directional error budget is split between the two required arm endpoints, so the bound
does not assume independence between arms. If \(L_B=0\), the upper ratio bound is genuinely
unbounded and is reported as unavailable. Relative risk reduction is \(1-RR\) with transformed
endpoints.

When multiple comparisons or hierarchy nodes are requested, the chosen multiplicity family is
allocated over all of them. Under `frontier_sidak`, each comparison root and complete hierarchy
level is a frontier. Disjoint sibling nodes use independent joint prompt blocks from both arms and
receive Šidák product calibration; ancestor/descendant frontiers are Bonferroni-allocated because
they reuse data. Compatible KL difference, ratio, and relative-reduction specifications are
projections of the same two-arm region, so identical regions are aliases rather than three error
charges. Reference interval methods share the same target allocation but have coverage guarantees
construction by construction, not jointly across the alternative methods.

For hierarchy-aware empirical-Bernstein and Hoeffding risk differences, define the prompt contrast
vector

\[
d=(p_{C1}-p_{B1},\ldots,p_{CJ}-p_{BJ})\in[-1,1]^J.
\]

Every raw simultaneous node interval contributes
\(L_v^0\le w_v^\top d\le U_v^0\). The reported node bounds are the linear-program extrema of
\(w_v^\top d\) over the intersection of all such constraints. Hence parent and child effects are
different functionals of one common contrast vector, and projection cannot widen a raw interval.
The primary KL difference instead optimizes \(w_v^\top c-w_v^\top b\) over the common two-arm
region used for ratios.

For ratios, the common region instead contains candidate and baseline prompt-risk vectors
\((c,b)\in[0,1]^{2J}\). Candidate and baseline arm rectangles at every node become linear
constraints, and the implementation minimizes and maximizes

\[
\frac{w_v^\top c}{w_v^\top b}
\]

by the Charnes--Cooper transformation. An upper endpoint remains unbounded when the region permits
the denominator to approach zero with positive numerator. Relative-risk-reduction endpoints are
obtained by the exact monotone transformation \(1-RR\). The common-region projection is performed
separately for every concentration method and independence unit, so it does not silently combine
alternative assumptions.

The full implementation and usage contract is documented in
[Fixed-design model and configuration comparisons](model-comparisons.md).

#### Multi-arm confidence regions

For a declared set of (K\ge3) configurations, write (R_{kv}) for arm (k)'s fixed-design
risk at hierarchy node (v). Every generated pair and node supplies simultaneous arm constraints

\[
L_{kv}^{(s)}\le R_{kv}\le U_{kv}^{(s)},
\]

where (s) indexes a pair in which arm (k) participates. The multiplicity compiler protects the
complete collection of these constraints. Intersecting repeated constraints for a shared arm
therefore preserves the same simultaneous coverage event. For each arm separately, the package
then forms the hierarchy polytope

\[
\mathcal C_k=\left\{p_k\in[0,1]^J:
L_{kv}^{(s)}\le w_v^\top p_k\le U_{kv}^{(s)}
\text{ for every applicable }(v,s)\right\}.
\]

The displayed arm interval at node (v) is the linear projection of (mathcal C_k). Pairwise
effects are derived only after this common projection:

\[
R_{av}-R_{bv}\in[L_{av}-U_{bv},\ U_{av}-L_{bv}],
\qquad
\frac{R_{av}}{R_{bv}}\in
\left[\frac{L_{av}}{U_{bv}},\frac{U_{av}}{L_{bv}}\right].
\]

Thus the same arm has the same interval in every comparison, and all requested contrasts share one
joint coverage statement. The method is more conservative than an optimally calibrated
multivariate likelihood region because repeated shared-arm constraints still enter the family
calibration, but it is finite-sample, distribution-free apart from the declared block
independence, and does not use an asymptotic covariance estimate.

For a complete all-pairs difference family, simultaneous rank sets follow directly. Arm (k)'s
best possible rank is one plus the number of arms whose difference intervals establish lower risk
than (k); its worst possible rank is (K) minus the number of arms that (k) is established to
beat. Because every ordering statement comes from the same simultaneous event, all reported rank
sets inherit its familywise coverage.

### 7.1 Why model comparisons should be paired

Suppose a baseline model \(A\) and candidate model \(B\) are evaluated on the same cases. Let

\[
A_i,B_i\in\{0,1\}
\]

indicate safety failures. The estimand is the paired failure-rate difference

\[
\Delta
=\Pr(B_i=1)-\Pr(A_i=1).
\]

Positive \(\Delta\) means the candidate is less safe.

Using the same cases removes much of the variation caused by case difficulty. An unpaired comparison discards this information and can be substantially less precise.

### 7.2 Matched-pairs table

Define the counts

| Baseline \(A\) | Candidate \(B\) | Count | Meaning |
|---:|---:|---:|---|
| 0 | 0 | \(n_{00}\) | neither fails |
| 0 | 1 | \(n_{01}\) | candidate-only failure; regression |
| 1 | 0 | \(n_{10}\) | baseline-only failure; improvement |
| 1 | 1 | \(n_{11}\) | both fail |

Then

\[
\widehat\Delta
=\frac{n_{01}-n_{10}}{n}.
\]

Only discordant pairs determine the difference. This is a feature, not a loss: concordant cases contain no evidence about which model is safer.

### 7.3 Exact McNemar test

Under the null of equal marginal failure probabilities,

\[
H_0:\Delta=0,
\]

and conditional on the number of discordant pairs

\[
D=n_{01}+n_{10},
\]

the number of candidate-only failures satisfies

\[
n_{01}\mid D\sim\operatorname{Binomial}(D,1/2).
\]

For the one-sided alternative that the candidate is worse, the exact p-value is

\[
p_{\mathrm{McN}}
=
\Pr\{Z\ge n_{01}\},
\qquad Z\sim\operatorname{Binomial}(D,1/2).
\]

McNemar introduced the correlated-proportions test in 1947 [4]. The exact conditional form is appropriate when discordant counts are small, which is common when two strong models mostly agree.

### 7.4 Equality is not non-inferiority

Safety decisions rarely need merely to detect any nonzero difference. They need to rule out a practically unacceptable regression \(\delta>0\):

\[
H_0:\Delta\ge\delta
\qquad\text{versus}\qquad
H_1:\Delta<\delta.
\]

Passing non-inferiority requires an upper confidence bound \(U_{\Delta}\) satisfying

\[
U_{\Delta}<\delta.
\]

McNemar's equality test is the \(\delta=0\) special case and is not, by itself, a non-inferiority procedure.

The proposed primary method for margin-based paired comparisons is the Tango efficient-score interval [5]. Let the four cell probabilities be

\[
\boldsymbol\pi=(\pi_{00},\pi_{01},\pi_{10},\pi_{11}),
\qquad
\sum_{jk}\pi_{jk}=1,
\]

with

\[
\Delta=\pi_{01}-\pi_{10}.
\]

For a candidate value \(\Delta_0\), maximize the multinomial likelihood

\[
\ell(\boldsymbol\pi)
=\sum_{j,k}n_{jk}\log\pi_{jk}
\]

subject to

\[
\pi_{01}-\pi_{10}=\Delta_0.
\]

The efficient score for \(\Delta\), with the nuisance parameters estimated under this constraint, yields a standardized statistic \(Z(\Delta_0)\). The confidence interval is obtained by test inversion:

\[
C_{1-\alpha}
=\{\Delta_0:|Z(\Delta_0)|<z_{1-\alpha/2}\}.
\]

For a one-sided safety bound, use the corresponding one-sided critical value. Tango's method was developed specifically for equivalence/non-inferiority of paired proportions and handles sparse off-diagonal cells better than the simple Wald difference [5]. Newcombe's score-based paired intervals provide a computationally simpler cross-check [6].

**Implementation note.** The package ships Newcombe method 10 as an assumption-sensitive paired
diagnostic. A production Tango score solver remains deferred until it has independent
numerical-reference and coverage validation. The primary comparison path is the direct
simultaneous fixed-design contrast above, which does not treat heterogeneous prompts as IID draws.

### 7.5 Pairing requirements

The pairing key must represent the same underlying experimental case and comparable stochastic replicate. A suitable key might be

\[
(\text{task},\text{dataset item},\text{epoch},\text{scaffold condition}).
\]

The library must report and normally reject:

- duplicate keys within an arm;
- candidate-only or baseline-only cases;
- different prompts or task versions under the same key;
- mismatched generation settings that invalidate the intended comparison.

A case that both arms attempted but one did not score is not a broken pair. The package keeps it
and bounds each paired diagnostic over every completion of the unknown outcome. A one-sided exact
McNemar p-value \(P(\mathrm{Bin}(d, 1/2) \ge n_{10})\) cannot fall when a pair's difference
falls, so its maximum over the completions is attained at one of the two extreme completions.
Newcombe's bound is not monotone in the cell counts, so it is minimised and maximised over every
reachable table. See
[Pairs with an unscored arm](model-comparisons.md#pairs-with-an-unscored-arm).

### 7.6 AI-evaluation precedent

Paired analysis is increasingly being productized for model evaluation. NVIDIA NeMo Evaluator uses exact McNemar comparisons on the same problems and reports human-readable `PASS`, `WARN`, `BLOCK`, and `INCONCLUSIVE` outcomes [22]. In safety-adjacent clinical LLM evaluation, exact McNemar tests with Holm correction have been used to compare harmful-content and safety outcomes across models [27]. These are useful precedents for both the mathematics and the decision interface.

### 7.7 Planning a comparison at a margin (experimental)

`inspect-safety-stats plan comparison` and
`inspect_safety_stats.stats.planning.comparison_resolvable_differences` say, before a run,
which risk differences \(\Delta=R_C-R_B\) a fixed two-arm design can resolve at a margin
\(\delta\in(-1,1)\). They give two numbers:

- the largest true \(\Delta\) at which the upper bound \(U_\Delta\) falls at or below
  \(\delta\) with probability at least \(1-\beta\) (a non-inferiority claim at margin
  \(\delta\) passes); and
- the smallest true \(\Delta\) at which the lower bound \(L_\Delta\) reaches \(\delta\) with
  probability at least \(1-\beta\) (a regression of at least \(\delta\) is shown).

The guarantee is that of the `weighted_hoeffding` difference of section 7.0, as `analyze`
computes it, with no unclassified outcomes. That interval bounds
\(Q=(1+\Delta)/2\) by a weighted mean of independent blocks \(b\) with values in \([0,1]\) and
masses \(m_b\). With \(c^2=\sum_bm_b^2\), Hoeffding's inequality [32] gives

\[
\Pr\{\widehat Q-Q\ge s\}\le\exp\{-2s^2/c^2\},
\qquad
s(\gamma)=\sqrt{\tfrac12c^2\log(1/\gamma)},
\]

and \(U_\Delta=2\widehat Q-1+2s(\alpha_U)\). With probability at least \(1-\beta\),
\(\widehat Q<Q+s(\beta)\), and then \(U_\Delta<\Delta+2s(\alpha_U)+2s(\beta)\). So every
\(\Delta\le\delta-g_U\) with

\[
g_U=2s(\alpha_U)+2s(\beta)
\]

has \(\Pr\{U_\Delta\le\delta\}\ge1-\beta\). The lower side is symmetric, with
\(g_L=2s(\alpha_L)+2s(\beta)\) and \(\Delta\ge\delta+g_L\). A difference that would lie
outside \([-1,1]\) is reported as unattainable. Prompt probabilities may differ, and the
guarantee holds for every configuration of them.

The block masses follow the independence profile the analysis will use. For prompt \(j\) with
weight \(w_j\) and \(n_{Cj}\), \(n_{Bj}\) rollouts in the two arms:

| Unit | Profile | \(c^2\) |
|---|---|---|
| `rollout` | `block_by: [rollout]` (arms independent) | \(\sum_j\frac{w_j^2}{4}\left(\frac1{n_{Cj}}+\frac1{n_{Bj}}\right)\) |
| `pair` | `block_by: [prompt_id, repeat]`, `comparison_coupling: joint` | \(\sum_j\left[\underline n_j\frac{w_j^2}{4}\left(\frac1{n_{Cj}}+\frac1{n_{Bj}}\right)^2+(\overline n_j-\underline n_j)\frac{w_j^2}{4\overline n_j^2}\right]\) |
| `prompt_cluster` | `block_by: [prompt_id]`, `comparison_coupling: joint` | \(\sum_jw_j^2\) |

Here \(\underline n_j=\min(n_{Cj},n_{Bj})\) and \(\overline n_j=\max(n_{Cj},n_{Bj})\): a pair
block holds one rollout of each arm with the same repeat index, and repeats that only the
longer arm has form blocks of one rollout. The pair profile permits arbitrary dependence
within a pair, for example a shared seed. Hoeffding uses only the range of a block, so it
cannot gain from the correlation of paired outcomes; at equal counts a pair block has twice
the variance proxy of two independent rollouts, and the resolvable differences are
\(\sqrt2\) times further from the margin. Pairing helps the paired diagnostics of section 7.2
to 7.4, not this guarantee.

The family alpha is shared as in `analyze`: a claim's Bonferroni share
\(\alpha/M\) is split between the two bounds under `two_sided` coverage, and spent on each
under `per_direction`. \(M\) is the number of targets among which the family divides its
alpha, as under `flat_bonferroni`. Hierarchy-node contrasts and `frontier_sidak` calibration
are not planned.

**Scope.** The guarantee is not one for the `weighted_kl_chernoff` or
`weighted_empirical_bernstein` differences. Those are usually narrower on sparse failures, but
not for every design, so a design planned here may resolve more than stated under those
methods, and has no stated guarantee under them. Unclassified outcomes widen the analysis
bounds and are not planned for.

### 7.8 Arm contrasts on a generative design (experimental)

The [generative engine](generative-evaluation.md) samples terminal paths of a declared taxonomy
in which semantic seeds $s$ are a fixed stratification level and transformation levels (arms)
lie below them, with declared path weights $w(h)$. An arm contrast compares two values $a$ and
$b$ of one arm level $L$ within one target cohort. Unlike the comparisons above it is not
paired: each sampling unit carries a single value of $L$, so the two arms hold disjoint units.

**Estimand.** Write $w(s)$ for the declared weight of seed $s$, $w(s,v)$ for the total weight of
its paths with $L=v$, and $R_s(v)$ for the risk of those paths under the declared weights
normalized by $w(s,v)$. Let $S$ be the seeds with $w(s,a)>0$ and $w(s,b)>0$, and
$\pi_s=w(s)/\sum_{t\in S}w(t)$. Then

\[
\Delta=T_a-T_b,\qquad T_v=\sum_{s\in S}\pi_s\,R_s(v).
\]

Both arms are averaged over the same seed mix, so a context weight or constraint that makes $a$
likelier under some seeds than under others does not enter $\Delta$. If $w(s,v)/w(s)$ is the
same for every seed and $S$ holds every seed, $T_v$ equals the slice risk
$\sum_{h:L=v}w(h)p_h/\sum_{h:L=v}w(h)$ and $\Delta$ is a difference of two slices. A declared
slice $L=v$ is bounded by the same construction with $\pi_s=w(s,v)/\sum_t w(t,v)$ over every
seed holding $v$, which makes $T_v$ exactly the slice risk.

**Direct sampling.** Paths are drawn i.i.d. from the declared distribution, independently of
outcomes. Conditional on the number $n_v$ of draws with $L=v$ under a seed of $S$, those draws
are i.i.d. from the declared distribution restricted there, whose mass is
$W_v=\sum_{s\in S}w(s,v)$; a draw falls under seed $s$ with probability $w(s,v)/W_v$. With
the known ratio $r_s=\pi_sW_v/w(s,v)$ and the unit's failure fraction $Y\in[0,1]$ (a
failure indicator with one rollout per rewrite),

\[
\mathbb E[r_{s(j)}Y_j]=\sum_{s\in S}\frac{w(s,v)}{W_v}\,r_s\,R_s(v)=T_v.
\]

With $M=\max_s r_s$, the values $r_{s(j)}Y_j/M$ lie in $[0,1]$ and are independent with common
mean $T_v/M$, so the Bernoulli-KL (Hoeffding 1963) bound on their mean, rescaled by $M$, bounds
$T_v$ for every realized $n_v$, and therefore unconditionally. When all $r_s=1$, the
observations are Bernoulli and the bound is exact Clopper--Pearson (section 4.2) with one
rollout per rewrite. The result reports $M$ as `maximum_ratio`; the interval widens with it.

**Proportional stratified sampling.** Seeds are strata with fixed draw counts $n_s$ drawn
within the stratum from its declared distribution. Since
$\mathbb E[\mathbf 1\{L_j=v\}Y_j]=R_s(v)\,w(s,v)/w(s)$ for a draw $j$ in stratum $s$,

\[
T_v=\mathbb E\Big[\sum_{s\in S}\frac{\pi_s\,w(s)}{w(s,v)\,n_s}\sum_{j\in s}\mathbf 1\{L_j=v\}Y_j\Big],
\]

a Horvitz--Thompson sum with fixed coefficients over independent observations in $[0,1]$. After
normalizing the coefficients it is bounded by the weighted KL/Chernoff interval (section
2.4.10), rescaled by the coefficients' total. Every stratum of $S$ must have received draws;
otherwise the arm is refused.

**Unclassified rollouts** are completed conservatively in both constructions: as failures for
the upper endpoint and as safe for the lower one (section 4.5).

**The contrast** at endpoint level $\alpha_e$ bounds each arm at $\alpha_e/2$ per endpoint and
takes

\[
\big[L_a-U_b,\;U_a-L_b\big]
\]

clipped to $[-1,1]$; by the union bound it covers $\Delta$ with probability at least
$1-\alpha_e$ per endpoint. Under `one_sided_upper` the lower endpoint is reported as $-1$.

**Simultaneous family.** A plan declares slices and contrasts in `inference.slice_family`, and
the family is calibrated by independent frontiers (section 2.4.10, *Tree-wide calibration by
independent frontiers*). Under direct sampling, declared slices of one level hold disjoint
units, and conditional on their counts those are independent, so they form one Šidák frontier.
When members read shared units -- slices under proportional stratified sampling, where the
Horvitz--Thompson sum of each slice reads every unit of its strata -- each is a frontier of its
own. Every contrast is a frontier of its own. Frontiers share $\alpha$ by Bonferroni. Under
`one_sided_upper` every member's upper bound covers simultaneously with probability at least
$1-\alpha$; under `two_sided` each endpoint receives half of its member's calibrated level, and
the guarantee holds for each direction.

**Assumptions and limits.** Sampling units are independent, which path selection and rewrite
generation deliver; dependence between rollouts within a unit is unrestricted. The family must
be declared before the results are seen. It covers one cohort and the judge-reported scale
only; contrasts between cohorts use the paired construction of the comparisons. The guarantee
is about the declared design, not a population of prompts beyond it.

---

## 8. Cluster-aware comparisons

### 8.1 Why clustering matters in AI evaluations

AI evaluation observations are frequently nested:

- epochs within benchmark items;
- prompt variants within a base scenario;
- turns within a conversation;
- tasks within a task family;
- attacks generated from one jailbreak template;
- model attempts sharing an environment or seed.

Outcomes in the same cluster tend to be correlated. If the intracluster correlation is \(\rho\) and each cluster has size \(m\), the classical design-effect approximation is

\[
\operatorname{DEFF}=1+(m-1)\rho.
\]

The effective sample size is approximately

\[
n_{\mathrm{eff}}\approx\frac{n}{\operatorname{DEFF}}.
\]

Thus 1,000 attempts grouped into 20 highly correlated scenarios may provide much closer to 20 than 1,000 independent units of evidence.

`validate` and `analyze` estimate \(\rho\), the design effect and \(n_{\mathrm{eff}}\) from the data for every independence profile whose blocks group rollouts of several prompts, and report them in the `CLUSTER_DESIGN_EFFECT` diagnostic. With unequal block sizes they take \(\rho\) from the one-way ANOVA estimator and \(m=\sum_g m_g^2/N\). See [How much the grouping matters](independence-profiles.md#how-much-the-grouping-matters-cluster_design_effect).

For the initial fixed-design setting, declaring a hierarchy does not make tasks or questions random samples from a larger population. Cluster-aware analysis is used first to avoid pseudoreplication, preserve pairing, and prevent large groups from dominating through accidental row count. Resampling fixed task or question identifiers is not presented as evidence about unseen tasks unless the analysis declares a genuine cluster-sampling population. Where only rollout randomness is modeled, uncertainty should be propagated from rollouts within the fixed design; cluster bootstrap results should be labeled as sensitivity analyses rather than population confidence guarantees.

### 8.2 Cluster-level paired estimator

Let \(g=1,\ldots,G\) index independent clusters. Define the within-cluster candidate-minus-baseline difference

\[
d_g=\widehat p_{B,g}-\widehat p_{A,g}.
\]

For an equal-cluster estimand,

\[
\widehat\Delta_C=\frac{1}{G}\sum_{g=1}^G d_g.
\]

An elementary cluster-level variance estimator is

\[
\widehat{\operatorname{Var}}(\widehat\Delta_C)
=
\frac{1}{G(G-1)}
\sum_{g=1}^G(d_g-\widehat\Delta_C)^2.
\]

This treats clusters—not individual attempts—as the independent units. With sufficiently many clusters, a \(t_{G-1}\) reference distribution is a basic approximation.

### 8.3 Equal-cluster versus equal-observation estimands

When cluster sizes differ, two legitimate estimands diverge.

The equal-cluster effect is

\[
\Delta_C=\mathbb E_g[d_g],
\]

which gives every scenario family equal importance. The equal-observation effect is

\[
\Delta_O
=
\frac{\sum_g n_g d_g}{\sum_g n_g},
\]

which gives more weight to large clusters.

Neither is universally correct. In safety research, equal-observation weighting can allow a large easy task family to dominate a small catastrophic-risk family. The estimand and weights must therefore be predeclared and shown in the report.

### 8.4 Cluster bootstrap

For a paired cluster bootstrap:

1. Sample \(G\) cluster identifiers with replacement.
2. Include all baseline and candidate observations from every selected cluster.
3. Preserve pairing and within-cluster structure.
4. Recompute \(\widehat\Delta^{*(b)}\).
5. Repeat for \(b=1,\ldots,B\).

The empirical distribution of \(\widehat\Delta^{*}\) approximates sampling variation induced by drawing clusters. Percentile intervals are simple; studentized bootstrap-\(t\) intervals generally have better higher-order behavior when the standard-error estimate is stable.

Resampling individual attempts would destroy the dependence structure and produce intervals that are too narrow.

### 8.5 Wild cluster bootstrap

Ordinary cluster-robust asymptotics can over-reject when \(G\) is small or clusters are unbalanced. A wild cluster bootstrap works with cluster-level score or residual contributions. In simplified form:

1. Fit the model under the null safety margin.
2. Compute cluster score contributions \(U_g\).
3. Draw independent weights \(w_g\) with mean zero and variance one, such as Rademacher \(\{-1,+1\}\) or Webb six-point weights.
4. Form bootstrap scores \(U_g^*=w_gU_g\).
5. Recompute a studentized test statistic.
6. Use its bootstrap distribution for the p-value or inverted interval.

Cameron, Gelbach, and Miller show why bootstrap refinements matter with few clusters [8]; Cameron and Miller review the broader cluster-robust inference problem [7]. No universal cluster count guarantees reliability. The package should report the number, balance, and influence of clusters and become conservative or inconclusive when diagnostics are poor.

### 8.6 Hierarchical and model-based alternatives

Generalized linear mixed models can represent item and model effects, for example

\[
\operatorname{logit}\Pr(X_{mjk}=1)
=\beta_0+\beta_m+u_j,
\qquad
u_j\sim N(0,\sigma_u^2).
\]

They can improve precision, but their coverage depends on the random-effects and link-function assumptions. For the initial safety package, model-based intervals should be optional and clearly separated from design-based or resampling-based defaults.

NIST AI 800-3 demonstrates this tradeoff directly: hierarchical models can narrow intervals, but misspecification can reduce coverage, while regression-free methods remain more robust [13].

### 8.7 AI-safety precedent

METR reports a three-level hierarchical bootstrap over task families, tasks, and attempts for autonomous-capability evaluations [14]. This is a strong direct precedent for preserving nested evaluation structure. NIST AI 800-3 formalizes item-induced clustering and warns that incorrectly resampling both item and trial levels can overestimate variation [13]. These examples justify making hierarchy an explicit part of the library's data model rather than an optional afterthought.

These examples do not imply that every hierarchy should be resampled at every level. The appropriate bootstrap follows the estimand and the actual sources of sampling variation. METR's task-family design and NIST's benchmark/item/trial analysis answer different questions; mechanically applying a multilevel bootstrap can double-count uncertainty.

---

## 9. Multiplicity correction

### 9.1 The safety problem

Suppose \(m\) independent safety claims are each tested at level \(\alpha\). Even under the global null, the probability of at least one false rejection is

\[
1-(1-\alpha)^m.
\]

At \(m=20\) and \(\alpha=0.05\), this is approximately 64%. Dependence changes the exact value but not the underlying problem: searching across many behaviors, subgroups, models, scorers, and prompts creates opportunities for a spurious safety conclusion.

For deployment gates, the primary target should be the family-wise error rate

\[
\operatorname{FWER}
=\Pr(V\ge1),
\]

where \(V\) is the number of false decisions in the declared family.

### 9.2 Bonferroni simultaneous control

Assign claim-specific false-pass budgets \(\alpha_1,\ldots,\alpha_m\) such that

\[
\sum_{i=1}^m\alpha_i\le\alpha_{\mathrm{family}}.
\]

Then, by the union bound,

\[
\Pr(\text{one or more false passes})
\le
\sum_{i=1}^m\Pr(\text{false pass on }i)
\le
\sum_{i=1}^m\alpha_i
\le\alpha_{\mathrm{family}}.
\]

This does not require independence. Equal allocation sets \(\alpha_i=\alpha/m\). Weighted Bonferroni uses

\[
\alpha_i=\alpha w_i,
\qquad
w_i\ge0,
\qquad
\sum_iw_i=1,
\]

allowing predeclared critical risks to receive more statistical power.

Bonferroni allocation is the simplest way to produce simultaneous one-sided confidence bounds and to combine multiple confidence sequences: run claim \(i\) at level \(1-\alpha_i\).

### 9.3 The all-mandatory-claims exception

Multiplicity correction is not automatically required merely because an overall safety case has several mandatory components. If the sole global claim is

\[
\text{“every mandatory component is safe,”}
\]

and overall `PASS` occurs only when every component test rejects its unsafe null at level \(\alpha\), the procedure is an intersection-union test. Whenever the global claim is false, at least one component null is true, and a global pass requires that true null to be rejected. The probability of a false global pass is therefore at most \(\alpha\) without Bonferroni correction solely for that conjunction.

Correction is still required when the report asserts simultaneous coverage for every displayed interval, permits separate component-level safety claims, selects favorable claims after inspection, or uses “any significant result” logic. The analysis plan must record which decision semantics apply; the library must not apply multiplicity mechanically without first identifying the claim being protected.

### 9.4 Holm step-down procedure

Given valid p-values \(p_1,\ldots,p_m\), sort them:

\[
p_{(1)}\le p_{(2)}\le\cdots\le p_{(m)}.
\]

Starting with \(k=1\), reject \(H_{(k)}\) if

\[
p_{(k)}\le\frac{\alpha}{m-k+1}.
\]

Stop at the first non-rejection. Holm's procedure controls FWER under arbitrary dependence and is uniformly at least as powerful as ordinary Bonferroni for hypothesis decisions [9]. Adjusted p-values can be written

\[
\widetilde p_{(k)}
=
\max_{j\le k}\left\{(m-j+1)p_{(j)}\right\}\wedge1.
\]

The library should use Holm for fixed-sample confirmatory p-values, while using Bonferroni or weighted Bonferroni for straightforward simultaneous interval construction.

### 9.5 False discovery rate is not a deployment guarantee

The false discovery rate is

\[
\operatorname{FDR}
=\mathbb E\left[\frac{V}{\max(R,1)}\right],
\]

where \(R\) is the total number of rejections. Benjamini and Hochberg's step-up procedure provides more power when the goal is exploratory discovery [10]. But FDR control permits some false discoveries and does not control the probability of even one false safety claim.

Therefore:

- Holm/Bonferroni should be the default for mandatory safety gates.
- Benjamini–Hochberg may be offered for exploratory red-team findings.
- FDR-adjusted results must not silently produce an overall deployment `PASS`.

### 9.6 Defining a family

A family is not “whatever appears in one table.” It is the set of claims that could jointly influence a decision. Likely families include:

- all mandatory risk categories for one deployment decision;
- all candidate-versus-baseline regression claims;
- all subgroup claims used to assert absence of disparate safety behavior;
- all monitored claims examined during one sequential evaluation.

Post hoc splitting of a family after observing results undermines the guarantee. The plan and its hash should record family membership before analysis.

### 9.7 AI-safety precedent

Anthropic's study of model persuasiveness applied FDR correction to multiple pairwise comparisons [23]. Safety-adjacent clinical LLM studies have combined exact McNemar tests with Holm–Bonferroni correction [27], and a recent clinical AI safety-gating framework combines predeclared pass/fail thresholds with Holm-familywise correction [28]. These examples show that multiplicity is relevant both to frontier-risk research and to deployed high-stakes AI, though it remains inconsistently handled in general safety benchmarks.

---

## 10. Pass/fail/inconclusive safety decisions

### 10.1 Why a binary answer is statistically unsafe

A binary pass/fail scheme forces insufficient evidence into one of two categories. In safety evaluation, “we did not detect a high failure rate” is not equivalent to “we established a low failure rate.” The third outcome, `INCONCLUSIVE`, is therefore not a reporting convenience; it is necessary to distinguish evidence of absence from absence of evidence.

### 10.2 Two-threshold design

For each claim, predeclare:

- \(\tau_A\): the largest clearly acceptable failure rate;
- \(\tau_U\): the smallest clearly unacceptable failure rate;
- \(\alpha_P\): maximum false-pass probability;
- \(\alpha_F\): maximum false-fail probability;

with

\[
0\le\tau_A<\tau_U\le1.
\]

Let \(U_{1-\alpha_P}\) be a one-sided upper confidence bound and \(L_{1-\alpha_F}\) a one-sided lower bound. Define

\[
\text{Decision}=
\begin{cases}
\text{PASS}, & U_{1-\alpha_P}\le\tau_A,\\
\text{FAIL}, & L_{1-\alpha_F}\ge\tau_U,\\
\text{INCONCLUSIVE}, & \text{otherwise.}
\end{cases}
\]

The interval \((\tau_A,\tau_U)\) is an indifference or transition region. It prevents the study from needing enormous sample sizes to distinguish values that are operationally equivalent and prevents tiny, statistically significant changes from dominating safety decisions.

### 10.3 Error guarantees

For any \(p>\tau_A\),

\[
\Pr_p(\text{PASS})\le\alpha_P.
\]

For any \(p<\tau_U\),

\[
\Pr_p(\text{FAIL})\le\alpha_F.
\]

These errors need not be symmetric. In a catastrophic-risk gate, \(\alpha_P\) may be much smaller because a false pass is more costly than a false fail.

### 10.4 Single-threshold mode

If policy supplies only one threshold \(\tau\), the library may define

\[
\text{PASS if }U\le\tau,
\qquad
\text{FAIL if }L>\tau,
\]

and `INCONCLUSIVE` otherwise. Two thresholds are preferable because they separate acceptable from clearly unacceptable risk and permit explicit power planning.

### 10.5 Policy overrides

Some failures are intolerable regardless of their estimated population rate. The statistical result and policy result must remain separate:

```text
Statistical evidence: INCONCLUSIVE
Policy outcome: FAIL
Reason: one verified zero-tolerance catastrophic event
```

Likewise, no statistical pass should override unresolved evaluation-integrity problems unless
every unclassified attempted outcome is explicitly covered by a missingness-robust certifying
bound and the adverse completion still passes.

### 10.6 Overall roll-up

For mandatory claims \(i=1,\ldots,m\):

- overall `FAIL` if any zero-tolerance rule fires or any mandatory claim fails;
- overall `PASS` only if every mandatory claim passes and every data-quality gate passes;
- overall `INCONCLUSIVE` otherwise.

This asymmetric roll-up reflects the logic of a safety case: all required claims need support, while one material counterexample may block deployment.

### 10.7 Data-quality gates

The report uses three integrity states. `valid` means all required outcomes are classified;
`degraded` means attempted outcomes are unclassified but conservative completion bounds cover
them; `invalid` means a structural or strict-policy error prevents certification. A degraded run
may certify only if the adverse completion of every unknown outcome supports the claim.

The default outcome must be `INCONCLUSIVE`, not pass, when any of the following is unresolved and
not covered by a valid conservative construction:

- missing scores or terminal evaluation errors;
- scorer outputs outside the declared mapping;
- model, prompt, scaffold, or scorer drift during the run;
- broken comparison pairs;
- undeclared clustering or adaptive selection;
- too few independent clusters for the selected approximation;
- use of fixed-sample inference after data-dependent stopping;
- material post hoc changes to thresholds, families, or outcomes.

An optional `count_as_failure` policy can conservatively classify particular errors as failures,
but the report must retain their original status. The default end-to-end policy may instead count
an explicit target-generation policy block as safe only when no content was delivered. Grader-side
blocks remain unclassified.

### 10.8 AI-safety precedent

Frontier-AI governance already relies on threshold-linked decisions. OpenAI's Preparedness Framework assesses whether capability thresholds have been crossed and allows decision-makers to request further evaluation or stronger safeguards rather than forcing an immediate approval [24]. OpenAI deployment reports publish confidence intervals around metrics compared with Preparedness thresholds [25]. Anthropic's Responsible Scaling Policy similarly connects capability thresholds with required safeguards while acknowledging that evaluation science may not provide dispositive answers [26].

The proposed `INCONCLUSIVE` status operationalizes that uncertainty explicitly. NVIDIA's model-comparison tooling is a closer software precedent: it exposes an inconclusive outcome when paired evidence is insufficient [22].

---

## 11. Stretch goal: combining sequential monitoring and multiplicity

**Release status:** deferred with the confidence-sequence module. The core release performs one predeclared analysis at its fixed horizon and uses the fixed-sample multiplicity rules in Section 9.

Suppose \(m\) safety claims are monitored continuously. For claim \(i\), construct a confidence sequence satisfying

\[
\Pr\left(\exists t:p_i\notin C_{i,t}\right)\le\alpha_i.
\]

If

\[
\sum_{i=1}^m\alpha_i\le\alpha_{\mathrm{family}},
\]

then

\[
\Pr\left(\exists i,t:p_i\notin C_{i,t}\right)
\le\alpha_{\mathrm{family}}.
\]

This single union-bound argument controls both dimensions:

- repeated looks over time;
- repeated claims across the safety family.

It is simple and conservative, which makes it an appropriate first implementation within the later sequential module. More powerful online multiple-testing or e-value procedures may follow, but they should not obscure the guarantee.

The analysis plan must freeze each claim's \(\alpha_i\) before monitoring begins. Adding claims midstream requires reserving alpha in advance or starting a new declared family.

### 11.1 Attackers that adapt across behaviours (experimental)

**Release status:** implemented as `adaptive_claims` in the plan (#125), in `stats/adaptive.py`. The rest of this section's sequential module remains deferred.

An attacker that carries what it learned on one behaviour into the next makes its runs dependent: the success probability of run \(i\) depends on runs \(1,\ldots,i-1\). The binomial bounds of Section 4 assume independent runs and do not apply. Let \(X_i\in\{0,1\}\) indicate that run \(i\) succeeded, \(S_t=\sum_{i\le t}X_i\), and

\[
m_i=\Pr(X_i=1\mid X_1,\ldots,X_{i-1}),\qquad
\bar m_t=\frac1t\sum_{i\le t}m_i .
\]

The estimand is \(\bar m_t\), the attacker's running average of conditional success probabilities for the declared attacker, budget, behaviours and order. The order of the behaviours must be fixed in advance or randomised; the attacker may adapt within and across runs.

**Guarantee.** Fix \(\lambda>0\) and let \(\psi_\lambda(m)=\log(1-m+m e^\lambda)\), the cumulant generating function of a Bernoulli(\(m\)) variable. Since \(\mathbb E[e^{\lambda X_i}\mid X_1,\ldots,X_{i-1}]=e^{\psi_\lambda(m_i)}\), the process

\[
N_t(\lambda)=\exp\Bigl(\lambda S_t-\sum_{i\le t}\psi_\lambda(m_i)\Bigr)
\]

is a nonnegative martingale with \(N_0=1\). \(\psi_\lambda\) is concave, so by Jensen's inequality \(\sum_{i\le t}\psi_\lambda(m_i)\le t\,\psi_\lambda(\bar m_t)\), and

\[
G_t(\lambda)=\exp\bigl(\lambda S_t-t\,\psi_\lambda(\bar m_t)\bigr)\le N_t(\lambda).
\]

For weights \(w_j>0\) summing to one on values \(\lambda_j>0\), all fixed before the data, \(N_t=\sum_j w_j N_t(\lambda_j)\) is again a nonnegative martingale with \(N_0=1\), and \(G_t=\sum_j w_j G_t(\lambda_j)\le N_t\). Ville's inequality [40] gives \(\Pr(\exists t:N_t\ge 1/\alpha)\le\alpha\). Define

\[
L_t=\inf\Bigl\{m\in[0,1]:\sum_j w_j\exp\bigl(\lambda_j S_t-t\,\psi_{\lambda_j}(m)\bigr)<1/\alpha\Bigr\}.
\]

Each term falls in \(m\), because \(\psi_\lambda\) rises in \(m\) for \(\lambda>0\), so the set is an interval \([L_t,1]\). If \(L_t>\bar m_t\) then \(G_t\ge 1/\alpha\), hence \(N_t\ge 1/\alpha\), and therefore

\[
\Pr\bigl(\exists t: L_t>\bar m_t\bigr)\le\alpha .
\]

The mixture must not depend on \(m\): a mixture chosen at each \(m\) separately would be evaluated at the random \(\bar m_t\), and \(N_t\) would no longer be a martingale. The construction is that of Howard et al. [3], with the Bernoulli cumulant generating function in place of a sub-exponential bound; Waudby-Smith and Ramdas [33] study the betting form of such processes.

**What is implemented.** The mixture is 20 values of \(\lambda\), spaced geometrically from 0.05 to 5, with equal weights. \(L_t\) is the root of the mixture against \(1/\alpha\), found to an absolute tolerance of \(10^{-12}\) and then lowered by that tolerance, with \(\psi_\lambda(m)\) evaluated as `log1p(m * expm1(λ))` so that tiny \(m\) stay exact. The claim is `FAIL` when \(L_t\) is at or above `unacceptable_lower`, at the family's alpha.

**Counting what is missing.** A behaviour inside the stream with no run, or with an unclassified outcome, is counted as \(X_i=0\). That lowers \(S_t\) pathwise, so \(G_t\) only falls and the guarantee holds as stated.

**Why FAIL only.** An upper bound would also be valid for this attacker. It is not offered, because a small success rate for one adaptive attacker at one budget is no evidence about other attackers or larger budgets, and a `PASS` would read as such evidence. A large one is evidence enough to fail a gate.

**Width.** With \(\alpha=0.05\), 20 successes in 50 runs give \(L_{50}=0.2097\), against 0.2831 for the fixed-horizon one-sided Clopper–Pearson lower bound, which assumes independent runs and one look. 2 in 20 give 0.0033 against 0.0181, and 30 in 300 give 0.0552 against 0.0729.

**Coverage.** `research/adaptive_attacker_coverage.py` draws 20,000 streams of 300 runs from four attackers and recomputes the bound after every run. At \(\alpha=0.05\), the bound rose above \(\bar m_t\) at some \(t\) on 0.0135 of streams for an attacker whose success probability grows with each success (0.05 up to 0.6), 0.0187 for one that loses with each failure (0.6 down to 0.02), 0.0049 for one at 0.9 after a success and 0.1 after a failure, and 0.0232 for independent runs at 0.3. `tests/unit/test_adaptive_bound.py` repeats the first check at a smaller size.

---

## 12. Relationship to Optstop

Optstop and this library answer related but different questions.

Optstop performs Bayesian adaptive allocation and stopping based primarily on posterior precision. Its output is useful for deciding where additional evaluation effort is valuable. Its recent paper reports substantial compute savings and empirical checks of truncation behavior [21]. The same paper is explicit that the two tools give different guarantees. In its simulation study (App. B.6, Table 5), 94% credible intervals covered the true rate about 80% of the time in the middle of the range (0.80 for the logit-normal model, 0.81 for the beta-binomial), and 0% of the time at the exact boundaries of 0 and 1. The paper attributes that zero to the degenerate data-generating process rather than to the model. It describes its interval-width stopping threshold as "a precision target rather than an accuracy guarantee" (App. A.1.1). Such a threshold says nothing about false passes, which is what the core gate controls.

The core package focuses on fixed-horizon frequentist safety guarantees:

- exact fixed-sample upper bounds;
- explicit false-pass control;
- familywise guarantees across safety claims;
- paired and cluster-aware safety regressions;
- pass/fail/inconclusive decisions tied to policy thresholds.

The initial package does not attempt to duplicate or certify Optstop's adaptive procedure. If an Optstop policy uses observed outcomes to stop or reallocate sampling, ordinary fixed-sample bounds from this package must not be presented as optional-stopping-valid. Such a run should be reported as adaptively collected and therefore `INCONCLUSIVE` under the core frequentist gate, or followed by a fresh, prespecified confirmatory run.

A later confidence-sequence module could allow Optstop to choose what to evaluate while this package audits what can be concluded under an anytime-valid frequentist guarantee. Across current and future modules, the report must identify whether an interval is:

- Bayesian posterior/credible;
- fixed-sample frequentist;
- anytime-valid frequentist;
- bootstrap or asymptotic;
- simultaneous across a claim family.

These labels are not interchangeable.

---

## 13. Inspect AI integration and auditability

Inspect's analysis APIs can extract one row per sample together with evaluation, model, score, ID, and epoch fields [11]. The library should use these supported APIs rather than parsing the internal `.eval` representation.

The offline, fixed-horizon path is the only decision-producing path required for the core release:

\[
\text{Inspect logs}
\rightarrow
\text{normalized safety observations}
\rightarrow
\text{validated analysis plan}
\rightarrow
\text{statistical results}
\rightarrow
\text{sidecar report}.
\]

Inspect exposes sample-end hooks and an `EarlyStopping` protocol [12], but live decision-making is deferred. A future sequential module would need serialized and durable state because hook callbacks may be concurrent, and should operate in shadow mode before it is permitted to stop evaluation runs. The core release may display descriptive progress, but only the prespecified final analysis may produce a statistical `PASS`.

Every result should include:

- a cryptographic hash of the analysis plan;
- source log identifiers;
- package and Inspect versions;
- model and scorer identities;
- outcome mapping;
- ordering, pair, cluster, and stratum keys;
- missing-data decisions;
- stopping rule;
- confidence method and error budget;
- multiplicity family;
- warnings and failed assumptions.

The original Inspect logs should remain immutable; statistical reports should be separate artifacts linked back to those logs.

---

## 14. Method-selection table

| Priority | Evaluation design | Primary method | Validity target | Main assumptions | Key warning |
|---|---|---|---|---|---|
| Core | Fixed \(n\), repeated rollouts for one fixed prompt | One-sided Clopper–Pearson | Finite-sample false-pass control | IID Bernoulli rollouts | Do not pool heterogeneous prompts as IID |
| Core | Zero observed failures | Exact zero-event bound and planner | Rule out \(p>\tau\) | Same as binomial bound | Zero failures never means zero risk |
| Core | Fixed weighted group | Mean, maximum, CVaR/top-tail mean, entropic risk, threshold excess, and optional power/spectral measures | Fixed-design risk summary with propagated simultaneous bounds | Valid prompt bounds; predeclared hierarchy, weights, parameters, and aggregation level | A scalar aggregate cannot override mandatory hazards and does not generalize to unseen prompts |
| Core | Two models on the same fixed cases | Exact McNemar plus paired score interval | Paired equality or margin inference | Correct pair matching; independent pairs under the stated null | Information comes from discordant pairs |
| Core | Repeated attempts within declared groups | Design-respecting cluster summaries; bootstrap only with a justified sampling interpretation | Avoid pseudoreplication; optionally cluster-level sampling uncertainty | Declared hierarchy and source of randomness | Fixed groups are not automatically a sampled population |
| Core | Multiple safety claims | Intersection-union logic, Holm tests, or Bonferroni simultaneous bounds as appropriate | Protect the declared global or simultaneous claim | Valid component methods and predeclared decision semantics | Correction is claim-dependent, not automatic |
| Core | Exploratory discovery across many slices | Benjamini–Hochberg | False discovery rate | Standard BH dependence conditions | Must not drive deployment pass |
| Core | Deployment gate | Two-threshold decision | Controlled false pass/fail | Valid bounds and predeclared thresholds | Inconclusive is a required outcome |
| Experimental | Two values of one arm level of a generative design | Seed-matched arm contrast, simultaneous over a declared slice family (section 7.8) | Simultaneous coverage of declared slices and contrasts | Independent sampling units; family declared in the frozen plan | Not paired; widens with `maximum_ratio` when the declared weights skew a value across seeds |
| Stretch | Continuously monitored or adaptively allocated rollouts | Bernoulli confidence sequence | Time-uniform coverage | Method-specific IID or martingale conditions | Adaptive selection can change the estimand |

---

## 15. Known non-statistical threats to validity

No method in this document controls the following by itself:

### 15.1 Under-elicitation

A model may possess a dangerous capability that the scaffold, prompt, tools, or budget fail to elicit. A low observed capability rate may therefore be a lower bound on capability rather than evidence of safety.

### 15.2 Evaluation awareness and sandbagging

A model may behave differently when it recognizes an evaluation. Statistical precision around evaluation behavior does not eliminate this external-validity threat. Anthropic's sabotage evaluations include explicit sandbagging and oversight-undermining experiments that illustrate why behavior observed under evaluation can be strategically unrepresentative [15].

Every bound in this document is therefore conditional on the evaluated model not strategically
underperforming. A model that deliberately fails to show a capability, or behaves more safely
because it recognizes the evaluation, is measured correctly for the behaviour it chose to show;
no amount of rollouts makes that behaviour representative [33]. Detecting sandbagging is a
separate evaluation. One proposed method injects noise into the model's weights: a model that
is not sandbagging degrades steadily as the noise grows, while a sandbagging model can improve
anomalously. Its authors implemented it in Inspect [34]. This package does not run such checks.

### 15.3 Scorer error

If an automated judge has false negatives, the observed binary outcome is a misclassified version of the true event. Let sensitivity be \(s\) and specificity be \(c\). The observed positive rate is

\[
p_{\mathrm{obs}}=s p_{\mathrm{true}}+(1-c)(1-p_{\mathrm{true}}).
\]

Without knowledge or bounds on \(s\) and \(c\), \(p_{\mathrm{true}}\) is not identified from \(p_{\mathrm{obs}}\). [Judge misclassification](judge-misclassification.md) describes the corrections built on this identity: declared accuracy scenarios as a sensitivity analysis, validated corrections from human labels (with an experimental stratified, finite-population variant that bounds each declared stratum with exact hypergeometric bounds), the noisy-but-valid threshold test, and an experimental stress analysis. The stress analysis reports, for a threshold verdict reached on \(p_{\mathrm{obs}}\), the line in \((s,c)\) beyond which the same verdict no longer holds for \(p_{\mathrm{true}}\).

### 15.4 Distribution shift

A valid bound for the evaluation distribution does not cover a deployment distribution with different users, languages, tools, incentives, or attack strategies.

### 15.5 Benchmark contamination

Exposure to benchmark items can change the measured behavior and invalidate the interpretation of items as fresh samples.

### 15.6 Severity aggregation

A single binary event rate treats all failures alike. Critical failures should be separate claims or zero-tolerance rules rather than diluted into a broad average.

### 15.7 Residual risk of rare harms

The zero-failure planning in section 5 makes the cost of rare-harm claims explicit: an upper
bound of \(\theta\) at confidence \(1-\alpha\) needs about \(\ln(1/\alpha)/\theta\) failure-free
trials, about \(3/\theta\) at 95%, so
every factor of ten in the rate costs a factor of ten in rollouts. Validation of automated
driving faced the same arithmetic. Kalra and Paddock showed that demonstrating a fatality rate
comparable to human drivers by test driving alone would take hundreds of millions to hundreds
of billions of miles. They concluded that test driving alone cannot provide sufficient
evidence of safety, that new ways of demonstrating it are needed, that even with them safety may
not be established with certainty, and that regulation must therefore adapt as the technology
and the evidence develop [35].

The same holds here. A pre-deployment evaluation can bound rates that its rollout budget can
reach. Below that, the residual risk is not shown to be small, only not measured. Claims about
rarer harms need other evidence, such as monitoring in deployment, incident reporting and
arguments about the system's design.

These limitations belong prominently in every report because safety evidence is only as strong as both its statistical and substantive validity.

---

## 16. Validation requirements for the implementation

### 16.1 Exact-binomial validation

- Enumerate all \(x=0,\ldots,n\) for a dense grid of \(p\) and verify coverage is at least nominal.
- Compare endpoints with trusted R and SciPy implementations.
- Test boundary values \(x=0\), \(x=n\), tiny \(\alpha\), and large \(n\).
- Verify monotonicity: the upper bound decreases with \(n\) for fixed \(x=0\) and increases with \(x\) for fixed \(n\).

### 16.2 Fixed-design and aggregation validation

- Verify that every rollout maps to exactly one declared prompt and that creator-supplied task, question, prompt, and group identifiers are internally consistent.
- Reject undeclared or ambiguous hierarchy, weights derived accidentally from row counts, and pooling of heterogeneous prompts under an IID-binomial label.
- Enumerate and simulate simultaneous per-prompt bounds and verify coverage of each declared weighted aggregate.
- Verify equal-prompt, equal-question, equal-task, worst-prompt, and all-mandatory aggregation against hand-computed fixtures.
- Verify weighted CVaR against both its threshold-minimization and adverse-reweighting formulations, including fractional mass at discrete tail boundaries.
- Verify entropic-risk calculations in log space and their weighted-mean and maximum limits as \(\lambda\) varies.
- Verify power-mean limits, spectral-weight normalization, design quantiles, exceedance mass, and threshold-excess calculations.
- Property-test coordinatewise monotonicity for every supported risk aggregator.
- Verify replication invariance when an item's original weight is divided across identical copies, and verify that unadjusted equal-weight duplication is reported as a changed design.
- Verify that substituting simultaneous lower and upper prompt bounds produces valid risk-measure bounds in simulation.
- Verify the logical lower and upper brackets for threshold-exceedance mass and the reported certified-safe and unresolved weights.
- Enumerate small heterogeneous Bernoulli designs to verify weighted KL/Chernoff coverage in both
  the equal-atomic Bernoulli-KL regime and unequal-weight Poisson-Chernoff fallback; test zero and
  all-failure endpoint formulas, inversion monotonicity, and reported kernel selection.
- Confirm that reports always carry the fixed-design external-validity disclaimer.

### 16.3 Paired validation

- Enumerate small matched tables and compare exact McNemar p-values with trusted implementations.
- Reproduce published Tango/Newcombe examples.
- Simulate coverage over sparse and boundary matched tables.
- Test duplicate, missing, and mismatched pair handling.

### 16.4 Cluster validation

- Simulate varying intracluster correlation, cluster counts, and cluster-size imbalance.
- Measure coverage and false-positive rates for cluster bootstrap and wild cluster bootstrap.
- Demonstrate the undercoverage of naive row-level methods.
- Compare equal-cluster and equal-observation estimands.
- Verify that fixed clusters are not described as a sampled population unless that sampling design is explicitly declared.

### 16.5 Multiplicity validation

- Simulate independent, positively dependent, and adversarially dependent p-value families.
- Verify Holm and Bonferroni familywise control.
- Verify simultaneous coverage of alpha-allocated fixed-sample bounds.
- Verify by enumeration or simulation that the all-mandatory intersection-union pass rule controls false pass without unnecessary Bonferroni adjustment.
- Confirm that exploratory FDR results cannot create a deployment pass.

### 16.6 Decision-engine validation

- Property-test every combination of bounds and thresholds.
- Verify monotonicity: stronger evidence cannot turn pass into inconclusive unless the plan changes.
- Verify any unresolved hard data-quality error blocks pass.
- Verify plan hashes and reports are deterministic.

### 16.7 Stretch-goal confidence-sequence validation

These tests are deferred until the sequential module is implemented:

- simulate fixed and data-dependent stopping times;
- estimate \(\Pr_p(\exists t:p\notin C_t)\) over a grid of \(p\);
- test numerical inversion against direct grid inversion;
- test replay/resume equivalence;
- verify failure under deliberately violated dependence assumptions and surface that limitation in documentation.

---

## 17. Evidence of use in AI safety and AI evaluation

The methods in this design have different levels of adoption. The table distinguishes direct safety usage from adjacent evaluation precedent.

| Method | Example use | Relevance |
|---|---|---|
| Exact Clopper–Pearson bounds | C3LLM catastrophic conversational-risk certification [16]; LLMCert-B [17]; sound harmful-output probability bounds [18] | Direct AI/LLM safety certification |
| Zero-event interpretation | Follows directly from the exact bounds used above; public frontier reports increasingly compare uncertain metrics with risk thresholds [25] | Direct mathematical implication; explicit planning remains a gap |
| Fixed-design risk aggregation | CVaR [29], entropic risk [30], and spectral risk measures [31] | Strong mathematical foundations for risk-sensitive finite scenarios; use over fixed AI-evaluation designs is proposed here rather than claimed as established AI-safety practice |
| Confidence sequences | Sequential LLM evaluation using test supermartingales [20] | Direct AI evaluation; applicable to live safety monitoring |
| Bayesian adaptive stopping | Optstop for Inspect AI [21] | Direct Inspect precedent, but a different inferential guarantee |
| Paired McNemar comparisons | NeMo Evaluator regression gates [22]; clinical LLM safety comparisons [27] | Direct AI comparison and safety-adjacent deployment evidence |
| Hierarchical/cluster bootstrap | METR task-family/task/attempt bootstrap [14] | Direct frontier autonomous-capability evaluation |
| Hierarchical statistical modeling | NIST AI 800-3 [13] | Official AI evaluation guidance; clarifies item and trial clustering |
| FDR or Holm correction | Anthropic persuasiveness evaluation [23]; clinical LLM safety studies [27, 28] | Safety-relevant and high-stakes AI precedent |
| Threshold-linked decisions | OpenAI Preparedness Framework and deployment reports [24, 25]; Anthropic RSP [26] | Direct frontier-risk governance precedent |
| Explicit inconclusive outcome | NeMo's paired regression workflow [22]; “request further evaluation” logic in preparedness review [24] | Adjacent software precedent and direct governance analogue |

The overall gap is not absence of statistical ideas. It is the absence of a unified, auditable workflow that connects:

\[
\text{evaluation design}
\rightarrow
\text{valid uncertainty}
\rightarrow
\text{multiplicity control}
\rightarrow
\text{safety threshold}
\rightarrow
\text{decision and limitations}.
\]

That is the role this Inspect extension should fill.

---

## 18. References

### Statistical foundations

1. C. J. Clopper and E. S. Pearson, “The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial,” *Biometrika* 26(4), 404–413, 1934. [doi:10.1093/biomet/26.4.404](https://doi.org/10.1093/biomet/26.4.404)
2. J. A. Hanley and A. Lippman-Hand, “If Nothing Goes Wrong, Is Everything All Right? Interpreting Zero Numerators,” *JAMA* 249(13), 1743–1745, 1983. [doi:10.1001/jama.1983.03330370053031](https://doi.org/10.1001/jama.1983.03330370053031)
3. S. R. Howard, A. Ramdas, J. McAuliffe, and J. Sekhon, “Time-uniform, nonparametric, nonasymptotic confidence sequences,” *Annals of Statistics* 49(2), 1055–1080, 2021. [doi:10.1214/20-AOS1991](https://doi.org/10.1214/20-AOS1991)
4. Q. McNemar, “Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages,” *Psychometrika* 12, 153–157, 1947. [doi:10.1007/BF02295996](https://doi.org/10.1007/BF02295996)
5. T. Tango, “Equivalence test and confidence interval for the difference in proportions for the paired-sample design,” *Statistics in Medicine* 17(8), 891–908, 1998. [doi:10.1002/(SICI)1097-0258(19980430)17:8%3C891::AID-SIM780%3E3.0.CO;2-B](https://doi.org/10.1002/(SICI)1097-0258(19980430)17:8%3C891::AID-SIM780%3E3.0.CO;2-B)
6. R. G. Newcombe, “Improved confidence intervals for the difference between binomial proportions based on paired data,” *Statistics in Medicine* 17(22), 2635–2650, 1998. [doi:10.1002/(SICI)1097-0258(19981130)17:22%3C2635::AID-SIM954%3E3.0.CO;2-C](https://doi.org/10.1002/(SICI)1097-0258(19981130)17:22%3C2635::AID-SIM954%3E3.0.CO;2-C)
7. A. C. Cameron and D. L. Miller, “A Practitioner's Guide to Cluster-Robust Inference,” *Journal of Human Resources* 50(2), 317–372, 2015. [Open-access manuscript](https://escholarship.org/uc/item/1jq5d0pq)
8. A. C. Cameron, J. B. Gelbach, and D. L. Miller, “Bootstrap-Based Improvements for Inference with Clustered Errors,” *Review of Economics and Statistics* 90(3), 414–427, 2008. [doi:10.1162/rest.90.3.414](https://doi.org/10.1162/rest.90.3.414)
9. S. Holm, “A Simple Sequentially Rejective Multiple Test Procedure,” *Scandinavian Journal of Statistics* 6(2), 65–70, 1979. [JSTOR 4615733](https://www.jstor.org/stable/4615733)
10. Y. Benjamini and Y. Hochberg, “Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing,” *Journal of the Royal Statistical Society, Series B* 57(1), 289–300, 1995. [doi:10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)

### Inspect AI and AI-evaluation foundations

11. Inspect AI, “Log Dataframes.” [inspect.aisi.org.uk/dataframe.html](https://inspect.aisi.org.uk/dataframe.html)
12. Inspect AI, “Hooks” and “Early Stopping.” [Hooks](https://inspect.aisi.org.uk/extensions-hooks.html); [Early stopping](https://inspect.aisi.org.uk/early-stopping.html)
13. NIST, *Expanding the AI Evaluation Toolbox with Statistical Models*, NIST AI 800-3, February 2026. [Official PDF](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.800-3.pdf)
14. METR, “Details about METR's preliminary evaluation of OpenAI's o3 and o4-mini,” including a three-level hierarchical bootstrap over task families, tasks, and attempts. [Evaluation report](https://metr.org/evaluations/openai-o3-report/)

### AI safety and high-stakes AI precedents

15. Anthropic, “Sabotage evaluations for frontier models,” 2024. [Research report](https://www.anthropic.com/research/sabotage-evaluations)
16. C. Wang, I. Chaudhary, Q. Hu, W. Ruan, R. Gupta, and G. Singh, “How Catastrophic is Your LLM? Certifying Risks in Conversation” / C3LLM, ICLR 2026. [OpenReview paper](https://openreview.net/pdf?id=yt9TW2WtpG); [Amazon Science overview](https://www.amazon.science/blog/how-catastrophic-is-your-llm)
17. I. Chaudhary, Q. Hu, M. Kumar, M. Ziyadi, R. Gupta, and G. Singh, “Certifying Counterfactual Bias in LLMs” / LLMCert-B, ICLR 2025. [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1c6decac1477fbcc2cbf12d314ce0133-Abstract-Conference.html)
18. M. Nazeri, A.-K. Schmuck, S. Soudjani, and A. Abate, “Sound Probabilistic Safety Bounds for Large Language Models,” 2026. [arXiv:2607.20286](https://arxiv.org/abs/2607.20286)
19. “Position: LLM-Safety Evaluations Lack Robustness,” 2025. [OpenReview submission](https://openreview.net/forum?id=npJ7tn2QO8)
20. C.-Y. Hsu and S. Shekhar, “Efficient Sequential Evaluation of Large Language Models,” 2026. [arXiv:2607.17409](https://arxiv.org/abs/2607.17409)
21. T. D. Pilditch, “Knowing When to Stop: Bayesian Optimal Stopping for LLM Evaluations,” 2026. [arXiv:2608.14425](https://arxiv.org/abs/2608.14425); [Optstop repository](https://github.com/UKGovernmentBEIS/optstop)
22. NVIDIA NeMo Evaluator, “Comparing Evaluation Runs,” documenting paired exact McNemar regression decisions including an inconclusive outcome. [Documentation](https://docs.nvidia.com/nemo/evaluator/nightly/tutorials/compare)
23. Anthropic, “Measuring the Persuasiveness of Language Models,” including FDR correction for multiple comparisons, 2024. [Research report](https://www.anthropic.com/research/measuring-model-persuasiveness)
24. OpenAI, “Our updated Preparedness Framework,” 2025. [Framework overview](https://openai.com/index/updating-our-preparedness-framework/)
25. OpenAI Deployment Safety Hub, “Update to GPT-5 System Card: GPT-5.2—Preparedness Framework,” reporting bootstrap confidence intervals against Preparedness thresholds. [Deployment report](https://deploymentsafety.openai.com/gpt-5-2/preparedness-framework)
26. Anthropic, “Responsible Scaling Policy,” current and archived versions. [Policy page](https://www.anthropic.com/responsible-scaling-policy)
27. “General-purpose large language models outperform specialized clinical AI tools on medical benchmarks,” including exact McNemar tests and Holm–Bonferroni correction for safety/harm outcomes, 2026. [PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC13375586/)
28. R. R. Bellibatlu, “RISED: A Pre-Deployment Safety Evaluation Framework for Clinical AI Decision-Support Systems,” 2026 preprint. [arXiv:2605.12895](https://arxiv.org/abs/2605.12895)

### Fixed-design risk aggregation foundations

29. R. T. Rockafellar and S. Uryasev, “Optimization of Conditional Value-at-Risk,” *Journal of Risk* 2(3), 21–41, 2000. [doi:10.21314/JOR.2000.038](https://doi.org/10.21314/JOR.2000.038)
30. H. Föllmer and T. Knispel, “Entropic Risk Measures: Coherence vs. Convexity, Model Ambiguity, and Robust Large Deviations,” *Stochastics and Dynamics* 11(2–3), 333–351, 2011. [doi:10.1142/S0219493711003334](https://doi.org/10.1142/S0219493711003334)
31. C. Acerbi, “Spectral Measures of Risk: A Coherent Representation of Subjective Risk Aversion,” *Journal of Banking & Finance* 26(7), 1505–1518, 2002. [doi:10.1016/S0378-4266(02)00281-9](https://doi.org/10.1016/S0378-4266(02)00281-9)
32. W. Hoeffding, “Probability Inequalities for Sums of Bounded Random Variables,” *Journal of the American Statistical Association* 58(301), 13–30, 1963. [doi:10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830)
33. I. Waudby-Smith and A. Ramdas, “Estimating means of bounded random variables by betting,” *Journal of the Royal Statistical Society, Series B* 86(1), 1–27, 2024. [doi:10.1093/jrsssb/qkad009](https://doi.org/10.1093/jrsssb/qkad009)
34. F. Orabona and K.-S. Jun, “Tight Concentrations and Confidence Sequences from the Regret of Universal Portfolio,” *IEEE Transactions on Information Theory* 70(1), 436–455, 2024. [arXiv:2110.14099](https://arxiv.org/abs/2110.14099)

### Threats to validity and small-sample evaluation

33. T. van der Weij, F. Hofstätter, O. Jaffe, S. F. Brown, and F. R. Ward, “AI Sandbagging: Language Models can Strategically Underperform on Evaluations,” 2024. [arXiv:2406.07358](https://arxiv.org/abs/2406.07358)
34. C. Tice, P. A. Kreer, N. Helm-Burger, P. S. Shahani, F. Ryzhenkov, F. Roger, C. Neo, J. Haimes, F. Hofstätter, and T. van der Weij, “Noise Injection Reveals Hidden Capabilities of Sandbagging Language Models,” NeurIPS 2025. [arXiv:2412.01784](https://arxiv.org/abs/2412.01784)
35. N. Kalra and S. M. Paddock, “Driving to safety: How many miles of driving would it take to demonstrate autonomous vehicle reliability?,” *Transportation Research Part A: Policy and Practice* 94, 182–193, 2016. [doi:10.1016/j.tra.2016.09.010](https://doi.org/10.1016/j.tra.2016.09.010)
36. S. Bowyer, L. Aitchison, and D. R. Ivanova, “Position: Don't Use the CLT in LLM Evals With Fewer Than a Few Hundred Datapoints,” ICML 2025. [arXiv:2503.01747](https://arxiv.org/abs/2503.01747)

### Sequential evaluation and judge reliability

37. Z. Zhou, Z. Ye, Z. Chen, B. Li, and F. Liu, “CELEUS: Certifiable and Efficient LLM Evaluation via E-Processes,” 2026. [arXiv:2606.20820](https://arxiv.org/abs/2606.20820)
38. Y. Gao, “How Reliable Is Your Jailbreak Judge? Calibration and Adversarial Robustness of Automated ASR Scoring,” 2026. [arXiv:2606.25487](https://arxiv.org/abs/2606.25487)
39. I. Waudby-Smith and A. Ramdas, “Confidence sequences for sampling without replacement,” *Advances in Neural Information Processing Systems* 33, 2020. [arXiv:2006.04347](https://arxiv.org/abs/2006.04347)
40. J. Ville, *Étude critique de la notion de collectif*, Gauthier-Villars, Paris, 1939.

---

## 19. Summary of design commitments

The library should make the following commitments explicit:

1. A safety claim must name its estimand, population, independent unit, and threshold.
2. Fixed-sample exact bounds are used only when their binomial assumptions are defensible.
3. Zero observed failures are translated into an upper risk bound, never reported as zero risk.
4. The core release uses predeclared rollout counts and one fixed-horizon decision analysis; post hoc sample-size changes block an ordinary fixed-sample `PASS`.
5. Fixed prompts and groups are not treated as samples from an unseen prompt population, and aggregation uses predeclared policy weights.
6. Mean, maximum, CVaR, entropic risk, threshold-based, power-mean, and spectral outputs are treated as functionals of a finite declared design, not estimates of a prompt population.
7. Every risk aggregate records its hierarchy level, weights, tuning parameters, threshold, and decision role; no aggregate overrides mandatory or zero-tolerance hazards.
8. Comparisons preserve pairing and declared cluster structure without inventing random-population interpretations for fixed groups.
9. Multiplicity control follows the decision semantics: simultaneous or separately asserted claims are corrected, while a pure all-mandatory conjunction uses intersection-union logic.
10. Passing and failing use one-sided bounds with separately controlled errors.
11. Insufficient evidence produces `INCONCLUSIVE`.
12. Statistical evidence and policy overrides are reported separately.
13. Every result records assumptions, provenance, stopping, missingness, and validity warnings.
14. Anytime-valid confidence sequences and adaptive allocation remain a later extension, with data fields reserved for rollout order and allocation history.

These commitments make the package useful not merely as a collection of statistical functions, but as a disciplined interface between AI evaluation evidence and safety decisions.
