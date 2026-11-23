import Mathlib

/-!
# Monotonicity of the package's risk measures

The package reports an interval for a risk measure `R` by pushing per-prompt bounds through it.
On the event that every prompt's interval covers its risk, `L ≤ p ≤ U`, it reports
`[R L, R U]`. That is a bound only because every measure is coordinatewise nondecreasing. This
file proves that for each measure in `src/safety_stats/stats/risk.py`.

Each definition follows the Python computation on real numbers. Where Python takes a different
route to the same value, such as the greedy CVaR or the partitioned top-`k` mean, a property test
in `tests/property/test_risk_properties.py` checks that the two agree, and the docstring names it.
Floating-point rounding is outside the scope of these proofs.
-/

noncomputable section

open Finset

namespace SafetyStats

variable {ι : Type*} [Fintype ι]

/-! ## Weighted mean -/

/-- `weighted_mean` in `stats/risk.py`. -/
def weightedMean (w x : ι → ℝ) : ℝ :=
  ∑ i, w i * x i

/-- Python counterpart: `test_measures_are_coordinatewise_monotone`. -/
theorem weightedMean_mono {w x y : ι → ℝ} (hw : ∀ i, 0 ≤ w i) (hxy : ∀ i, x i ≤ y i) :
    weightedMean w x ≤ weightedMean w y :=
  Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (hxy i) (hw i)

/-! ## Maximum -/

/-- `maximum_risk` in `stats/risk.py`: the largest value, whatever its weight. -/
def maximumRisk [Nonempty ι] (x : ι → ℝ) : ℝ :=
  univ.sup' univ_nonempty x

/-- Python counterpart: `test_measures_are_coordinatewise_monotone`. -/
theorem maximumRisk_mono [Nonempty ι] {x y : ι → ℝ} (hxy : ∀ i, x i ≤ y i) :
    maximumRisk x ≤ maximumRisk y :=
  Finset.sup'_le _ _ fun i _ => (hxy i).trans (Finset.le_sup' y (mem_univ i))

/-! ## Threshold exceedance and excess -/

/-- `threshold_exceedance` in `stats/risk.py`: the weight of the items strictly above their
threshold. A scalar threshold is a constant function. -/
def thresholdExceedance (w x τ : ι → ℝ) : ℝ :=
  ∑ i, w i * (if τ i < x i then 1 else 0)

/-- Python counterpart: `test_measures_are_coordinatewise_monotone`. -/
theorem thresholdExceedance_mono {w x y τ : ι → ℝ} (hw : ∀ i, 0 ≤ w i)
    (hxy : ∀ i, x i ≤ y i) : thresholdExceedance w x τ ≤ thresholdExceedance w y τ := by
  refine Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left ?_ (hw i)
  by_cases hx : τ i < x i
  · have hy : τ i < y i := hx.trans_le (hxy i)
    simp [hx, hy]
  · by_cases hy : τ i < y i <;> simp [hx, hy]

/-- `threshold_excess` in `stats/risk.py`: the weighted mean excess over the threshold. -/
def thresholdExcess (w x τ : ι → ℝ) : ℝ :=
  ∑ i, w i * max (x i - τ i) 0

/-- Python counterpart: `test_measures_are_coordinatewise_monotone`. -/
theorem thresholdExcess_mono {w x y τ : ι → ℝ} (hw : ∀ i, 0 ≤ w i) (hxy : ∀ i, x i ≤ y i) :
    thresholdExcess w x τ ≤ thresholdExcess w y τ :=
  Finset.sum_le_sum fun i _ =>
    mul_le_mul_of_nonneg_left (max_le_max (sub_le_sub_right (hxy i) (τ i)) le_rfl) (hw i)

/-! ## Entropic risk -/

/-- `entropic_risk` in `stats/risk.py`, `log (∑ w exp (η x)) / η`. Python shifts the exponent by
the maximum for stability, which gives the same value. For `η < 1e-7` it returns the weighted
mean, which `weightedMean_mono` covers. -/
def entropicRisk (w x : ι → ℝ) (η : ℝ) : ℝ :=
  Real.log (∑ i, w i * Real.exp (η * x i)) / η

/-- Nonnegative weights that sum to one include a positive weight. -/
theorem exists_pos_weight {w : ι → ℝ} (hw : ∀ i, 0 ≤ w i) (hsum : ∑ i, w i = 1) :
    ∃ j, 0 < w j := by
  by_contra h
  have : ∑ i, w i = 0 :=
    Finset.sum_eq_zero fun i _ => le_antisymm (not_lt.mp fun hi => h ⟨i, hi⟩) (hw i)
  linarith

/-- Python counterpart: `test_measures_are_coordinatewise_monotone`. -/
theorem entropicRisk_mono {w x y : ι → ℝ} {η : ℝ} (hη : 0 < η) (hw : ∀ i, 0 ≤ w i)
    (hsum : ∑ i, w i = 1) (hxy : ∀ i, x i ≤ y i) :
    entropicRisk w x η ≤ entropicRisk w y η := by
  obtain ⟨j, hj⟩ := exists_pos_weight hw hsum
  have hpos : 0 < ∑ i, w i * Real.exp (η * x i) :=
    Finset.sum_pos' (fun i _ => mul_nonneg (hw i) (Real.exp_pos _).le)
      ⟨j, mem_univ j, mul_pos hj (Real.exp_pos _)⟩
  have hle : ∑ i, w i * Real.exp (η * x i) ≤ ∑ i, w i * Real.exp (η * y i) :=
    Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left
      (Real.exp_le_exp.mpr (mul_le_mul_of_nonneg_left (hxy i) hη.le)) (hw i)
  unfold entropicRisk
  exact div_le_div_of_nonneg_right (Real.log_le_log hpos hle) hη.le

/-! ## Worst of `k` attempts -/

/-- `worst_of_k_risk` in `stats/risk.py`: the weighted probability that at least one of `k`
independent attempts fails. Python evaluates `1 - (1 - p) ^ k` as `-expm1 (k * log1p (-p))`,
which is the same real number. -/
def worstOfK (w x : ι → ℝ) (k : ℕ) : ℝ :=
  ∑ i, w i * (1 - (1 - x i) ^ k)

/-- The values are probabilities, so only the upper ends need a bound.
Python counterpart: `test_measures_are_coordinatewise_monotone`. -/
theorem worstOfK_mono {w x y : ι → ℝ} (k : ℕ) (hw : ∀ i, 0 ≤ w i) (hxy : ∀ i, x i ≤ y i)
    (hy : ∀ i, y i ≤ 1) : worstOfK w x k ≤ worstOfK w y k := by
  refine Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left ?_ (hw i)
  have h0 : 0 ≤ 1 - y i := by linarith [hy i]
  have hle : 1 - y i ≤ 1 - x i := by linarith [hxy i]
  have hpow : (1 - y i) ^ k ≤ (1 - x i) ^ k := pow_le_pow_left₀ h0 hle k
  linarith

/-- Bernoulli's inequality, for any `p ≤ 1`. -/
theorem one_sub_mul_le_pow {p : ℝ} (h1 : p ≤ 1) (n : ℕ) :
    1 - n * p ≤ (1 - p) ^ n := by
  induction n with
  | zero => simp
  | succ n ih =>
    have hq : 0 ≤ 1 - p := by linarith
    have step : (1 - n * p) * (1 - p) ≤ (1 - p) ^ n * (1 - p) :=
      mul_le_mul_of_nonneg_right ih hq
    have hn : (0 : ℝ) ≤ n := Nat.cast_nonneg n
    rw [pow_succ]
    push_cast
    nlinarith [mul_nonneg hn (sq_nonneg p)]

/-- One item of `worst_of_k_risk` lies between a single attempt and the union bound.
Python counterpart: `test_worst_of_k_item_is_sandwiched`. -/
theorem worstOfK_item_sandwich {p : ℝ} (h0 : 0 ≤ p) (h1 : p ≤ 1) {k : ℕ} (hk : 1 ≤ k) :
    p ≤ 1 - (1 - p) ^ k ∧ 1 - (1 - p) ^ k ≤ min 1 (k * p) := by
  have hq0 : 0 ≤ 1 - p := by linarith
  have hq1 : 1 - p ≤ 1 := by linarith
  refine ⟨?_, le_min ?_ ?_⟩
  · have : (1 - p) ^ k ≤ 1 - p := pow_le_of_le_one hq0 hq1 (by omega)
    linarith
  · have : 0 ≤ (1 - p) ^ k := pow_nonneg hq0 k
    linarith
  · have := one_sub_mul_le_pow h1 k
    linarith

/-! ## Conditional value at risk -/

/-- The Rockafellar–Uryasev objective, whose infimum over `t` is the CVaR at level `β`. -/
def cvarObjective (w x : ι → ℝ) (β t : ℝ) : ℝ :=
  t + (1 / (1 - β)) * ∑ i, w i * max (x i - t) 0

/-- CVaR at level `β` in Rockafellar–Uryasev form. `weighted_cvar` in `stats/risk.py` computes it
greedily, taking the worst `1 - β` of the weight. Python counterpart for that equality:
`test_weighted_cvar_equals_its_rockafellar_uryasev_form`. -/
def cvar (w x : ι → ℝ) (β : ℝ) : ℝ :=
  ⨅ t : ℝ, cvarObjective w x β t

/-- Every value of the objective is at least the weighted mean, so the infimum exists and CVaR
is never below the mean. Python counterpart:
`test_tail_measures_sit_between_the_mean_and_the_worst_item`. -/
theorem weightedMean_le_cvarObjective {w x : ι → ℝ} {β : ℝ} (hw : ∀ i, 0 ≤ w i)
    (hsum : ∑ i, w i = 1) (hβ1 : β < 1) (hβ0 : 0 ≤ β) (t : ℝ) :
    weightedMean w x ≤ cvarObjective w x β t := by
  have hpos : 0 < 1 - β := by linarith
  have hc : 1 ≤ 1 / (1 - β) := (one_le_div hpos).mpr (by linarith)
  have hA0 : 0 ≤ ∑ i, w i * max (x i - t) 0 :=
    Finset.sum_nonneg fun i _ => mul_nonneg (hw i) (le_max_right _ _)
  have hlin : ∑ i, w i * (x i - t) = weightedMean w x - t := by
    simp only [mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, hsum, one_mul, weightedMean]
  have hAt : weightedMean w x - t ≤ ∑ i, w i * max (x i - t) 0 := by
    rw [← hlin]
    exact Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (le_max_left _ _) (hw i)
  unfold cvarObjective
  nlinarith [mul_nonneg (sub_nonneg.mpr hc) hA0]

/-- Python counterpart: `test_measures_are_coordinatewise_monotone`. -/
theorem cvar_mono {w x y : ι → ℝ} {β : ℝ} (hw : ∀ i, 0 ≤ w i) (hsum : ∑ i, w i = 1)
    (hβ0 : 0 ≤ β) (hβ1 : β < 1) (hxy : ∀ i, x i ≤ y i) : cvar w x β ≤ cvar w y β := by
  have hc : 0 ≤ 1 / (1 - β) := (one_div_pos.mpr (by linarith)).le
  unfold cvar
  apply ciInf_mono
  · exact ⟨weightedMean w x, by
      rintro _ ⟨t, rfl⟩
      exact weightedMean_le_cvarObjective hw hsum hβ1 hβ0 t⟩
  · intro t
    unfold cvarObjective
    have hle : ∑ i, w i * max (x i - t) 0 ≤ ∑ i, w i * max (y i - t) 0 :=
      Finset.sum_le_sum fun i _ =>
        mul_le_mul_of_nonneg_left (max_le_max (sub_le_sub_right (hxy i) t) le_rfl) (hw i)
    have := mul_le_mul_of_nonneg_left hle hc
    linarith

/-! ## Top-`k` mean -/

/-- `top_k_mean` in `stats/risk.py`: the unweighted mean of the `k` largest values, which is the
CVaR at `β = 1 - k / n` with equal weights. Python counterpart for that equality:
`test_top_k_mean_is_cvar_with_equal_weights`. -/
def topKMean (x : ι → ℝ) (k : ℕ) : ℝ :=
  cvar (fun _ => 1 / (Fintype.card ι : ℝ)) x (1 - k / Fintype.card ι)

/-- Python counterpart: `test_measures_are_coordinatewise_monotone`. -/
theorem topKMean_mono [Nonempty ι] {x y : ι → ℝ} {k : ℕ} (hk : 1 ≤ k)
    (hkn : k ≤ Fintype.card ι) (hxy : ∀ i, x i ≤ y i) : topKMean x k ≤ topKMean y k := by
  have hn : (0 : ℝ) < Fintype.card ι := by exact_mod_cast Fintype.card_pos
  have hsum : ∑ _i : ι, 1 / (Fintype.card ι : ℝ) = 1 := by
    rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    exact mul_one_div_cancel hn.ne'
  unfold topKMean
  refine cvar_mono (fun _ => div_nonneg zero_le_one hn.le) hsum ?_ ?_ hxy
  · have : (k : ℝ) / Fintype.card ι ≤ 1 := by
      rw [div_le_one hn]
      exact_mod_cast hkn
    linarith
  · have : 0 < (k : ℝ) / Fintype.card ι := div_pos (Nat.cast_pos.mpr (by omega)) hn
    linarith

end SafetyStats
