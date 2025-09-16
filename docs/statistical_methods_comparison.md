# Statistical Methods Comparison: myImplementation vs official/SO

## Scope
This report compares statistical features implemented in `myImplementation/speechocean_experiments.py` against those computed in `official/SO/speechocean_quantification.py`, and identifies the closest counterparts.

## official/SO key metrics
- Max Logit: max over frames of target phoneme logits.
- Logit Variance: variance over frames of target logits (population).
- Mean Logit Margin: mean over frames of (target logit − best competing logit).
- Prosetrior Probability: mean softmax probability for the target phoneme; GOP score = −log(mean prob).
- Combined Score: `alpha * mean_logit_margin − (1 − alpha) * GOP_score`.

## myImplementation methods
- Moment-Based (Skewness, Kurtosis): `calculate_gop_moments(logits)`.
- Information-Theoretic: `calculate_gop_entropy(posteriors)`, `calculate_gop_kl_divergence(posteriors, target_id)`.
- GMM Features: `calculate_gop_gmm(logits, n_components)`.
- Temporal Smoothness: `calculate_gop_autocorrelation(logits, lag)`.
- Extreme-Value (Top-k): `calculate_gop_evt(logits, k)`.

## Closest correspondences
- official Max Logit ↔ myImpl Extreme-Value (Top-k): set `k = 1` to approximate exact Max Logit; `k > 1` provides a more robust peak estimate.
- official Logit Variance ↔ myImpl Moment-Based: variance is the 2nd moment; skewness/kurtosis extend this family (no explicit variance return, but same distribution-shape axis).
- official Mean Logit Margin ↔ myImpl KL/Entropy (closest by intent): margin measures target vs competitors per frame; KL/Entropy capture distribution concentration vs dispersion across all classes. Not numerically equivalent, but conceptually aligned (separation/confidence).
- official Prosetrior Probability / GOP score ↔ myImpl KL to one‑hot (closest): cross-entropy to a one‑hot target relates to −log p(target). Note: direct `D_KL(p || one_hot)` needs smoothing since zeros in the target distribution cause infinities.
- official Combined Score ↔ no direct myImpl counterpart: could be emulated by a weighted mix of a separation term (e.g., KL/Entropy) and a peak/shape term (e.g., EVT or Moments).
- myImpl Autocorrelation, GMM: no direct official counterparts; they add temporal structure and multi‑modal shape modeling.

## Notes
- Scale differences: margin is in logit units; entropy/KL are distributional and may require normalization for fair combination.
- Practical tip: if using KL to one‑hot, apply epsilon smoothing to the target or compute frame‑wise cross‑entropy `−log p(target)` as in official GOP score.

## Method Mapping Table (Pros/Cons)

| Official metric | Closest myImpl method | Why similar | Pros | Cons/Notes |
|---|---|---|---|---|
| Max Logit | EVT (Top‑k), k=1 | Both capture peak confidence in logits | Simple, robust for clear peaks | Sensitive to outliers; logit‑scale only |
| Logit Variance | Moments (skew/kurtosis) | Same family of distribution shape | Captures stability and asymmetry/outliers | Moments beyond variance can be noisy on short spans |
| Mean Logit Margin | Entropy / KL | All measure separation vs competitors | Distribution‑aware; frame‑wise separability | Not in logit units; normalize for mixing |
| Prosetrior Prob. → GOP(=−log p̄) | KL to one‑hot / cross‑entropy | Cross‑entropy to target equals −log p(target) | Interpretable; aligns with CE loss | KL to strict one‑hot needs smoothing |
| Combined Score (α·margin − (1−α)·GOP) | Mix: EVT/Moments + Entropy/KL | Weighted balance of peak/shape vs separation | Tunable to task | Requires α tuning; unit scaling matters |
| — | Autocorrelation | Temporal smoothness of logits | Sensitive to jitter vs stability | No direct official counterpart |
| — | GMM features | Multi‑modal logit dynamics | Captures phases (onset/steady) | Needs enough frames; model selection |

## Mini Numeric Examples

Assume target‑phoneme logits over 3 frames: `t = [1.0, 2.0, 0.5]`, competitor max logits per frame: `c = [0.8, 1.0, 0.7]`, and target posteriors per frame: `p = [0.6, 0.7, 0.2]`.

- Max Logit (official): `max(t) = 2.0`
- EVT Top‑k (myImpl):
  - `k=1 → 2.0` (matches Max Logit)
  - `k=2 → mean(top2)=mean(2.0,1.0)=1.5`
- Logit Variance (official, population): `var(t) ≈ 0.3889`
  - Moments (myImpl): skew/kurtosis computed on `t` (extend variance insight)
- Mean Logit Margin (official): `mean(t − c) = mean([0.2, 1.0, −0.2]) = 0.3333`
- Prosetrior Probability (official): `p̄ = mean(p) = 0.5`; GOP `= −ln(0.5) ≈ 0.6931`
- Entropy (myImpl, per‑frame; binary illustration):
  - `H([0.6,0.4]) ≈ 0.673`, `H([0.7,0.3]) ≈ 0.611`, `H([0.2,0.8]) ≈ 0.500`
  - Mean entropy (illustrative) `≈ 0.595` (higher ⇒ more uncertainty)
- KL to one‑hot target (myImpl, proxy): frame CE `= −ln p_target`; mean CE here `≈ 0.6931` (matches GOP on mean p).

Tip: when mixing metrics, z‑score or min‑max scale logit‑based and probability‑based terms before weighting.

## When To Use Which
- Need a quick confidence peak: use official Max Logit or myImpl EVT with `k=1` (robust: `k=3`).
- Suspect jitter/instability over time: check Logit Variance; add myImpl Autocorrelation (low → unstable).
- Want separation from competitors: use Mean Logit Margin; cross‑check with myImpl Entropy (lower is better).
- Align with loss‑style scoring: use GOP `= −log p̄`; myImpl mean cross‑entropy (KL to one‑hot) matches intent.
- Expect multi‑phase dynamics (onset/steady/offset): extract myImpl GMM features (means/variances/weights).
- Combine strength of both worlds: emulate Combined Score via weighted mix of Margin and GOP (or Entropy/EVT).

## Traceability (Functions & Anchors)
- official/SO/speechocean_quantification.py
  - Function: `align_phonemes_with_ctc_frames(...)`
  - Metrics dictionary keys: `"max_logit"`, `"logit_variance"`, `"mean_logit_margin"`, `"prosetrior_probability"`, `"combined_score"`.
  - Anchors to grep: `metrics = {`, `max_competitor`, `margins = target_logits - max_competitor`, `combined_score = (alpha * ... )`.
- myImplementation/speechocean_experiments.py
  - `calculate_gop_evt(logits, k)` → EVT / Max Logit analogue.
  - `calculate_gop_moments(logits)` → skewness, kurtosis (shape family with variance).
  - `calculate_gop_entropy(posteriors)` and `calculate_gop_kl_divergence(posteriors, target_id)` → dispersion/separation.
  - `calculate_gop_autocorrelation(logits, lag)` → temporal stability.
  - `calculate_gop_gmm(logits, n_components)` → multi‑modal dynamics features.

## Quick CLI Snippets
- Ripgrep (preferred):
  - Locate official metric calculations with context:
    - `rg -n -C 3 "combined_score|mean_logit_margin|max_competitor|metrics = \{" official/SO/speechocean_quantification.py`
  - Jump to alignment function:
    - `rg -n "^def align_phonemes_with_ctc_frames" official/SO/speechocean_quantification.py`
  - List myImplementation methods:
    - `rg -n "^def calculate_gop_(evt|moments|entropy|kl_divergence|autocorrelation|gmm)" myImplementation/speechocean_experiments.py`
- Fallback (grep):
  - `grep -n "combined_score\|mean_logit_margin\|max_competitor\|metrics = {" official/SO/speechocean_quantification.py`
  - `grep -n "^def align_phonemes_with_ctc_frames" official/SO/speechocean_quantification.py`
  - `grep -n "^def calculate_gop_" myImplementation/speechocean_experiments.py`
