# Scenario Report — MVP-B

## Overview

This report summarizes the first scenario-comparison runs performed with the MVP-B primary model for `target_dose_rate` prediction.

The objective of these runs was not to validate physical truth directly, but to test whether the current MVP-B package can be used as a **controlled scenario-comparison tool**. In practical terms, the question was:

> If contamination-related inputs are modified in a structured way, does the model respond in a stable, interpretable, and directionally meaningful manner?

The short answer is yes.

The current MVP-B primary model already behaves like a usable scenario-analysis instrument for comparative tabular runs.

---

## Model context

The scenario runs were produced with the MVP-B **primary model**:

* **Mode:** `nuclide_mode`
* **Feature set:** `env_plus_no_ratio`
* **Model family:** `extra_trees`

This means the working predictor uses:

* environment features,
* contamination features (`cs137_kBq_m2`, `sr90_kBq_m2`),
* natural radionuclide context (`k40_Bq_kg`, `ra226_Bq_kg`, `th232_Bq_kg`),
* and intentionally excludes `ratio_cs_sr` from the selected working configuration.

This choice is fully consistent with the earlier MVP-A and MVP-B findings: contamination-related features are the strongest stable source of predictive signal, and the no-ratio variant remains a strong and operationally clean candidate.

---

## Scenario design

A baseline nuclide-capable table was used as the scenario template. Controlled modifications were applied to contamination-related inputs while keeping the spatial support and environmental context fixed.

The following scenario set was evaluated:

1. **baseline**

   * no modification

2. **cs137_plus_20**

   * `cs137_kBq_m2 *= 1.20`

3. **cs137_minus_20**

   * `cs137_kBq_m2 *= 0.80`

4. **sr90_plus_20**

   * `sr90_kBq_m2 *= 1.20`

5. **remediation_light**

   * `cs137_kBq_m2 *= 0.70`
   * `sr90_kBq_m2 *= 0.85`

All scenario outputs were compared against the same baseline prediction table.

---

## Quantitative summary

### Baseline reference

* **Number of rows compared:** 545
* **Baseline mean prediction:** 0.10431
* **Baseline median prediction:** 0.10253

### Scenario summary table

| Scenario            | Scenario mean | Mean delta | Mean delta % | Delta min | Delta max | Positive rows | Negative rows |
| ------------------- | ------------: | ---------: | -----------: | --------: | --------: | ------------: | ------------: |
| `cs137_plus_20`     |       0.10568 |   +0.00137 |       +1.29% |  -0.00302 |  +0.02048 |           468 |            75 |
| `cs137_minus_20`    |       0.10316 |   -0.00115 |       -1.01% |  -0.02326 |  +0.00252 |            91 |           449 |
| `sr90_plus_20`      |       0.10452 |   +0.00021 |       +0.21% |  -0.00188 |  +0.00228 |           405 |           116 |
| `remediation_light` |       0.10244 |   -0.00187 |       -1.65% |  -0.02894 |  +0.00313 |            70 |           474 |

---

## Main findings

### 1. The model reacts in the expected direction

The scenario responses are directionally sensible:

* increasing `cs137` increases the average prediction,
* decreasing `cs137` lowers the average prediction,
* increasing `sr90` raises the prediction only slightly,
* remediation lowers the prediction most strongly.

This is an important milestone because it shows that MVP-B is already capable of **controlled comparative response**, not just one-off prediction.

---

### 2. `cs137` is the dominant scenario driver

The strongest result of the scenario runs is that **`cs137` clearly dominates `sr90` in practical scenario sensitivity**.

A +20% change in `cs137` produced an average increase of about **+1.29%** in the model output, while a +20% change in `sr90` produced only about **+0.21%**.

In other words, for the current primary model, the response to `cs137` perturbation is roughly several times stronger than the response to an equally sized `sr90` perturbation.

This strongly reinforces the earlier project conclusion that the contamination block is the main signal carrier and that `cs137` is the most influential single contamination-related predictor.

---

### 3. Remediation produces the strongest average reduction

Among the tested scenarios, `remediation_light` yields the strongest mean downward shift.

* mean delta: **-0.00187**
* mean relative change: **-1.65%**
* negative delta rows: **474 / 545**

This makes the remediation scenario the most practically impactful intervention within the current test pack.

---

### 4. Responses are spatially and locally heterogeneous

Even when the average effect is modest, local responses can be noticeably stronger.

Examples:

* `cs137_plus_20` reaches local increases up to about **+0.02048**
* `cs137_minus_20` reaches local decreases down to about **-0.02326**
* `remediation_light` reaches local decreases down to about **-0.02894**

This means the model is already useful not only for comparing average scenario outcomes, but also for identifying **locally sensitive points**.

---

### 5. The response is not strictly monotonic point-by-point

Although most rows move in the expected direction, the effect is not perfectly monotonic at every single point.

Examples:

* `cs137_plus_20` still contains some negative deltas,
* `cs137_minus_20` still contains some positive deltas,
* `remediation_light` also contains a small number of positive deltas.

This does **not automatically indicate an error**. It is more likely a consequence of the nonlinear interaction structure of the tree-based model. The current predictor should therefore be interpreted as a **nonlinear scenario-response model**, not as a simple direct formula where every row must move symmetrically and independently.

---

## Interpretation

At this stage, the scenario results support the following interpretation:

1. MVP-B is already suitable for **relative scenario comparison**.
2. The primary predictor is substantially more sensitive to `cs137` than to `sr90`.
3. Combined contamination reduction has the strongest downward effect among the tested scenarios.
4. Local scenario sensitivity varies by point, so the tool is useful for hotspot-style ranking in addition to average comparison.
5. The model should be used as an **engineering and comparative instrument**, not yet as a final physically calibrated simulator.

---

## Practical conclusions

The current MVP-B package can already support the following workflow:

* generate a baseline table,
* create controlled scenario variants,
* run prediction for each variant,
* compare outputs point-by-point and in aggregate,
* identify which scenario shifts are strongest overall,
* identify which points are most sensitive to a given scenario.

That is already a meaningful transition from “the model exists” to “the model can be used for scenario comparison.”

---

## Limitations

These scenario results should be interpreted with several constraints in mind:

1. **This is still a model-based comparison**, not a direct measurement-based physical validation.
2. **Out-of-domain scenario rows can occur**, especially when contamination values move beyond the original training range.
3. **The predictor is nonlinear**, so local counterintuitive row-level changes are possible.
4. **The tested scenario pack is intentionally simple**, designed for workflow validation rather than final scientific deployment.
5. **The current analysis is point-table based**, not yet a raster or map-based scenario system.

---

## What can already be said with confidence

The following statements are already justified by the current MVP-B scenario runs:

* MVP-B works as a controlled scenario-comparison pipeline.
* Scenario outputs are directionally sensible.
* `cs137` is the strongest operational scenario lever among the tested inputs.
* `sr90` matters, but its isolated effect is much weaker in the current model.
* Light remediation produces the strongest average reduction among the tested scenarios.
* The model captures both average effects and local hotspot-like sensitivity.

---

## Recommended next steps

### Immediate next step

Create a single aggregated scenario summary table across all tested runs and keep it as a compact monitoring artifact.

### Recommended analysis extension

Inspect the top changed rows for:

* `cs137_plus_20`
* `remediation_light`

and identify whether the same locations repeatedly appear among the most sensitive points.

### Recommended modeling extension

Add a second scenario pack with more physically plausible and policy-oriented perturbations, such as:

* mild contamination increase,
* stronger remediation,
* asymmetric `cs137` vs `sr90` shifts,
* bounded scenarios that stay fully inside training ranges.

### Recommended product extension

Move from point-table comparison toward a more map-ready workflow, while preserving the same schema-driven prediction contract.

---

## Final statement

**MVP-B has successfully reached the stage where the model can be used not only for prediction, but also for structured scenario comparison.**

The current scenario runs already show that the system responds coherently to contamination perturbations, with `cs137` acting as the strongest practical driver, `sr90` acting as a weaker secondary lever, and remediation scenarios producing the strongest mean reductions.

This is enough to treat MVP-B as the first operational baseline for comparative scenario analysis within the current project scope.
