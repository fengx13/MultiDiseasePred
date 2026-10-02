# The published pipeline

Everything reported in the manuscript comes from the scripts in `pipeline/`.
This page is the map. The scripts themselves carry the detail, and the reasoning
behind each choice is written into their docstrings rather than left implicit.

If you are here to check what was actually done, this page and `pipeline/` are
the authoritative pair. The notebook under `notebook/` is a teaching example that
predates this pipeline and differs from it; the tutorial pages describe that
notebook, not this.

---

## Cohorts

`00_prepare_data.py` splits the University of Minnesota data by time: encounters
from 2019–2023 form the development cohort, encounters from 2024–2025 are held
out as the internal validation set. Within the development cohort a random 20%
(n = 262,687) is reserved for variable selection and for choosing the training
checkpoint, and is not used to fit the final model. The two external cohorts are
reserved whole.

The internal validation set is a time-based holdout, not a random split. It was
used for nothing except the final evaluation.

---

## Variable selection

`11_greedy_select.py` runs greedy forward selection over 62 candidate variables.

At each step it fits a gradient-boosted tree separately for each of the seven
acute conditions on a random 400,000-encounter subsample, and adds the single
remaining variable that most improves the macro-average AUROC over those seven
endpoints on the held-out selection split, given everything already chosen.

Two decisions inside that sentence are worth stating plainly:

**Greedy, not a prefix of a SHAP ranking.** Attribution answers "how much does
the model use this variable", one variable at a time. A prefix of that ordering
answers nothing in particular when candidates are correlated: two variables
carrying the same information both score highly and both get taken. Greedy
selection asks the marginal question — which variable most improves held-out
performance *given* what is already in — so a redundant variable falls to the
bottom on its own.

**The two disposition endpoints do not vote.** The selection criterion is the
macro-AUROC over the seven acute conditions only. `--targets all` exists and
produces a different ordering, written to a different file; the reported model
uses the acute-only ordering. The consequence is stated as a limitation in the
manuscript: the twelve-variable set is optimized for the acute conditions, and a
set chosen with disposition weighted equally would probably differ.

The stopping rule was fixed before the curve was examined: the smallest k
reaching 98% of the AUROC obtained with all 62 candidates. That returned k = 12.

**Triage acuity was excluded from the candidate set a priori** (`--drop
triage_acuity`), because the nurse's acuity score partly encodes the nurse's own
judgement of the endpoints being predicted. It appears in the manuscript only as
a variable used to describe the cohorts.

---

## The model

`models.py` and `01_train.py`. The twelve standardized variables are passed to a
shared pool of 13 feedforward expert subnetworks with hidden width 256. Each of
the nine targets carries its own gate over that pool:

```
p_task = 0.5 * softmax(S / T) + 0.5 * softmax(A / T)        # per-target
p      = p_task / p_task.sum()
```

with temperature `T = 0.2`. `S` and `A` are learned parameter matrices; the gate
does not depend on the input, so a target applies the same weighting of experts
to every encounter. The expert outputs are combined by each target's own weights
into a target-specific representation, which goes to that target's linear output
layer. All nine risks come out of one forward pass.

**The gate is dense.** Every expert receives non-zero weight for every target.
There is no top-k selection in the forward pass. What differs between targets is
the weighting, not which experts are used.

**No auxiliary regularization.** The loss is binary cross-entropy summed over the
nine targets. Three mechanisms were evaluated and not retained: an
input-dependent shared router mixed into the per-target gates together with a
symmetric Kullback–Leibler term tying the two, an L1 penalty on the gate logits,
and expert dropout. Each pulls the gate toward the uniform distribution; in the
arms that carried them the nine gates converged on almost the same weighting
(mean pairwise cosine 1.00, 12.9 of 13 experts in effective use). Removing them
left discrimination unchanged (acute macro-AUROC 0.868 in both cases) while the
mean pairwise cosine fell to 0.74. This is the arm `G2_no_l1_no_drop` in
`models.py`; all fourteen arms are compared in eTable 18 of the paper.

Training: Adam at 1e-3, batch size 256, gradient-norm clipping at 3.0, ten
epochs, `ReduceLROnPlateau` on mean validation AUROC. Every reported result is
the mean over three seeds, 42, 43 and 44.

---

## Calibration

Per-target isotonic regression on the logit scale, **fitted on the development
cohort and applied unchanged** to the internal and external validation sets.

This is the point where the notebook and the pipeline diverge most sharply. The
notebook splits the external dataset and fits the calibrator inside it. The
pipeline does not: the external results are reported without local retraining and
without recalibration, which is what makes them a transportability result rather
than a local-refit result.

Isotonic recalibration on the logit is monotone, so it cannot change AUROC or
AUPRC — except where a fitted slope comes out negative, which inverts the
ranking. That happens once in the site-adaptation analysis and is reported rather
than smoothed over.

---

## Evaluation

`14_metrics.py` and `10_ci.py`. Discrimination is AUROC and AUPRC, with 95%
intervals from 1,000 bootstrap resamples. Comparisons between models are
descriptive: a difference is called a difference only when the intervals do not
overlap.

Operating characteristics are computed at the highest threshold reaching 90%
sensitivity per target. Sensitivity rather than positive predictive value is the
quantity fixed, because prevalence is below 6% for every target except
hospitalization, so a threshold carries information mainly through its negative
predictive value.

Acute conditions and disposition endpoints are macro-averaged separately.
Predicting whether an acute condition is present depends on the patient;
disposition also depends on whether there is a bed. They are not the same kind of
quantity and are not pooled.

---

## Site adaptation

`21_finetune.py` asks what a new hospital would gain from local labels. Four arms
— zero shot, recalibrate slope and intercept, retrain the output heads, full
fine-tuning — at 100 to 5,000 local encounters.

All four are scored on the same fixed evaluation split: a held-out half of the
external cohort, drawn once before any fitting set is cut and disjoint from every
fitting set. No UMN data is pooled in. Because that is half the cohort, the
zero-shot numbers here differ slightly from the main external validation table,
which uses the complete cohort; they are not competing estimates of the same
quantity.

The fitting sets are nested subsets drawn once rather than resampled, so the
curves carry no sampling variability beyond the three model seeds. Sizes are
counted in encounters rather than events, which is what a hospital actually knows
about its own data; recalibration is skipped for any endpoint with fewer than
five events in the fitting set.
