# MultiDiseasePred

One model for many outcomes that can happen at the same time. MultiDiseasePred
trains a single network over a shared pool of experts, gives each outcome its own
weighting of that pool, and returns a calibrated probability for every outcome in
one pass. It then writes a dashboard for *your* outcomes.

!!! note "Where to start"
    **[Using the package](quickstart.md)** — install `multidiseasepred`, fit it on
    your own outcomes, and generate a dashboard for them.
    **[The published pipeline](pipeline.md)** — the scripts and settings behind
    every number in the manuscript, kept so the results can be checked.

## What is in the repository

```
multidiseasepred/    the package — this is the thing to use
examples/            build_ed_dashboard.py builds the ED demo using the package
pipeline/            supplementary: the exact scripts behind the manuscript
tests/               pytest suite
docs/                this site
```

## The published application

MultiDiseasePred was developed on 2.2 million emergency department encounters
across three health systems, predicting nine outcomes from twelve variables
recorded at triage, and externally validated at two independent sites without
local retraining or recalibration.

## Data

No clinical data ship with this repository. MIMIC-IV-ED and MC-MED are available
through PhysioNet under their credentialing and data use agreements. The
University of Minnesota data cannot be shared.

## Citation

Please cite the associated manuscript once it is available.
