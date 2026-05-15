# MR-ART JDAC sanity check — 2026-05-15

Objective: verify whether JDAC works when applied to MR-ART images prepared according to the preprocessing described in the JDAC paper.

## Data

Dataset: MR-ART / OpenNeuro `ds004173`

Subjects used:
- sub-000103
- sub-000148
- sub-000149
- sub-000159
- sub-000175
- sub-862915

Conditions:
- standard: motion-free reference
- headmotion1: minor motion
- headmotion2: stronger motion

## Preprocessing

The article describes minimal preprocessing:
- skull stripping
- intensity normalization to [0,1]

Local preprocessed images are stored outside the Git repo:

`~/Documents/Datasets/MRART_jdac_ready/`

The NIfTI images are not tracked in Git.

## JDAC inference

JDAC was run with a modified inference script:

`jdac_infer_no_internal_preproc.py`

This disables the internal preprocessing step because the inputs are already skull-stripped and normalized.

Outputs were stored locally:

`~/Documents/Datasets/MRART_jdac_ready_outputs_all6/`

The NIfTI outputs are not tracked in Git.

## Results

Tracked CSV files:
- `mrart_jdac_all6_detailed_metrics.csv`
- `mrart_jdac_all6_summary_metrics.csv`
- `mrart_jdac_ready_all6.csv`
- `jdac_log_all6.csv`

Summary: JDAC improves all average metrics after article-consistent preprocessing.
