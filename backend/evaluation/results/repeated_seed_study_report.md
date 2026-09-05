# Pre-specified repeated-seed study

Seeds 0–9; 3,000 events/seed; 70/30 train/held-out split; unchanged simulator, policy, costs, threshold, and model architecture. Conditional truth is Monte-Carlo integration over the simulator's hidden variables (512 draws).

## Policy value (realized synthetic holdout)

| policy | mean | median | std | min | max |
| --- | --- | --- | --- | --- | --- |
| causapay | 39388.36 | 36338.85 | 12588.01 | 20585.16 | 62765.26 |
| consent_safe_baseline | 47486.29 | 46316.61 | 19108.61 | 21903.37 | 75878.80 |
| naive_likelihood_matched_volume | 34895.07 | 32714.76 | 13961.27 | 19429.70 | 68232.05 |
| oracle | 77898.12 | 70158.49 | 20295.21 | 56383.40 | 111611.28 |

## Win rate versus matched-volume naive likelihood targeting

| policy | win_rate_vs_naive |
| --- | --- |
| causapay | 0.70 |
| consent_safe_baseline | 0.90 |
| oracle | 1.00 |

## Estimator diagnostics (mean across seeds)

| diagnostic | mae | bias | correlation | calibration_slope | rate | count |
| --- | --- | --- | --- | --- | --- | --- |
| abstention:combined |  |  |  |  | 0.1384 | 124.6000 |
| observable_cate_holdout:retry | 0.1609 | -0.0008 | 0.0106 | 0.0009 |  |  |
| observable_cate_holdout:whatsapp | 0.2234 | -0.0016 | 0.0504 | 0.0055 |  |  |
| outcome_nuisance_oof:none | 0.0871 | -0.0050 | 0.8922 | 0.8213 |  |  |
| outcome_nuisance_oof:retry | 0.0621 | 0.0001 | 0.9336 | 0.8526 |  |  |
| outcome_nuisance_oof:whatsapp | 0.1290 | 0.0036 | 0.7557 | 0.6419 |  |  |
| propensity_oof:none | 0.0326 | -0.0015 | 0.9262 | 0.8546 |  |  |
| propensity_oof:retry | 0.0381 | 0.0021 | 0.9184 | 0.8617 |  |  |
| propensity_oof:whatsapp | 0.0241 | -0.0006 | 0.8331 | 0.8829 |  |  |
