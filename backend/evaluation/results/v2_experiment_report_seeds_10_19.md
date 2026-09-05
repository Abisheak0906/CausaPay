# V2 Direct CATE Experiment Report (Seeds 10–19)

Pre-specified evaluation comparing V1 (AIPW pseudo-outcome regression) against V2 (cross-fitted pairwise R-learner) on untouched seeds 10–19.

## 1. Executive Summary & Promotion Decision

**Promotion Decision: `KEEP_V1`**

While V2 reduced MAE on both treatments and achieved a higher mean policy value on these seeds, it **failed Criteria 1 and 2** (correlation lift $\ge +0.10$ and minimum correlation $\ge 0.10$). Individual treatment-effect ranking correlation remained below 0.10 for both arms (retry: 0.0579, whatsapp: 0.0961). In accordance with pre-specified rules, V1 is retained as the production estimator and V2 remains an experimental comparator.

## 2. Pre-specified Promotion Criteria Evaluation

| Criterion | Target | Treatment / Arm | V1 Value | V2 Value | Difference / Ratio | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 1. Corr Lift | $\ge +0.10$ | Retry | 0.1087 | 0.0579 | -0.0508 | **FAILED** |
| 1. Corr Lift | $\ge +0.10$ | WhatsApp | 0.0246 | 0.0961 | +0.0715 | **FAILED** |
| 2. Min Corr | $\ge 0.10$ | Retry | — | 0.0579 | — | **FAILED** |
| 2. Min Corr | $\ge 0.10$ | WhatsApp | — | 0.0961 | — | **FAILED** |
| 3. Max MAE Worsening | $\le +0.02$ | Retry | 0.1522 | 0.1209 | -0.0313 | **PASSED** |
| 3. Max MAE Worsening | $\le +0.02$ | WhatsApp | 0.2142 | 0.1599 | -0.0543 | **PASSED** |
| 4. Policy Value Ratio | $\ge 0.95$ | Overall | ₹37,686.63 | ₹42,459.13 | 1.1266x | **PASSED** |
| 5. Naive Win Rate | $\ge$ V1 | Overall | 60% | 80% | — | **PASSED** |

## 3. Policy Performance Summary (Seeds 10–19)

| Policy | Mean (₹) | Median (₹) | Std Dev (₹) | Min (₹) | Max (₹) |
| --- | --- | --- | --- | --- | --- |
| causapay_v1 | ₹37,686.63 | ₹37,948.83 | ₹10,762.13 | ₹21,406.65 | ₹57,147.11 |
| causapay_v2 | ₹42,459.13 | ₹43,306.84 | ₹15,389.15 | ₹20,039.03 | ₹65,138.36 |
| consent_safe_baseline | ₹34,744.47 | ₹28,617.21 | ₹11,687.32 | ₹23,876.70 | ₹56,774.83 |
| naive_matched_to_v1 | ₹31,014.83 | ₹33,466.89 | ₹8,889.57 | ₹11,777.59 | ₹43,559.75 |
| naive_matched_to_v2 | ₹30,845.61 | ₹31,644.03 | ₹8,777.33 | ₹11,777.59 | ₹44,221.58 |
| oracle | ₹67,839.69 | ₹74,207.85 | ₹20,754.42 | ₹39,467.77 | ₹100,617.55 |

## 4. Estimator Diagnostics (Holdout CATE against Monte Carlo Conditional Truth)

| Estimator | Treatment | MAE (mean) | Corr (mean) | Corr (median) | Calib Slope (mean) |
| --- | --- | --- | --- | --- | --- |
| v1 | retry | 0.1522 | 0.1087 | 0.0818 | 0.0070 |
| v1 | whatsapp | 0.2142 | 0.0246 | 0.0302 | 0.0024 |
| v2 | retry | 0.1209 | 0.0579 | 0.0619 | 0.0050 |
| v2 | whatsapp | 0.1599 | 0.0961 | 0.0918 | 0.0156 |

