# CausaPay: Causal Payment Recovery Intelligence

CausaPay is a causal decision-making engine for failed-payment recovery. It shifts the recovery paradigm from predicting **"Who is likely to pay?"** to answering **"Which intervention is most likely to cause additional recovery?"**

## 1. The Problem

Traditional recovery systems target customers with the highest probability of payment. However, many of these customers would have paid anyway (self-cure). Applying costly interventions (retries, WhatsApp) to self-curers creates waste, unnecessary friction, and higher costs without increasing total recovery.

CausaPay focuses on **Incremental Recovery**: maximizing the difference between the outcome with an intervention and the outcome if nothing were done.

## 2. Decision Logic

The system follows a rigorous causal pipeline to determine the optimal action:

1. **Counterfactual Estimation**: Uses an Augmented Inverse Probability Weighting (AIPW) estimator to predict recovery probabilities for three scenarios: `None`, `Retry`, and `WhatsApp`.
2. **Incremental Lift**: Calculates the lift for each action: $\text{Lift}_a = P(\text{recovery} | a) - P(\text{recovery} | \text{none})$.
3. **Expected Net Incremental Value (ENIV)**: Incorporates payment amount and intervention costs:
   $$\text{ENIV}_a = (\text{Payment Amount} \times \text{Lift}_a) - \text{Cost}_a$$
4. **Constraint Enforcement**: Filters out actions that violate business rules (e.g., WhatsApp opt-out).
5. **Uncertainty Screening**: Checks the model's uncertainty (tree dispersion). If uncertainty is too high, the system **abstains** from the causal recommendation and falls back to a gross-recovery baseline.
6. **Final Action**: Selects the action with the highest positive ENIV, or the fallback if abstention occurred.

## 3. Architecture

- **Causal Engine**: V1 AIPW Estimator using double machine learning (cross-fitting) to estimate potential outcomes.
- **Policy Stack**:
  - **Incrementality-Aware**: Optimizes ENIV.
  - **Gross-Recovery Baseline**: Targets likely recoverers based on historical segment means.
  - **Matched-Volume Naive**: A non-causal comparator that targets likely recoverers but is constrained to the same intervention volume as CausaPay.
- **Infrastructure**: FastAPI backend, React frontend, SQLite for audit logs, and a synthetic simulator for ground-truth validation.

## 4. Evaluation Methodology

To ensure scientific honesty, CausaPay is evaluated using a **repeated-seed study** (Seeds 0–9):

- **Held-out Validation**: Models are trained on 70% of the data; results are measured on a 30% held-out set.
- **Ground Truth**: Recovery values are measured against Monte-Carlo integrated conditional potential outcomes.
- **Fair Comparison**: The naive comparator is volume-matched to CausaPay's actual intervention count for every seed to prevent "volume-inflation" bias.
- **Reproducibility**: All costs, thresholds, and seeds are pre-registered.

## 5. Results

### Policy Performance (Mean Value over Seeds 0–9)
| Policy | Mean Value (₹) | Win Rate vs Naive | Win Rate vs Baseline |
| :--- | :--- | :--- | :--- |
| **CausaPay V1** | **₹39,388** | **70%** | **30%** |
| Naive (Matched) | ₹34,895 | — | — |
| Gross Baseline | ₹47,486 | — | — |
| Oracle (Upper Bound) | ₹77,898 | — | — |

**Analysis**: CausaPay consistently outperforms simple likelihood targeting (Naive). However, it underperforms the gross-recovery baseline in this synthetic setting, suggesting that while it identifies incremental value, the cost of interventions sometimes outweighs the incremental gain compared to a "do nothing" or "gross-best" approach.

### Estimator Diagnostics
- **Propensity Estimation**: Strong. Overlap and positivity checks show sufficient support.
- **Individual CATE Ranking**: Low correlation with true individual treatment effects. The model is better at estimating *average* effects than *individual* lift.
- **Abstention Validation**: Empirical tests show a positive correlation between uncertainty scores and absolute CATE error, confirming the abstention heuristic effectively screens high-error predictions.

## 6. Diagnostics & Limitations

### Positivity & Overlap
The system exposes propensity overlap diagnostics. While observed support is generally good, overlap does not prove the absence of unmeasured confounding.

### Uncertainty Proxy
The uncertainty score is based on random-forest tree dispersion. **This is a heuristic proxy, not a calibrated confidence interval.** It is used to trigger abstention when predictions are unstable.

### Known Limitations
- **Synthetic Validation**: Results are based on simulation; real-world causal uplift requires randomized controlled trials (RCTs).
- **CATE Noise**: Individual-level treatment effect rankings are noisy.
- **In-Memory Data**: Uploaded CSV datasets are stored in memory and reset on backend restart.

## 7. What Real-World Validation Would Require

To move beyond a prototype, the following would be necessary:
1. **Randomized Experiments**: A production A/B test (Randomized Controlled Trial) to measure actual incremental lift.
2. **Observational Identification**: Stronger assumptions or instruments to rule out unmeasured confounding in real data.
3. **Calibration**: Monitoring the actual recovery rate of recommended actions vs predictions to calibrate the uncertainty threshold.
4. **Treatment Logging**: Rigorous logging of all "no-action" cases to build a true counterfactual holdout set.

## 8. Running the Project

### Backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## csv sample file
https://drive.google.com/drive/folders/1kaOu7YX1xKRhyIv3_hOIfl0b67blIl4H?usp=sharing
