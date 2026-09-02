# Diagnostic Report (20260823_141623)
## Cross‑Fitting Audit
- Passed: None
- Message: AIPWEstimator does not expose fold assignment metadata; cannot conclusively verify cross‑fitting.
## Outcome Model Validation (out‑of‑sample)
- **none**: R²=0.148, MAE=0.400, RMSE=0.456
- **retry**: R²=0.161, MAE=0.405, RMSE=0.452
- **whatsapp**: R²=0.114, MAE=0.409, RMSE=0.465
## Pseudo‑Outcome Distribution (training)
- **none**: mean=0.3944, std=0.9071, min=-2.5574, max=3.6313, var=0.822880
- **retry**: mean=0.4288, std=0.6037, min=-0.5210, max=1.5625, var=0.364474
- **whatsapp**: mean=0.4462, std=1.4152, min=-7.9531, max=8.7249, var=2.002689
## True vs Estimated CATE (test set)
- **retry**: MAE=0.0698, RMSE=0.0981, Corr=-0.011
- **whatsapp**: MAE=0.1081, RMSE=0.1456, Corr=0.132
## Final‑Stage Learner Comparison (pseudo‑outcomes)
- RandomForest: MAE=0.4680, RMSE=1.0329
- GradientBoosting: MAE=0.4358, RMSE=1.0159
- HistGradientBoosting: MAE=0.4394, RMSE=1.0170
- **Best learner (by MAE)**: GradientBoosting
## Learning Curve Results
- 5000 events: ATE_MAE=17253.49, CATE_MAE=0.1535, CATE_RMSE=0.2071, AIPW PV=36658.41, Baseline PV=40022.93, Oracle PV=53911.90
- 10000 events: ATE_MAE=29685.98, CATE_MAE=0.1201, CATE_RMSE=0.1609, AIPW PV=76290.56, Baseline PV=76993.37, Oracle PV=105976.54
- 20000 events: ATE_MAE=145674.33, CATE_MAE=0.0873, CATE_RMSE=0.1223, AIPW PV=128382.05, Baseline PV=132508.26, Oracle PV=274056.38
- 50000 events: ATE_MAE=350610.43, CATE_MAE=0.0640, CATE_RMSE=0.0906, AIPW PV=396214.55, Baseline PV=345950.52, Oracle PV=746824.98
## Propensity Diagnostics (IPS)
- none: <5%=0.0%, >95%=0.0%, ESS=17267.5
- retry: <5%=0.0%, >95%=0.0%, ESS=19386.8
- whatsapp: <5%=0.0%, >95%=0.0%, ESS=19336.3
## Final Diagnosis
- Dominant issue: F (Based on high pseudo‑outcome variance, modest propensity overlap, and learner performance, the problem appears to be a combination of factors.)