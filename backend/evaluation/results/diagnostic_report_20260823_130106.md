# Diagnostic Report (20260823_130106)
## Cross‑Fitting Audit
- Passed: None
- Message: AIPWEstimator does not expose fold assignment metadata; cannot conclusively verify cross‑fitting.
## Outcome Model Validation (out‑of‑sample)
- **none**: R²=0.148, MAE=0.400, RMSE=0.456
- **retry**: R²=0.161, MAE=0.405, RMSE=0.452
- **whatsapp**: R²=0.114, MAE=0.409, RMSE=0.465
## Pseudo‑Outcome Distribution (training)
- **none**: mean=0.3948, std=0.8986, min=-2.5574, max=3.5688, var=0.807456
- **retry**: mean=0.4287, std=0.5985, min=-0.5210, max=1.5625, var=0.358144
- **whatsapp**: mean=0.4501, std=1.5772, min=-9.5620, max=10.3740, var=2.487512
## True vs Estimated CATE (test set)
- **retry**: MAE=0.0696, RMSE=0.0996, Corr=-0.006
- **whatsapp**: MAE=0.1229, RMSE=0.1699, Corr=0.149
## Final‑Stage Learner Comparison (pseudo‑outcomes)
- RandomForest: MAE=0.4829, RMSE=1.1106
- GradientBoosting: MAE=0.4491, RMSE=1.0914
- HistGradientBoosting: MAE=0.4503, RMSE=1.0919
- **Best learner (by MAE)**: GradientBoosting
## Learning Curve Results
- 5000 events: ATE_MAE=15968.96, CATE_MAE=0.1595, CATE_RMSE=0.2130, AIPW PV=37942.94, Baseline PV=40022.93, Oracle PV=53911.90
- 10000 events: ATE_MAE=29087.17, CATE_MAE=0.1277, CATE_RMSE=0.1705, AIPW PV=76889.37, Baseline PV=76993.37, Oracle PV=105976.54
- 20000 events: ATE_MAE=136604.69, CATE_MAE=0.0929, CATE_RMSE=0.1324, AIPW PV=137451.69, Baseline PV=132508.26, Oracle PV=274056.38
- 50000 events: ATE_MAE=357318.02, CATE_MAE=0.0670, CATE_RMSE=0.0960, AIPW PV=389506.96, Baseline PV=345950.52, Oracle PV=746824.98
## Propensity Diagnostics (IPS)
- none: <5%=0.0%, >95%=0.0%, ESS=16678.7
- retry: <5%=0.0%, >95%=0.0%, ESS=19360.7
- whatsapp: <5%=29.9%, >95%=0.0%, ESS=16972.0
## Final Diagnosis
- Dominant issue: F (Based on high pseudo‑outcome variance, modest propensity overlap, and learner performance, the problem appears to be a combination of factors.)