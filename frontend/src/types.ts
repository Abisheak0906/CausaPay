export type Summary = {
  total_failed_payments: number;
  gross_recovered: number;
  incremental_recovered: number;
  intervention_cost: number;
  policy_value: number;
  action_distribution: Record<string, number>;
  dataset_id?: string;
  source?: string;
};

export type Policy = {
  policy_name: string;
  gross_recovered: number;
  /** Simulator ground truth — only valid for synthetic held-out evaluation */
  true_incremental_recovered: number;
  /** AIPW model estimate — used for uploaded-dataset evaluation (no ground truth available) */
  estimated_incremental_recovered?: number;
  intervention_cost: number;
  policy_value: number;
  recovery_rate: number;
  action_distribution: Record<string, number>;
};

export type EventItem = {
  event_id: string;
  customer_id?: string;
  amount: number;
  plan_tier?: string;
  payment_method?: string;
  failure_context?: string;
  decline_signal_bucket?: string;
  engagement_score?: number;
  historical_failure_count?: number;
  whatsapp_opted_in?: boolean;
  recommended_action?: string;
  baseline_action?: string;
  is_abstain?: boolean;
  prob_none?: number;
  prob_retry?: number;
  prob_whatsapp?: number;
  inc_prob_retry?: number;
  inc_prob_whatsapp?: number;
  source?: 'dynamic' | 'seeded_demo';
  payment_id?: string;
  failure_reason?: string;
  workflow?: {
    required: boolean;
    status: string;
    mode: string;
    workflow_id?: string | null;
    decision_id?: string | null;
    failure_reason?: string | null;
    last_updated?: string | null;
  };
};

export type Decision = {
  event_id: string;
  prob_none: number;
  prob_retry: number;
  prob_whatsapp: number;
  inc_prob_retry: number;
  inc_prob_whatsapp: number;
  eniv_none: number;
  eniv_retry: number;
  eniv_whatsapp: number;
  recommended_action: string;
  is_abstain: boolean;
  explanation: string;
  causal_preferred_action?: string | null;
  fallback_action?: string | null;
  uncertainty_retry?: number | null;
  uncertainty_whatsapp?: number | null;
  uncertainty_threshold?: number | null;
  whatsapp_eligible?: boolean | null;
  baseline_action?: string;
  workflow?: {
    required: boolean;
    status: string;
    mode: string;
    workflow_id?: string | null;
    decision_id?: string | null;
    failure_reason?: string | null;
    last_updated?: string | null;
  };
};

export type OutcomeMetrics = { r2: number; mae: number; rmse: number };
export type DistStats = {
  min: number; max: number; mean: number; std: number; var: number;
  p1: number; p5: number; p25: number; p50: number; p75: number; p95: number; p99: number;
};
export type CateBlock = {
  mae: number;
  rmse: number;
  corr: number;
  calibration: Record<string, { mean_true: number; mean_est: number }>;
};
export type Diagnostics = {
  cross_fitting_audit?: { passed: boolean | null; message: string };
  outcome_model_validation?: Record<string, OutcomeMetrics>;
  pseudo_outcome_distribution?: Record<string, DistStats>;
  true_vs_estimated_cate?: Record<string, CateBlock>;
  final_stage_learner_comparison?: Record<string, { mae: number; rmse: number }>;
  best_learner?: string;
  learning_curve?: Record<string, {
    ATE_MAE: number; CATE_MAE: number; CATE_RMSE: number;
    AIPW_policy_value: number; Baseline_policy_value: number; Oracle_policy_value: number;
  }>;
  propensity_diagnostics?: Array<{ treatment: string; 'pct_below_0.05': number; 'pct_above_0.95': number; ESS: number }>;
  final_diagnosis?: { dominant_issue: string; reasoning: string };
};

export type BatchPolicyMetrics = {
  gross_recovered: number;
  /** Simulator ground truth — only valid in synthetic held-out evaluation */
  true_incremental_recovered?: number;
  /** AIPW model estimate — used for uploaded-dataset evaluation */
  estimated_incremental_recovered?: number;
  intervention_cost: number;
  policy_value: number;
  recovery_rate: number;
  action_distribution: Record<string, number>;
};

export type BatchValidationMetric = {
  ground_truth_effect: number;
  aipw_estimate: number;
  absolute_error: number;
};

export type BatchEvaluation = {
  evaluation_type: 'synthetic_held_out' | 'uploaded_dataset';
  /** Present for uploaded_dataset mode */
  dataset_id?: string;
  dataset_filename?: string;
  batch_size: number;
  /** null in uploaded_dataset mode (no ground truth available) */
  baseline: BatchPolicyMetrics | null;
  causapay: BatchPolicyMetrics;
  /** null in uploaded_dataset mode */
  oracle: { policy_value: number } | null;
  /** null in uploaded_dataset mode */
  incremental_value_vs_baseline: number | null;
  /** null in uploaded_dataset mode (no simulator ground truth) */
  validation: {
    retry: BatchValidationMetric;
    whatsapp: BatchValidationMetric;
  } | null;
  naive_likelihood?: BatchPolicyMetrics & { description?: string };
};

export type ModelDiagnostics = {
  overlap: {
    groups: Array<{ treatment: string; assigned_count: number; all_rows: { min: number; p05: number; median: number; p95: number; max: number; mean: number }; histogram: Array<{ start: number; end: number; count: number }> }>;
    central_90_percent_common_interval: { lower: number; upper: number; has_common_support: boolean };
    interpretation: string;
    limitation: string;
  };
  uncertainty: {
    threshold: number; rows_above_threshold: number; rows_below_or_equal_threshold: number; abstention_proxy_rate: number; source: string; limitation: string;
    distributions: Record<string, { min: number; p05: number; median: number; p95: number; max: number; mean: number; histogram: Array<{ start: number; end: number; count: number }> }>;
  };
};

export type PageId = 'overview' | 'policy' | 'payment' | 'try-data' | 'upload' | 'cannibal' | 'diagnostics' | 'batch-evaluation';
