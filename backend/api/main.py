import hashlib
import io
import json
import glob
import os
import csv
from datetime import datetime
from uuid import uuid4

import numpy as np
import pandas as pd
import redis
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

# Adjust imports to work when run from backend module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import (
    CausaPayDecision,
    PaymentEvent,
    PaymentRecoveryState,
    RecoveryTransitionLog,
    SessionLocal,
    WorkflowExecution,
    get_event_decision,
    init_db,
    json_safe,
    load_audit_payload,
    load_json_payload,
)
from services.workflow_dispatcher import dispatch_workflow, workflow_status_snapshot
from simulator.generator import SimulatorConfig, generate_events
from simulator.bias import assign_historical_treatments
from causal_engine.aipw_estimator import AIPWEstimator
from causal_engine.estimator import TLearner  # kept for benchmark comparison only
from policies.baseline import GrossRecoveryBaseline
from policies.incrementality import IncrementalityAwarePolicy
from policies.oracle import OraclePolicy
from evaluation.engine import CausaPayEvaluationEngine, build_demo_evaluation
from evaluation.evaluator import evaluate_policy

app = FastAPI(title="Razorpay CausaPay Demo API")
redis_client = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://causa-pay-olive.vercel.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global objects – initialized once on server start
# ---------------------------------------------------------------------------

aipw_estimator: AIPWEstimator = None
baseline_policy: GrossRecoveryBaseline = None
inc_policy: IncrementalityAwarePolicy = None
oracle_policy: OraclePolicy = None
train_obs: pd.DataFrame = None
test_obs: pd.DataFrame = None
test_hid: pd.DataFrame = None
events_df: pd.DataFrame = None  # stripped version for API exposure
benchmark_results: list = []

# Populated when a CSV is uploaded via /api/upload-dataset.
# Consumed by /evaluation/run to evaluate the uploaded batch instead of the
# synthetic held-out set.  Cleared when a new upload replaces the previous one.
latest_uploaded_dataset: dict | None = None

costs = {"none": 0.0, "retry": 2.0, "whatsapp": 15.0}
EXTERNAL_REQUIRED_FIELDS = [
    "event_id", "customer_id", "amount", "plan_tier", "payment_method",
    "failure_context", "decline_signal_bucket", "engagement_score",
    "historical_failure_count", "whatsapp_opted_in",
]
EXTERNAL_OPTIONAL_FIELDS = [
    "email_verified", "days_since_last_failure", "historical_payment_count",
    "day_of_month", "tenure_days", "baseline_action", "intervention_assigned",
    "outcome_recovered", "is_abstain", "recommended_action",
]
EXTERNAL_ALLOWED_ACTIONS = ["none", "retry", "whatsapp"]
OBSERVABLE_SCHEMA = {
    "required": [
        "event_id",
        "customer_id",
        "amount",
        "plan_tier",
        "payment_method",
        "failure_context",
        "decline_signal_bucket",
        "engagement_score",
        "historical_failure_count",
        "whatsapp_opted_in",
    ],
    "optional": [
        "email_verified",
        "days_since_last_failure",
        "historical_payment_count",
        "day_of_month",
        "tenure_days",
        "baseline_action",
        "intervention_assigned",
        "outcome_recovered",
        "is_abstain",
        "recommended_action",
    ],
    "categorical": ["plan_tier", "payment_method", "failure_context", "decline_signal_bucket"],
    "numerical": [
        "amount",
        "engagement_score",
        "historical_failure_count",
        "days_since_last_failure",
        "historical_payment_count",
        "day_of_month",
        "tenure_days",
    ],
    "boolean": ["email_verified", "whatsapp_opted_in"],
}
OBSERVABLE_DEFAULTS = {
    "email_verified": False,
    "days_since_last_failure": 365,
    "historical_payment_count": 0,
    "day_of_month": 1,
    "tenure_days": 0,
    "payment_method": "unknown",
    "plan_tier": "unknown",
    "failure_context": "unknown",
    "decline_signal_bucket": "unknown",
    "customer_id": "unknown",
    "baseline_action": "none",
}

RECOVERY_CONFIG = {
    "max_total_interventions": 3,
    "max_retry_attempts": 2,
    "max_whatsapp_attempts": 1,
    "retry_cooldown_minutes": 30,
    "whatsapp_cooldown_minutes": 60,
}

RECOVERY_STATES = {
    "NEW",
    "DECIDED",
    "RETRY_SCHEDULED",
    "WHATSAPP_SCHEDULED",
    "WAITING_FOR_OUTCOME",
    "RECOVERED",
    "STOPPED",
    "ABSTAINED",
    "EXHAUSTED",
}


def _stable_hash(value: object) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _connect_redis():
    global redis_client
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        redis_client = None
        app.state.redis_status = {"enabled": False, "connected": False, "fallback_mode": True}
        return None
    try:
        client = redis.Redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=1, socket_timeout=1)
        client.ping()
        redis_client = client
        app.state.redis_status = {"enabled": True, "connected": True, "fallback_mode": False}
        return client
    except Exception:
        redis_client = None
        app.state.redis_status = {"enabled": True, "connected": False, "fallback_mode": True}
        return None


def _redis_get_json(key: str):
    if redis_client is None:
        return None
    try:
        raw = redis_client.get(key)
        if raw in (None, ""):
            return None
        return json.loads(raw)
    except Exception:
        return None


def _redis_set_json(key: str, payload: object, ttl_seconds: int = 3600):
    if redis_client is None:
        return False
    try:
        redis_client.set(key, json.dumps(payload, default=str), ex=ttl_seconds)
        return True
    except Exception:
        return False


def _redis_lock(key: str, ttl_seconds: int = 15) -> str | None:
    if redis_client is None:
        return None
    token = uuid4().hex
    try:
        acquired = redis_client.set(key, token, nx=True, ex=ttl_seconds)
        if acquired:
            return token
    except Exception:
        return None
    return None


def _redis_unlock(key: str, token: str | None):
    if redis_client is None or not token:
        return
    try:
        current = redis_client.get(key)
        if current == token:
            redis_client.delete(key)
    except Exception:
        pass


def _decision_idempotency_key(req: "DecisionRequest") -> str:
    payload = {"event_id": str(req.event_id), **{key: _normalize_json_value(value) for key, value in req.dict().items() if key != "event_id"}}
    return f"decision:{_stable_hash(payload)}"


def _next_action_idempotency_key(event_id: str, state: PaymentRecoveryState | None) -> str:
    action_count = 0
    if state is not None:
        action_count = len(_load_json_list(state.action_history))
    payload = {
        "event_id": str(event_id),
        "state": state.current_state if state is not None else "NEW",
        "outcome_status": state.outcome_status if state is not None else "PENDING",
        "action_count": action_count,
    }
    return f"next_action:{_stable_hash(payload)}"


def _outcome_idempotency_key(event_id: str, status: str, source: str | None) -> str:
    payload = {"event_id": str(event_id), "status": str(status).upper(), "source": str(source or "manual")}
    return f"outcome:{_stable_hash(payload)}"


def _normalize_recovery_action(action: str | None) -> str | None:
    if action is None:
        return None
    return str(action).strip().upper()


def _load_json_list(raw: str | None) -> list[dict]:
    if raw in (None, ""):
        return []
    try:
        loaded = json.loads(raw)
        if isinstance(loaded, list):
            return [item for item in loaded if isinstance(item, dict)]
    except (TypeError, ValueError):
        pass
    return []


def _ensure_recovery_state(session: SessionLocal, event_id: str) -> PaymentRecoveryState:
    state = session.query(PaymentRecoveryState).filter(PaymentRecoveryState.event_id == str(event_id)).first()
    if state is not None:
        return state
    state = PaymentRecoveryState(
        event_id=str(event_id),
        current_state="NEW",
        outcome_status="PENDING",
        action_history=json_safe([]),
        state_history=json_safe([]),
        last_action=None,
        last_updated=datetime.utcnow(),
    )
    session.add(state)
    try:
        session.flush()
    except IntegrityError:
        session.rollback()
        state = session.query(PaymentRecoveryState).filter(PaymentRecoveryState.event_id == str(event_id)).first()
        if state is None:
            raise
        return state
    return state


def _append_recovery_action_history(session: SessionLocal, event_id: str, action: str, status: str, decision_id: str | None = None, reason: str | None = None, outcome: str | None = None):
    state = _ensure_recovery_state(session, event_id)
    history = _load_json_list(state.action_history)
    entry = {
        "action": _normalize_recovery_action(action) or "NONE",
        "timestamp": datetime.utcnow().isoformat(),
        "status": status,
        "decision_id": decision_id,
        "reason": reason,
        "outcome": outcome,
    }
    history.append(entry)
    state.action_history = json_safe(history)
    state.last_action = entry["action"]
    state.last_updated = datetime.utcnow()
    session.add(state)
    return entry


def _append_recovery_transition(session: SessionLocal, event_id: str, previous_state: str | None, new_state: str, triggering_action: str | None, reason: str | None, details: dict | None = None):
    state = _ensure_recovery_state(session, event_id)
    previous = state.current_state if previous_state is None else previous_state
    history = _load_json_list(state.state_history)
    transition = {
        "previous_state": previous,
        "new_state": new_state,
        "triggering_action": _normalize_recovery_action(triggering_action),
        "reason": reason,
        "timestamp": datetime.utcnow().isoformat(),
        "details": details or {},
    }
    history.append(transition)
    state.state_history = json_safe(history)
    state.current_state = new_state
    state.last_updated = datetime.utcnow()
    if triggering_action:
        state.last_action = _normalize_recovery_action(triggering_action)
    session.add(state)
    session.flush()

    log = RecoveryTransitionLog(
        transition_id=str(uuid4()),
        event_id=str(event_id),
        previous_state=previous,
        new_state=new_state,
        triggering_action=_normalize_recovery_action(triggering_action),
        reason=reason,
        timestamp=datetime.utcnow(),
        details=json_safe(details or {}),
    )
    session.add(log)
    return transition


def _event_input_payload(event: PaymentEvent) -> dict:
    payload = load_json_payload(event.input_payload)
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("event_id", str(event.event_id))
    payload.setdefault("customer_id", event.customer_id or "unknown")
    payload.setdefault("amount", float(event.amount) if event.amount is not None else 0.0)
    payload.setdefault("whatsapp_opted_in", bool(event.whatsapp_opted_in))
    payload.setdefault("failure_context", event.failure_reason or "unknown")
    return payload


def _recovery_state_for_response(session: SessionLocal, event_id: str) -> dict:
    state = session.query(PaymentRecoveryState).filter(PaymentRecoveryState.event_id == str(event_id)).first()
    if state is None:
        return {
            "event_id": str(event_id),
            "current_state": "NEW",
            "outcome_status": "PENDING",
            "action_history": [],
            "state_history": [],
        }
    return {
        "event_id": str(event_id),
        "current_state": state.current_state,
        "outcome_status": state.outcome_status,
        "action_history": _load_json_list(state.action_history),
        "state_history": _load_json_list(state.state_history),
    }


def _strip_hidden(df: pd.DataFrame) -> pd.DataFrame:
    """Remove any hidden ground‑truth columns before JSON serialisation.
    Columns that start with 'true_' or 'Y_' are considered hidden.
    """
    hidden = [c for c in df.columns if c.startswith("true_") or c.startswith("Y_")]
    return df.drop(columns=hidden, errors="ignore")


def _normalize_json_value(value):
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    except ValueError:
        pass
    return value


def _normalize_record(record):
    return {key: _normalize_json_value(value) for key, value in record.items()}


def _python_bool(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _db_decision_payload(decision: CausaPayDecision) -> dict:
    payload = {
        "event_id": str(decision.event_id),
        "prob_none": float(decision.probability_none),
        "prob_retry": float(decision.probability_retry),
        "prob_whatsapp": float(decision.probability_whatsapp),
        "inc_prob_retry": float(decision.incremental_retry),
        "inc_prob_whatsapp": float(decision.incremental_whatsapp),
        "eniv_none": 0.0,
        "eniv_retry": float(decision.eniv_retry),
        "eniv_whatsapp": float(decision.eniv_whatsapp),
        "recommended_action": decision.recommended_action,
        "baseline_action": decision.baseline_action,
        "uncertainty_retry": float(decision.uncertainty_retry) if decision.uncertainty_retry is not None else None,
        "uncertainty_whatsapp": float(decision.uncertainty_whatsapp) if decision.uncertainty_whatsapp is not None else None,
        "abstained": bool(decision.abstained),
        "is_abstain": bool(decision.abstained),
        "explanation": decision.decision_reason or f"Decision persisted for {str(decision.event_id)}.",
        "decision_id": decision.decision_id,
        "decision_reason": decision.decision_reason,
        "model_version": decision.model_version,
        "created_at": decision.created_at.isoformat() if decision.created_at else None,
    }
    audit = load_audit_payload(decision.audit_payload)
    if audit:
        payload["audit"] = audit
    return payload


def _build_decision_response(row: pd.Series) -> dict:
    """Given a row (already containing probabilities and stds),
    compute incremental probabilities, ENIV, and recommendation.
    This mirrors IncrementalityAwarePolicy logic for a single observation.
    """
    p_none = row["prob_none"]
    p_retry = row["prob_retry"]
    p_wa = row["prob_whatsapp"]
    std_retry = row.get("std_retry", np.nan)
    std_wa = row.get("std_whatsapp", np.nan)
    amount = row["amount"]

    inc_prob_retry = p_retry - p_none
    inc_prob_wa = p_wa - p_none
    eniv_none = 0.0
    eniv_retry = inc_prob_retry * amount - costs["retry"]
    eniv_wa = inc_prob_wa * amount - costs["whatsapp"]

    # WhatsApp eligibility — mirrors IncrementalityAwarePolicy.predict() behaviour
    wa_eligible = bool(row.get("whatsapp_opted_in", True))

    # uncertainty abstention
    causal_preferred = "none"
    candidate_wa = eniv_wa if wa_eligible else float("-inf")
    best_eniv = max(eniv_none, eniv_retry, candidate_wa)
    if best_eniv > 0:
        causal_preferred = "retry" if best_eniv == eniv_retry else "whatsapp"

    fallback_action = None
    if std_retry > inc_policy.uncertainty_threshold or std_wa > inc_policy.uncertainty_threshold:
        recommended = baseline_policy.predict(pd.DataFrame([row]))[0]
        # Respect opt-in even in the abstain/baseline fallback path
        if not wa_eligible and recommended == "whatsapp":
            recommended = "none"
        is_abstain = True
        fallback_action = recommended
    else:
        is_abstain = False
        # Exclude WhatsApp when customer has not opted in
        if best_eniv <= 0:
            recommended = "none"
        elif best_eniv == eniv_retry:
            recommended = "retry"
        else:
            recommended = "whatsapp"

    if is_abstain:
        explanation = "Model abstained because tree-dispersion uncertainty exceeded the configured threshold; the displayed action is the baseline fallback."
    elif recommended == "none":
        explanation = "No eligible intervention had positive expected net incremental value."
    else:
        explanation = f"{recommended.capitalize()} has the highest positive eligible expected net incremental value."

    return {
        "event_id": str(row["event_id"]),
        "prob_none": float(p_none),
        "prob_retry": float(p_retry),
        "prob_whatsapp": float(p_wa),
        "inc_prob_retry": float(inc_prob_retry),
        "inc_prob_whatsapp": float(inc_prob_wa),
        "eniv_none": float(eniv_none),
        "eniv_retry": float(eniv_retry),
        "eniv_whatsapp": float(eniv_wa),
        "recommended_action": recommended,
        "is_abstain": is_abstain,
        "causal_preferred_action": causal_preferred,
        "fallback_action": fallback_action,
        "uncertainty_retry": float(std_retry) if pd.notna(std_retry) else None,
        "uncertainty_whatsapp": float(std_wa) if pd.notna(std_wa) else None,
        "uncertainty_threshold": float(inc_policy.uncertainty_threshold),
        "whatsapp_eligible": bool(wa_eligible),
        "explanation": explanation,
    }


def _event_payload(row: pd.Series) -> dict:
    """Return the observable fields needed to browse a real simulated event."""
    fields = [
        "event_id", "customer_id", "amount", "plan_tier", "payment_method",
        "failure_context", "decline_signal_bucket", "engagement_score",
        "historical_failure_count", "whatsapp_opted_in", "recommended_action",
        "baseline_action", "is_abstain",
    ]
    payload = {field: row[field] for field in fields if field in row.index}
    if "event_id" in payload:
        payload["event_id"] = str(payload["event_id"])
    return {
        key: (value.item() if isinstance(value, np.generic) else value)
        for key, value in payload.items()
    }


def _event_payload_from_dict(record: dict, source: str = "seeded_demo") -> dict:
    fields = [
        "event_id", "customer_id", "amount", "plan_tier", "payment_method",
        "failure_context", "decline_signal_bucket", "engagement_score",
        "historical_failure_count", "whatsapp_opted_in", "recommended_action",
        "baseline_action", "is_abstain",
    ]
    payload = {field: record.get(field) for field in fields if field in record}
    if "event_id" in payload:
        payload["event_id"] = str(payload["event_id"])
    payload["source"] = source
    return {
        key: _normalize_json_value(value)
        for key, value in payload.items()
    }


def _dynamic_input_metadata(payload: dict) -> dict:
    supplied_fields = sorted({key for key in payload if key != "event_id"})
    defaulted_fields: list[str] = []
    for field in OBSERVABLE_SCHEMA["optional"]:
        if (field not in payload or payload.get(field) is None or payload.get(field) == "") and field in OBSERVABLE_DEFAULTS:
            defaulted_fields.append(field)

    validation_warnings: list[str] = []
    if "whatsapp_opted_in" in payload and not _python_bool(payload.get("whatsapp_opted_in", True)):
        validation_warnings.append("WhatsApp intervention is disabled because whatsapp_opted_in is false.")
    if "plan_tier" in payload:
        plan = str(payload.get("plan_tier", "")).lower()
        if plan not in {"basic", "pro", "enterprise", "starter", "business", "unknown"}:
            validation_warnings.append("plan_tier value is not in the known demo schema; the existing model preprocessing will keep it as-is.")

    return {
        "source": "dynamic",
        "supplied_fields": supplied_fields,
        "defaulted_fields": sorted(set(defaulted_fields)),
        "validation_warnings": validation_warnings,
    }


def _build_dynamic_event_response(event: PaymentEvent, decision: CausaPayDecision | None = None) -> dict:
    payload = load_json_payload(event.input_payload)
    if not isinstance(payload, dict):
        payload = {}
    event_id = str(event.event_id)
    payload["event_id"] = event_id
    payload["customer_id"] = event.customer_id or payload.get("customer_id", "unknown")
    payload["amount"] = event.amount if event.amount is not None else payload.get("amount", 0.0)
    payload["whatsapp_opted_in"] = event.whatsapp_opted_in if event.whatsapp_opted_in is not None else payload.get("whatsapp_opted_in", False)
    payload["failure_reason"] = event.failure_reason or payload.get("failure_reason", "unknown")

    response = {
        "event_id": event_id,
        "source": "dynamic",
        "customer_id": payload.get("customer_id"),
        "amount": _normalize_json_value(payload.get("amount", 0.0)),
        "plan_tier": payload.get("plan_tier", "unknown"),
        "payment_method": payload.get("payment_method", "unknown"),
        "failure_context": payload.get("failure_context", "unknown"),
        "decline_signal_bucket": payload.get("decline_signal_bucket", "unknown"),
        "engagement_score": payload.get("engagement_score", 0.0),
        "historical_failure_count": payload.get("historical_failure_count", 0),
        "whatsapp_opted_in": bool(payload.get("whatsapp_opted_in", False)),
        "input_metadata": _dynamic_input_metadata(payload),
    }

    if decision is not None:
        decision_payload = _db_decision_payload(decision)
        response.update({
            "prob_none": decision_payload["prob_none"],
            "prob_retry": decision_payload["prob_retry"],
            "prob_whatsapp": decision_payload["prob_whatsapp"],
            "inc_prob_retry": decision_payload["inc_prob_retry"],
            "inc_prob_whatsapp": decision_payload["inc_prob_whatsapp"],
            "eniv_none": decision_payload["eniv_none"],
            "eniv_retry": decision_payload["eniv_retry"],
            "eniv_whatsapp": decision_payload["eniv_whatsapp"],
            "recommended_action": decision_payload["recommended_action"],
            "baseline_action": decision_payload["baseline_action"],
            "is_abstain": decision_payload["is_abstain"],
            "whatsapp_opted_in": bool(payload.get("whatsapp_opted_in", event.whatsapp_opted_in or False)),
            "explanation": decision_payload["explanation"],
            "decision_id": decision_payload.get("decision_id"),
            "decision_reason": decision_payload.get("decision_reason"),
            "model_version": decision_payload.get("model_version"),
            "created_at": decision_payload.get("created_at"),
        })

    return response


def _validate_and_default_observable_input(data: dict) -> tuple[dict, dict]:
    supplied_fields = [key for key in data.keys() if key not in {"event_id", "amount"}]
    input_data = dict(data)
    defaulted_fields: list[str] = []
    validation_warnings: list[str] = []

    for field in OBSERVABLE_SCHEMA["required"]:
        if field not in input_data or input_data[field] is None or input_data[field] == "":
            if field == "event_id":
                continue
            raise HTTPException(status_code=422, detail=f"Missing required observable field: {field}")

    for field in OBSERVABLE_SCHEMA["optional"]:
        if field not in input_data or input_data[field] is None or input_data[field] == "":
            if field in OBSERVABLE_DEFAULTS:
                input_data[field] = OBSERVABLE_DEFAULTS[field]
                defaulted_fields.append(field)

    for field in OBSERVABLE_SCHEMA["boolean"]:
        if field in input_data and input_data[field] is not None:
            input_data[field] = _python_bool(input_data[field])

    for field in ["amount", "engagement_score", "historical_failure_count", "days_since_last_failure", "historical_payment_count", "day_of_month", "tenure_days"]:
        if field in input_data and input_data[field] is not None:
            try:
                input_data[field] = float(input_data[field])
            except (TypeError, ValueError):
                raise HTTPException(status_code=422, detail=f"Field '{field}' must be numeric.")

    if "amount" in input_data and float(input_data["amount"]) <= 0:
        raise HTTPException(status_code=422, detail="Field 'amount' must be greater than zero.")

    if "whatsapp_opted_in" in input_data and input_data["whatsapp_opted_in"] is False:
        validation_warnings.append("WhatsApp intervention is disabled because whatsapp_opted_in is false.")

    if "plan_tier" in input_data and str(input_data["plan_tier"]).lower() not in {"basic", "pro", "enterprise", "starter", "business", "unknown"}:
        validation_warnings.append("plan_tier value is not in the known demo schema; the existing model preprocessing will keep it as-is.")

    input_metadata = {
        "source": "dynamic",
        "supplied_fields": sorted(set(input_data.keys()) - {"event_id"}),
        "defaulted_fields": sorted(set(defaulted_fields)),
        "validation_warnings": validation_warnings,
    }
    return input_data, input_metadata


@app.on_event("startup")
async def load_models_and_data():
    global aipw_estimator, baseline_policy, inc_policy, oracle_policy
    global train_obs, test_obs, test_hid, events_df, benchmark_results

    init_db()
    _connect_redis()

    engine = CausaPayEvaluationEngine()
    obs_biased, hid = build_demo_evaluation(seed=42, num_events=15000)
    engine.fit_from_batch(obs_biased, hid, seed=42)

    train_obs = engine.train_obs
    test_obs = engine.test_obs
    test_hid = engine.test_hid
    events_df = engine.events_df
    benchmark_results = engine.benchmark_results
    aipw_estimator = engine.aipw_estimator
    baseline_policy = engine.baseline_policy
    inc_policy = engine.incrementality_policy
    oracle_policy = engine.oracle_policy

    app.state.full_events = engine.full_events
    app.state.events_lookup = {str(row["event_id"]): row for _, row in events_df.iterrows()}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class DecisionRequest(BaseModel):
    event_id: str = Field(..., description="Unique identifier for the failed payment")
    amount: float
    class Config:
        extra = "allow"

class DecisionResponse(BaseModel):
    event_id: str
    prob_none: float
    prob_retry: float
    prob_whatsapp: float
    inc_prob_retry: float
    inc_prob_whatsapp: float
    eniv_none: float
    eniv_retry: float
    eniv_whatsapp: float
    recommended_action: str
    is_abstain: bool
    explanation: str
    causal_preferred_action: str | None = None
    fallback_action: str | None = None
    uncertainty_retry: float | None = None
    uncertainty_whatsapp: float | None = None
    uncertainty_threshold: float | None = None
    whatsapp_eligible: bool | None = None
    input_metadata: dict | None = None

class SummaryResponse(BaseModel):
    total_failed_payments: int
    gross_recovered: float
    incremental_recovered: float
    intervention_cost: float
    policy_value: float
    action_distribution: dict

class PolicyResult(BaseModel):
    policy_name: str
    gross_recovered: float
    true_incremental_recovered: float
    intervention_cost: float
    policy_value: float
    recovery_rate: float
    action_distribution: dict


class EventListResponse(BaseModel):
    events: list[dict]
    total: int


class ExternalEvaluationRequest(BaseModel):
    events: list[dict] = Field(..., description="Observable payment failure events in the CausaPay schema")
    hidden: list[dict] | None = Field(default=None, description="Optional hidden ground-truth rows for simulator-only benchmark metrics")


@app.post("/api/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Process an uploaded CSV through the existing observable decision path."""
    global latest_uploaded_dataset
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Please upload a CSV file.")

    raw = await file.read()
    if len(raw) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="CSV file must be 5 MB or smaller.")
    try:
        text_content = raw.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text_content))
        if not reader.fieldnames:
            raise HTTPException(status_code=422, detail="CSV must include a header row.")
        header_map = {str(name).strip().lower(): name for name in reader.fieldnames if name}
        required = [field for field in EXTERNAL_REQUIRED_FIELDS if field not in {"event_id"}]
        missing = [field for field in required if field.lower() not in header_map]
        if missing:
            raise HTTPException(status_code=422, detail=f"Missing required CSV columns: {missing}")
        rows = list(reader)
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="CSV must be UTF-8 encoded.") from exc
    except csv.Error as exc:
        raise HTTPException(status_code=422, detail=f"Malformed CSV: {exc}") from exc

    dataset_id = f"upload_{uuid4().hex}"
    results = []
    errors = []
    counts = {"retry": 0, "whatsapp": 0, "none": 0, "abstained": 0}
    duplicate_rows = 0
    incremental_recovery = 0.0
    gross_recovery = 0.0
    eniv = 0.0
    intervention_cost = 0.0

    for row_number, raw_row in enumerate(rows, start=2):
        normalized = {
            key.strip().lower(): value.strip() if isinstance(value, str) else value
            for key, value in raw_row.items()
            if key
        }
        data = {field: normalized.get(field.lower()) for field in EXTERNAL_OPTIONAL_FIELDS + EXTERNAL_REQUIRED_FIELDS if field.lower() in normalized}
        data["event_id"] = data.get("event_id") or f"{dataset_id}_row_{row_number}"
        data["payment_id"] = data.get("payment_id") or f"pay_{data['event_id']}"
        try:
            with SessionLocal() as session:
                if session.query(PaymentEvent).filter(PaymentEvent.event_id == str(data["event_id"])).first() is not None:
                    duplicate_rows += 1
            decision = make_decision(DecisionRequest(**data))
            action = str(decision.recommended_action).lower()
            if decision.is_abstain:
                counts["abstained"] += 1
            else:
                counts[action] = counts.get(action, 0) + 1
            amount = float(data["amount"])
            incremental_recovery += max(float(decision.inc_prob_retry if action == "retry" else decision.inc_prob_whatsapp if action == "whatsapp" else 0.0), 0.0) * amount
            gross_recovery += float(decision.prob_retry if action == "retry" else decision.prob_whatsapp if action == "whatsapp" else decision.prob_none) * amount
            eniv += float(decision.eniv_retry if action == "retry" else decision.eniv_whatsapp if action == "whatsapp" else 0.0)
            intervention_cost += 2.0 if action == "retry" else 15.0 if action == "whatsapp" else 0.0
            results.append({"row": row_number, "event": {**data, "event_id": str(data["event_id"])}, "decision": decision.dict()})
        except HTTPException as exc:
            errors.append({"row": row_number, "error": str(exc.detail)})
        except (ValueError, TypeError, KeyError) as exc:
            errors.append({"row": row_number, "error": str(exc)})

    # Store the processed results so /evaluation/run can evaluate this dataset
    # instead of the synthetic preset when an upload is active.
    latest_uploaded_dataset = {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "processed_rows": len(results),
        "results": results,
        "summary": {
            **counts,
            "incremental_recovery": incremental_recovery,
            "gross_recovery": gross_recovery,
            "eniv": eniv,
            "intervention_cost": intervention_cost,
        },
    }

    return {
        "status": "success",
        "dataset_id": dataset_id,
        "total_rows": len(rows),
        "processed_rows": len(results),
        "duplicate_rows": duplicate_rows,
        "failed_rows": len(errors),
        "summary": {
            **counts,
            "incremental_recovery": incremental_recovery,
            "gross_recovery": gross_recovery,
            "eniv": eniv,
            "intervention_cost": intervention_cost,
        },
        "results": results,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/evaluate/schema")
def get_evaluation_schema():
    """Return a structured schema and example payload for the /api/evaluate endpoint.

    Fields:
    - description: short human-friendly explanation
    - required_fields: list of field names that must be present on each event
    - optional_fields: list of optional observable fields
    - example: a realistic example payload containing at least 2 events
    """
    example_events = [
        {
            "event_id": "evt_1001",
            "customer_id": 200312,
            "amount": 1499.00,
            "plan_tier": "Business",
            "payment_method": "Card",
            "failure_context": "insufficient_funds",
            "decline_signal_bucket": "soft",
            "engagement_score": 0.72,
            "historical_failure_count": 1,
            "whatsapp_opted_in": True,
            "email_verified": True,
            "days_since_last_failure": 4,
            "historical_payment_count": 12,
            "day_of_month": 14,
            "tenure_days": 240
        },
        {
            "event_id": "evt_1002",
            "customer_id": 198745,
            "amount": 299.50,
            "plan_tier": "Starter",
            "payment_method": "UPI Autopay",
            "failure_context": "card_expired",
            "decline_signal_bucket": "hard",
            "engagement_score": 0.33,
            "historical_failure_count": 3,
            "whatsapp_opted_in": False,
            "email_verified": False,
            "days_since_last_failure": 30,
            "historical_payment_count": 4,
            "day_of_month": 2,
            "tenure_days": 85
        },
    ]

    return {
        "description": "Observable failed-payment events expected by POST /api/evaluate. Provide an 'events' array of objects using the listed required fields. Optionally include 'hidden' with ground-truth outcomes for benchmark mode (simulator-only).",
        "required_fields": EXTERNAL_REQUIRED_FIELDS,
        "optional_fields": EXTERNAL_OPTIONAL_FIELDS,
        "example": {"events": example_events, "hidden": None},
    }


@app.post("/api/evaluate")
def evaluate_external_batch(payload: dict):
    if isinstance(payload, list):
        raw_events = payload
        raw_hidden = None
    elif isinstance(payload, dict):
        raw_events = payload.get("events")
        raw_hidden = payload.get("hidden")
    else:
        raise HTTPException(status_code=422, detail="Expected a list of events or an object with an 'events' array.")

    if raw_events is None or not isinstance(raw_events, list) or len(raw_events) == 0:
        raise HTTPException(status_code=422, detail="The evaluation payload requires a non-empty 'events' array.")

    events_df = pd.DataFrame(raw_events).where(pd.notnull(pd.DataFrame(raw_events)), None)
    missing = [field for field in EXTERNAL_REQUIRED_FIELDS if field not in events_df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required observable fields: {missing}")

    if raw_hidden is not None:
        hidden_df = pd.DataFrame(raw_hidden)
        if hidden_df.empty:
            raise HTTPException(status_code=422, detail="The optional 'hidden' array cannot be empty when supplied.")
        engine = CausaPayEvaluationEngine()
        result = engine.fit_from_batch(events_df, hidden_df, seed=42)
        rows = [_normalize_record(row) for row in result["events_df"].to_dict(orient="records")]
        for row in rows:
            row["event_id"] = str(row.get("event_id", ""))
        action_counts = {key: int(value) for key, value in pd.Series(result["events_df"]["recommended_action"]).value_counts().to_dict().items()}
        abstentions = int(result["events_df"]["is_abstain"].sum())
        waiver_violations = int(
            ((~result["events_df"]["whatsapp_opted_in"].fillna(True).astype(bool)) & (result["events_df"]["recommended_action"].astype(str) == "whatsapp")).sum()
        )
        invalid = int((~result["events_df"]["recommended_action"].isin(EXTERNAL_ALLOWED_ACTIONS).fillna(False).to_numpy()).sum())
        benchmark = {
            "baseline": result["benchmark_results"][0],
            "incrementality_aware": result["benchmark_results"][1],
            "oracle": result["benchmark_results"][2],
        }
        return {
            "mode": "hidden-benchmark",
            "batch_size": len(rows),
            "recommended_action_distribution": action_counts,
            "abstention_count": abstentions,
            "abstention_rate": round(abstentions / len(rows), 4) if rows else 0.0,
            "wa_opt_in_violation_count": waiver_violations,
            "invalid_recommendations": invalid,
            "results": rows,
            "aggregate_policy_metrics": benchmark,
        }

    if aipw_estimator is None or baseline_policy is None or inc_policy is None:
        raise HTTPException(status_code=503, detail="No trained demo model is available for external observable-only evaluation.")

    engine = CausaPayEvaluationEngine()
    engine.aipw_estimator = aipw_estimator
    engine.baseline_policy = baseline_policy
    engine.incrementality_policy = inc_policy
    engine.oracle_policy = oracle_policy
    result = engine.evaluate_observable_batch(events_df)
    rows = [_normalize_record(row) for row in result["events_df"].to_dict(orient="records")]
    for row in rows:
        row["event_id"] = str(row.get("event_id", ""))
    action_counts = {key: int(value) for key, value in pd.Series(result["events_df"]["recommended_action"]).value_counts().to_dict().items()}
    abstentions = int(result["events_df"]["is_abstain"].sum())
    waiver_violations = int(
        ((~result["events_df"]["whatsapp_opted_in"].fillna(True).astype(bool)) & (result["events_df"]["recommended_action"].astype(str) == "whatsapp")).sum()
    )
    invalid = int((~result["events_df"]["recommended_action"].isin(EXTERNAL_ALLOWED_ACTIONS).fillna(False).to_numpy()).sum())
    return {
        "mode": "observable-only",
        "batch_size": len(rows),
        "recommended_action_distribution": action_counts,
        "abstention_count": abstentions,
        "abstention_rate": round(abstentions / len(rows), 4) if rows else 0.0,
        "wa_opt_in_violation_count": waiver_violations,
        "invalid_recommendations": invalid,
        "results": rows,
        "aggregate_policy_metrics": None,
    }


@app.post("/evaluation/run")
def run_judge_evaluation():
    """Run batch evaluation.

    If a CSV was uploaded via /api/upload-dataset, evaluate that dataset and
    return structured results for the uploaded batch.  Otherwise run the
    reproducible synthetic held-out evaluation for the judge dashboard.
    """
    if latest_uploaded_dataset is not None:
        # ---------------------------------------------------------------
        # Uploaded-dataset path
        # ---------------------------------------------------------------
        ds = latest_uploaded_dataset
        results = ds["results"]
        summary = ds["summary"]

        action_distribution = {
            "retry": int(summary.get("retry", 0)),
            "whatsapp": int(summary.get("whatsapp", 0)),
            "none": int(summary.get("none", 0)),
            "abstained": int(summary.get("abstained", 0)),
        }
        action_distribution["acted_upon"] = (
            action_distribution["retry"] + action_distribution["whatsapp"]
        )

        gross_recovered = float(summary.get("gross_recovery", 0.0))
        incremental_recovered = float(summary.get("incremental_recovery", 0.0))
        intervention_cost = float(summary.get("intervention_cost", 0.0))
        policy_value = float(summary.get("eniv", 0.0))
        total_amount = sum(
            float(r["event"].get("amount", 0) or 0) for r in results
        )
        recovery_rate = gross_recovered / total_amount if total_amount > 0 else 0.0

        return {
            "evaluation_type": "uploaded_dataset",
            "dataset_id": ds["dataset_id"],
            "dataset_filename": ds["filename"],
            "batch_size": ds["processed_rows"],
            "causapay": {
                "gross_recovered": gross_recovered,
                # For uploaded datasets there is no simulator ground truth.
                # This is an AIPW model estimate of incremental recovery,
                # NOT a realized simulator outcome.  Use the honest field name.
                "estimated_incremental_recovered": incremental_recovered,
                "intervention_cost": intervention_cost,
                "policy_value": policy_value,
                "recovery_rate": recovery_rate,
                "action_distribution": action_distribution,
            },
            # No baseline or oracle comparison available for observable-only
            # uploads — ground-truth outcomes are not present in user CSVs.
            "baseline": None,
            "oracle": None,
            "incremental_value_vs_baseline": None,
            "validation": None,
        }

    # ---------------------------------------------------------------
    # Synthetic held-out path (original behaviour, unchanged)
    # ---------------------------------------------------------------
    evaluation_engine = CausaPayEvaluationEngine()
    observables, hidden = build_demo_evaluation(seed=42, num_events=15000)
    result = evaluation_engine.fit_from_batch(observables, hidden, seed=42)

    baseline = result["benchmark_results"][0]
    naive = result["benchmark_results"][1]
    causapay = result["benchmark_results"][2]
    oracle = result["benchmark_results"][3]
    test_hidden = result["test_hid"]
    counterfactuals = evaluation_engine.aipw_estimator.predict_counterfactuals(result["test_obs"])

    validation = {}
    for treatment in ("retry", "whatsapp"):
        ground_truth = float((test_hidden[f"prob_{treatment}"] - test_hidden["prob_none"]).mean())
        aipw_estimate = float((counterfactuals[f"prob_{treatment}"] - counterfactuals["prob_none"]).mean())
        validation[treatment] = {
            "ground_truth_effect": ground_truth,
            "aipw_estimate": aipw_estimate,
            "absolute_error": abs(ground_truth - aipw_estimate),
        }

    events_df = result["events_df"]
    causapay_distribution = {
        "retry": int(
            ((~events_df["is_abstain"]) & (events_df["recommended_action"] == "retry")).sum()
        ),
        "whatsapp": int(
            ((~events_df["is_abstain"]) & (events_df["recommended_action"] == "whatsapp")).sum()
        ),
        "none": int(
            ((~events_df["is_abstain"]) & (events_df["recommended_action"] == "none")).sum()
        ),
        "abstained": int(events_df["is_abstain"].sum()),
    }
    causapay_distribution["acted_upon"] = (
        causapay_distribution["retry"] + causapay_distribution["whatsapp"]
    )

    return {
        "evaluation_type": "synthetic_held_out",
        "batch_size": len(result["test_obs"]),
        "baseline": {
            "gross_recovered": baseline["gross_recovered"],
            "intervention_cost": baseline["intervention_cost"],
            "policy_value": baseline["policy_value"],
            "recovery_rate": baseline["recovery_rate"],
            "action_distribution": baseline["action_distribution"],
        },
        "naive_likelihood": {
            "gross_recovered": naive["gross_recovered"],
            "true_incremental_recovered": naive["true_incremental_recovered"],
            "intervention_cost": naive["intervention_cost"],
            "policy_value": naive["policy_value"],
            "recovery_rate": naive["recovery_rate"],
            "action_distribution": naive["action_distribution"],
            "description": "Non-causal likelihood ranking, evaluated on the same held-out rows and matched to CausaPay's intervention volume.",
        },
        "causapay": {
            "gross_recovered": causapay["gross_recovered"],
            "true_incremental_recovered": causapay["true_incremental_recovered"],
            "intervention_cost": causapay["intervention_cost"],
            "policy_value": causapay["policy_value"],
            "recovery_rate": causapay["recovery_rate"],
            "action_distribution": causapay_distribution,
        },
        "oracle": {"policy_value": oracle["policy_value"]},
        "incremental_value_vs_baseline": causapay["policy_value"] - baseline["policy_value"],
        "validation": validation,
    }


@app.get("/api/health")
def health_check():
    db_connected = False
    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
        db_connected = True
    except Exception:
        db_connected = False

    redis_status = getattr(app.state, "redis_status", {"enabled": False, "connected": False, "fallback_mode": True})
    n8n_enabled = str(os.getenv("N8N_ENABLED", "false")).strip().lower() in {"1", "true", "yes", "on"}
    n8n_url = str(os.getenv("N8N_WEBHOOK_URL", "")).strip()
    return {
        "status": "ok",
        "model_loaded": aipw_estimator is not None,
        "database": {"status": "connected" if db_connected else "disconnected"},
        "redis": {
            "enabled": bool(redis_status.get("enabled", False)),
            "connected": bool(redis_status.get("connected", False)),
            "fallback_mode": bool(redis_status.get("fallback_mode", True)),
        },
        "n8n": {
            "enabled": n8n_enabled,
            "configured": bool(n8n_url),
            "webhook_url_configured": bool(n8n_url),
            "mode": "local" if not n8n_enabled else ("n8n" if n8n_url else "disabled"),
        },
    }

@app.get("/api/summary", response_model=SummaryResponse)
def get_summary():
    if events_df is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    full = app.state.full_events
    actions = events_df["recommended_action"].values
    true_y = np.zeros(len(actions))
    for i, a in enumerate(actions):
        true_y[i] = test_hid[f"Y_{a}"].iloc[i]
    amounts = full["amount"].values
    gross = (true_y * amounts).sum()
    true_none = test_hid["Y_none"].values
    incremental = ((true_y - true_none) * amounts).sum()
    total_cost = sum(costs.get(a, 0.0) for a in actions)
    policy_val = incremental - total_cost
    distribution = pd.Series(actions).value_counts().to_dict()
    return SummaryResponse(
        total_failed_payments=len(events_df),
        gross_recovered=round(gross, 2),
        incremental_recovered=round(incremental, 2),
        intervention_cost=round(total_cost, 2),
        policy_value=round(policy_val, 2),
        action_distribution=distribution,
    )

@app.post("/api/decision", response_model=DecisionResponse)
def make_decision(req: DecisionRequest):
    if aipw_estimator is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    event_key = str(req.event_id)
    decision_key = _decision_idempotency_key(req)
    cached = _redis_get_json(decision_key)
    if cached is not None:
        return DecisionResponse(**cached)

    lock_token = _redis_lock(f"decision:lock:{event_key}", ttl_seconds=20)
    if redis_client is not None and lock_token is None:
        with SessionLocal() as session:
            existing_decision = get_event_decision(session, event_key)
            if existing_decision is not None:
                payload = _db_decision_payload(existing_decision)
                payment_event = session.query(PaymentEvent).filter(PaymentEvent.event_id == event_key).first()
                stored_payload = load_json_payload(payment_event.input_payload) if payment_event is not None else {}
                if not isinstance(stored_payload, dict):
                    stored_payload = {}
                payload["input_metadata"] = _dynamic_input_metadata(stored_payload)
                _redis_set_json(decision_key, payload, ttl_seconds=3600)
                return DecisionResponse(**payload)
        return JSONResponse(
            status_code=409,
            content={
                "event_id": event_key,
                "status": "in_progress",
                "detail": "A decision request for this event is already being processed.",
            },
        )
    try:
        with SessionLocal() as session:
            payment_event = session.query(PaymentEvent).filter(PaymentEvent.event_id == event_key).first()
            existing_decision = get_event_decision(session, event_key)
            if payment_event is not None and existing_decision is not None:
                payload = _db_decision_payload(existing_decision)
                stored_payload = load_json_payload(payment_event.input_payload)
                if not isinstance(stored_payload, dict):
                    stored_payload = {}
                payload["input_metadata"] = _dynamic_input_metadata(stored_payload)
                _redis_set_json(decision_key, payload, ttl_seconds=3600)
                return DecisionResponse(**payload)

            if payment_event is not None and existing_decision is None:
                stored_payload = load_json_payload(payment_event.input_payload)
                if not isinstance(stored_payload, dict):
                    stored_payload = {}
                payload = {
                    "event_id": event_key,
                    "prob_none": 0.0,
                    "prob_retry": 0.0,
                    "prob_whatsapp": 0.0,
                    "inc_prob_retry": 0.0,
                    "inc_prob_whatsapp": 0.0,
                    "eniv_none": 0.0,
                    "eniv_retry": 0.0,
                    "eniv_whatsapp": 0.0,
                    "recommended_action": "none",
                    "is_abstain": False,
                    "explanation": "Event already exists; no decision was stored for this event_id.",
                    "input_metadata": _dynamic_input_metadata(stored_payload),
                }
                _redis_set_json(decision_key, payload, ttl_seconds=3600)
                return DecisionResponse(**payload)

            data = req.dict()
            data["event_id"] = event_key
            data, input_metadata = _validate_and_default_observable_input(data)

            existing = getattr(app.state, "events_lookup", {}).get(event_key)
            if existing is not None:
                merged = existing.to_dict()
                merged.update({key: value for key, value in data.items() if value is not None})
                data = merged

            df = pd.DataFrame([data])
            try:
                cf = aipw_estimator.predict_counterfactuals(df)
            except KeyError as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"Missing required observable feature: {exc.args[0]}",
                ) from exc

            row = df.iloc[0].copy()
            for column, value in cf.iloc[0].items():
                if column != "event_id":
                    row[column] = value

            response = _build_decision_response(row)
            baseline_action = baseline_policy.predict(df)[0]
            response["baseline_action"] = baseline_action
            response["probability_none"] = float(response["prob_none"])
            response["probability_retry"] = float(response["prob_retry"])
            response["probability_whatsapp"] = float(response["prob_whatsapp"])
            response["incremental_retry"] = float(response["inc_prob_retry"])
            response["incremental_whatsapp"] = float(response["inc_prob_whatsapp"])
            response["eniv_retry"] = float(response["eniv_retry"])
            response["eniv_whatsapp"] = float(response["eniv_whatsapp"])
            response["uncertainty_retry"] = float(row.get("std_retry", np.nan)) if pd.notna(row.get("std_retry", np.nan)) else None
            response["uncertainty_whatsapp"] = float(row.get("std_whatsapp", np.nan)) if pd.notna(row.get("std_whatsapp", np.nan)) else None
            response["abstained"] = bool(response["is_abstain"])
            response["decision_reason"] = response["explanation"]
            response["model_version"] = "causapay-v1"
            response["created_at"] = datetime.utcnow().isoformat()
            response["input_metadata"] = input_metadata

            payment_event = session.query(PaymentEvent).filter(PaymentEvent.event_id == event_key).first()
            if payment_event is None:
                payment_event = PaymentEvent(
                    event_id=event_key,
                    payment_id=str(data.get("payment_id") or f"pay_{event_key}"),
                    customer_id=str(data.get("customer_id") or "unknown"),
                    amount=float(data.get("amount") or 0.0),
                    failure_reason=str(data.get("failure_context") or data.get("failure_reason") or "unknown"),
                    timestamp=datetime.utcnow(),
                    whatsapp_opted_in=_python_bool(data.get("whatsapp_opted_in", False)),
                    input_payload=json_safe(data),
                )
                try:
                    session.add(payment_event)
                    session.flush()
                except IntegrityError:
                    session.rollback()
                    payment_event = session.query(PaymentEvent).filter(PaymentEvent.event_id == event_key).first()
                    if payment_event is not None:
                        existing_decision = get_event_decision(session, event_key)
                        if existing_decision is not None:
                            payload = _db_decision_payload(existing_decision)
                            stored_payload = load_json_payload(payment_event.input_payload)
                            if not isinstance(stored_payload, dict):
                                stored_payload = {}
                            payload["input_metadata"] = _dynamic_input_metadata(stored_payload)
                            _redis_set_json(decision_key, payload, ttl_seconds=3600)
                            return DecisionResponse(**payload)
                    raise

            decision_id = str(uuid4())
            audit_payload = {
                "event_id": event_key,
                "timestamp": response["created_at"],
                "input_data": {key: _normalize_json_value(value) for key, value in data.items()},
                "recommended_action": response["recommended_action"],
                "baseline_action": response["baseline_action"],
                "probabilities": {
                    "prob_none": response["probability_none"],
                    "prob_retry": response["probability_retry"],
                    "prob_whatsapp": response["probability_whatsapp"],
                },
                "incremental_effects": {
                    "retry": response["incremental_retry"],
                    "whatsapp": response["incremental_whatsapp"],
                },
                "eniv_values": {
                    "none": float(response["eniv_none"]),
                    "retry": float(response["eniv_retry"]),
                    "whatsapp": float(response["eniv_whatsapp"]),
                },
                "uncertainty": {
                    "retry": response["uncertainty_retry"],
                    "whatsapp": response["uncertainty_whatsapp"],
                    "threshold": response["uncertainty_threshold"],
                    "kind": "tree-dispersion heuristic; not a calibrated confidence interval",
                },
                "abstained": response["abstained"],
                "causal_preferred_action": response["causal_preferred_action"],
                "fallback_action": response["fallback_action"],
                "whatsapp_eligible": response["whatsapp_eligible"],
                "decision_reason": response["decision_reason"],
                "model_version": response["model_version"],
                "input_metadata": input_metadata,
            }

            decision_record = CausaPayDecision(
                decision_id=decision_id,
                event_id=event_key,
                recommended_action=response["recommended_action"],
                baseline_action=response["baseline_action"],
                probability_none=response["probability_none"],
                probability_retry=response["probability_retry"],
                probability_whatsapp=response["probability_whatsapp"],
                incremental_retry=response["incremental_retry"],
                incremental_whatsapp=response["incremental_whatsapp"],
                eniv_retry=response["eniv_retry"],
                eniv_whatsapp=response["eniv_whatsapp"],
                uncertainty_retry=response["uncertainty_retry"],
                uncertainty_whatsapp=response["uncertainty_whatsapp"],
                abstained=response["abstained"],
                decision_reason=response["decision_reason"],
                model_version=response["model_version"],
                created_at=datetime.utcnow(),
                audit_payload=json_safe(audit_payload),
            )
            session.add(decision_record)
            session.commit()

            _redis_set_json(decision_key, response, ttl_seconds=3600)
            return DecisionResponse(**response)
    finally:
        _redis_unlock(f"decision:lock:{event_key}", lock_token)

@app.get("/api/policy-comparison", response_model=list[PolicyResult])
def get_policy_comparison():
    if not benchmark_results:
        raise HTTPException(status_code=503, detail="Benchmark not ready")
    return [PolicyResult(**r) for r in benchmark_results]

@app.get("/api/model-diagnostics")
def get_model_diagnostics():
    """Live, structured diagnostics from the fitted demo model and held-out rows."""
    if aipw_estimator is None or test_obs is None or inc_policy is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    uncertainty = aipw_estimator.uncertainty_diagnostics(test_obs)
    cf = aipw_estimator.predict_counterfactuals(test_obs)
    high = (cf['std_retry'] > inc_policy.uncertainty_threshold) | (cf['std_whatsapp'] > inc_policy.uncertainty_threshold)
    return {
        "overlap": aipw_estimator.propensity_overlap_diagnostics(),
        "uncertainty": {
            "threshold": float(inc_policy.uncertainty_threshold),
            "source": "standard deviation across final-stage random-forest tree predictions on AIPW pseudo-outcomes",
            "distributions": uncertainty,
            "rows_above_threshold": int(high.sum()),
            "rows_below_or_equal_threshold": int((~high).sum()),
            "abstention_proxy_rate": float(high.mean()),
            "limitation": "This tree-dispersion score is an abstention heuristic, not a calibrated confidence interval.",
        },
    }

@app.get("/api/events/{event_id}")
def get_event(event_id: str):
    if events_df is None:
        raise HTTPException(status_code=503, detail="Data not ready")

    with SessionLocal() as session:
        stored_event = session.query(PaymentEvent).filter(PaymentEvent.event_id == event_id).first()
        if stored_event is not None:
            decision = get_event_decision(session, event_id)
            response = _build_dynamic_event_response(stored_event, decision)
            workflow_row = session.query(WorkflowExecution).filter(WorkflowExecution.event_id == str(event_id)).order_by(WorkflowExecution.dispatch_timestamp.desc().nullslast()).first()
            if workflow_row is not None:
                response["workflow"] = {
                    "required": True,
                    "status": workflow_row.execution_status,
                    "mode": workflow_row.mode,
                    "workflow_id": workflow_row.workflow_id,
                    "decision_id": workflow_row.decision_id,
                    "action": workflow_row.action,
                    "failure_reason": workflow_row.failure_reason,
                    "last_updated": workflow_row.last_updated.isoformat() if workflow_row.last_updated else None,
                }
            else:
                response["workflow"] = {"required": False, "status": "NOT_REQUIRED", "mode": "local", "workflow_id": None, "decision_id": None, "action": None, "failure_reason": None, "last_updated": None}
            return response

    row = app.state.events_lookup.get(event_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Event not found")

    response = _build_decision_response(row)
    response["source"] = "seeded_demo"
    response["recommended_action"] = row["recommended_action"]
    response["baseline_action"] = row.get("baseline_action", "none")
    response["whatsapp_opted_in"] = row.get("whatsapp_opted_in")
    response["is_abstain"] = bool(row["is_abstain"])
    response["input_metadata"] = {
        "source": "seeded_demo",
        "supplied_fields": sorted([str(col) for col in row.index if col not in {"event_id", "recommended_action", "baseline_action", "is_abstain"}]),
        "defaulted_fields": [],
        "validation_warnings": [],
    }
    response["explanation"] = (
        f"ENIV (none)={response['eniv_none']:.2f}, "
        f"ENIV (retry)={response['eniv_retry']:.2f}, "
        f"ENIV (whatsapp)={response['eniv_whatsapp']:.2f}. "
        f"Recommendation: {response['recommended_action']}."
    )
    response["workflow"] = {"required": False, "status": "NOT_REQUIRED", "mode": "local", "workflow_id": None, "decision_id": None, "action": None, "failure_reason": None, "last_updated": None}
    return response


@app.get("/api/events/{event_id}/audit")
def get_event_audit(event_id: str):
    with SessionLocal() as session:
        decision = get_event_decision(session, event_id)
        state = session.query(PaymentRecoveryState).filter(PaymentRecoveryState.event_id == str(event_id)).first()
        workflow_rows = session.query(WorkflowExecution).filter(WorkflowExecution.event_id == str(event_id)).order_by(WorkflowExecution.dispatch_timestamp.asc()).all()
        if decision is None and state is None and not workflow_rows:
            raise HTTPException(status_code=404, detail="Audit trail not found for this event")
        audit = load_audit_payload(decision.audit_payload) if decision is not None else {}
        workflow_events = [
            {
                "workflow_id": row.workflow_id,
                "event_id": row.event_id,
                "action": row.action,
                "status": row.execution_status,
                "mode": row.mode,
                "dispatch_timestamp": row.dispatch_timestamp.isoformat() if row.dispatch_timestamp else None,
                "completion_timestamp": row.completion_timestamp.isoformat() if row.completion_timestamp else None,
                "failure_reason": row.failure_reason,
                "metadata": load_json_payload(row.workflow_metadata),
            }
            for row in workflow_rows
        ]
        response = {
            "event_id": str(event_id),
            "decision_id": decision.decision_id if decision else None,
            "timestamp": decision.created_at.isoformat() if decision and decision.created_at else None,
            "audit": audit,
            "recovery_lifecycle": _load_json_list(state.state_history) if state else [],
            "action_history": _load_json_list(state.action_history) if state else [],
            "workflow_events": workflow_events,
        }
        return response


@app.post("/api/workflows/outcome")
def workflow_outcome_callback(payload: dict):
    event_id = str(payload.get("event_id") or "").strip()
    workflow_id = str(payload.get("workflow_id") or "").strip()
    action = str(payload.get("action") or "").upper().strip()
    status = str(payload.get("status") or "PENDING").upper().strip()
    if not event_id:
        raise HTTPException(status_code=422, detail="Missing required field: event_id")
    if not workflow_id:
        raise HTTPException(status_code=422, detail="Missing required field: workflow_id")
    if action not in {"RETRY", "WHATSAPP"}:
        raise HTTPException(status_code=422, detail="action must be RETRY or WHATSAPP")
    if status not in {"SUCCESS", "FAILED", "PENDING"}:
        raise HTTPException(status_code=422, detail="status must be SUCCESS, FAILED, or PENDING")

    with SessionLocal() as session:
        execution = session.query(WorkflowExecution).filter(WorkflowExecution.workflow_id == workflow_id).first()
        if execution is None:
            execution = session.query(WorkflowExecution).filter(
                WorkflowExecution.event_id == event_id,
                WorkflowExecution.action == action,
                WorkflowExecution.execution_status.in_({"DISPATCHED", "EXECUTING", "PENDING"}),
            ).order_by(WorkflowExecution.dispatch_timestamp.desc().nullslast()).first()
        if execution is None:
            raise HTTPException(status_code=404, detail="Workflow execution not found for this callback")

        callback_payload = load_json_payload(json_safe(payload))
        if execution.callback_payload is not None:
            previous = load_json_payload(execution.callback_payload)
            if previous == callback_payload and execution.execution_status == status:
                return {
                    "event_id": event_id,
                    "workflow_id": workflow_id,
                    "status": "idempotent_duplicate",
                    "message": "Duplicate callback ignored.",
                }

        execution.callback_payload = json_safe(payload)
        execution.last_updated = datetime.utcnow()
        execution.execution_status = "COMPLETED" if status == "SUCCESS" else ("FAILED" if status == "FAILED" else "PENDING")
        execution.completion_timestamp = datetime.utcnow() if status in {"SUCCESS", "FAILED"} else execution.completion_timestamp
        execution.failure_reason = None if status == "SUCCESS" else (payload.get("failure_reason") or execution.failure_reason)
        execution.workflow_metadata = json_safe({"callback": payload, "updated_by": "workflow_callback"})
        session.add(execution)

        result_status = "RECOVERED" if status == "SUCCESS" else "FAILED" if status == "FAILED" else "PENDING"
        recorded = record_event_outcome(event_id, {"status": result_status, "source": "n8n_callback"})
        session.commit()
        response = {
            "event_id": event_id,
            "workflow_id": workflow_id,
            "action": action,
            "status": status,
            "recorded_outcome": result_status,
            "message": "Workflow callback accepted.",
            "result": recorded,
        }
        _redis_set_json(f"workflow_callback:{workflow_id}", response, ttl_seconds=3600)
        return response


@app.post("/api/events/{event_id}/outcome")
def record_event_outcome(event_id: str, payload: dict):
    status = str(payload.get("status", "PENDING")).upper()
    allowed_statuses = {"RECOVERED", "FAILED", "PENDING"}
    if status not in allowed_statuses:
        raise HTTPException(status_code=422, detail="status must be one of RECOVERED, FAILED, or PENDING")

    outcome_key = _outcome_idempotency_key(event_id, status, str(payload.get("source", "manual")))
    cached = _redis_get_json(outcome_key)
    if cached is not None:
        return cached

    lock_token = _redis_lock(f"outcome:lock:{event_id}", ttl_seconds=20)
    if redis_client is not None and lock_token is None:
        with SessionLocal() as session:
            current_state = session.query(PaymentRecoveryState).filter(PaymentRecoveryState.event_id == str(event_id)).first()
            if current_state is not None and current_state.outcome_status == status:
                payload_response = {
                    "event_id": str(event_id),
                    "status": status,
                    "current_state": current_state.current_state,
                    "outcome_status": current_state.outcome_status,
                    "recovery_lifecycle": _load_json_list(current_state.state_history),
                    "message": f"Recovery outcome recorded: {status}",
                }
                _redis_set_json(outcome_key, payload_response, ttl_seconds=3600)
                return payload_response
        return JSONResponse(
            status_code=409,
            content={"event_id": str(event_id), "status": "in_progress", "detail": "An outcome update for this event is already being processed."},
        )

    try:
        with SessionLocal() as session:
            event = session.query(PaymentEvent).filter(PaymentEvent.event_id == str(event_id)).first()
            if event is None:
                raise HTTPException(status_code=404, detail="Event not found")
            state = _ensure_recovery_state(session, event_id)
            previous_state = state.current_state
            current_outcome = state.outcome_status or "PENDING"

            if status == current_outcome and state.current_state in {"RECOVERED", "STOPPED", "WAITING_FOR_OUTCOME"}:
                response = {
                    "event_id": str(event_id),
                    "status": status,
                    "current_state": state.current_state,
                    "outcome_status": state.outcome_status,
                    "recovery_lifecycle": _load_json_list(state.state_history),
                    "message": f"Recovery outcome recorded: {status}",
                }
                _redis_set_json(outcome_key, response, ttl_seconds=3600)
                return response

            if status == "RECOVERED":
                new_state = "RECOVERED"
            elif status == "FAILED":
                new_state = "STOPPED"
            else:
                new_state = state.current_state or "WAITING_FOR_OUTCOME"

            if status == "PENDING":
                state.outcome_status = "PENDING"
                state.current_state = "WAITING_FOR_OUTCOME"
                new_state = "WAITING_FOR_OUTCOME"
            else:
                state.outcome_status = status
                state.current_state = new_state

            if previous_state != new_state or current_outcome != status:
                _append_recovery_transition(
                    session,
                    event_id,
                    previous_state,
                    new_state,
                    "OUTCOME",
                    f"outcome_status={status}",
                    {"status": status, "source": payload.get("source", "manual"), "outcome": status},
                )

            if status in {"RECOVERED", "FAILED"}:
                _append_recovery_action_history(
                    session,
                    event_id,
                    "OUTCOME",
                    "completed",
                    reason=f"status={status}",
                    outcome=status,
                )

            session.commit()
            response = {
                "event_id": str(event_id),
                "status": status,
                "current_state": state.current_state,
                "outcome_status": state.outcome_status,
                "recovery_lifecycle": _load_json_list(state.state_history),
                "message": f"Recovery outcome recorded: {status}",
            }
            _redis_set_json(outcome_key, response, ttl_seconds=3600)
            return response
    finally:
        _redis_unlock(f"outcome:lock:{event_id}", lock_token)


@app.post("/api/events/{event_id}/next-action")
def get_next_action(event_id: str):
    with SessionLocal() as session:
        event = session.query(PaymentEvent).filter(PaymentEvent.event_id == str(event_id)).first()
        if event is None:
            raise HTTPException(status_code=404, detail="Event not found")

        state = _ensure_recovery_state(session, event_id)
        cache_key = _next_action_idempotency_key(event_id, state)
        cached = _redis_get_json(cache_key)
        if cached is not None:
            return cached

        lock_token = _redis_lock(f"next_action:lock:{event_id}", ttl_seconds=20)
        if redis_client is not None and lock_token is None:
            recovery_snapshot = _recovery_state_for_response(session, event_id)
            return JSONResponse(
                status_code=409,
                content={
                    "event_id": str(event_id),
                    "status": "in_progress",
                    "detail": "A next-action request for this event is already being processed.",
                    "recovery_state": recovery_snapshot,
                },
            )
        try:
            action_history = _load_json_list(state.action_history)
            state_history = _load_json_list(state.state_history)
            current_state = state.current_state or "NEW"
            outcome_status = state.outcome_status or "PENDING"

            blocked_actions = []
            reasons = []
            if outcome_status == "RECOVERED":
                reasons.append("recovery_success")
                blocked_actions.append({"action": "WHATSAPP", "reason": "payment_already_recovered"})
                blocked_actions.append({"action": "RETRY", "reason": "payment_already_recovered"})
            if event.whatsapp_opted_in is False:
                blocked_actions.append({"action": "WHATSAPP", "reason": "whatsapp_opt_in_required"})

            retry_count = sum(1 for entry in action_history if str(entry.get("action", "")).upper() == "RETRY")
            whatsapp_count = sum(1 for entry in action_history if str(entry.get("action", "")).upper() == "WHATSAPP")
            if retry_count >= RECOVERY_CONFIG["max_retry_attempts"]:
                blocked_actions.append({"action": "RETRY", "reason": "max_retry_attempts_exceeded"})
            if whatsapp_count >= RECOVERY_CONFIG["max_whatsapp_attempts"]:
                blocked_actions.append({"action": "WHATSAPP", "reason": "max_whatsapp_attempts_exceeded"})

            if len(action_history) >= RECOVERY_CONFIG["max_total_interventions"]:
                reasons.append("max_total_interventions_reached")

            last_action = None
            last_timestamp = None
            for entry in reversed(action_history):
                action = str(entry.get("action", "")).upper()
                ts = entry.get("timestamp")
                if action in {"RETRY", "WHATSAPP"}:
                    last_action = action
                    last_timestamp = ts
                    break

            if last_action == "RETRY" and last_timestamp:
                try:
                    last_dt = datetime.fromisoformat(last_timestamp)
                    if (datetime.utcnow() - last_dt).total_seconds() < RECOVERY_CONFIG["retry_cooldown_minutes"] * 60:
                        blocked_actions.append({"action": "RETRY", "reason": "retry_cooldown_active"})
                except ValueError:
                    pass

            if last_action == "WHATSAPP" and last_timestamp:
                try:
                    last_dt = datetime.fromisoformat(last_timestamp)
                    if (datetime.utcnow() - last_dt).total_seconds() < RECOVERY_CONFIG["whatsapp_cooldown_minutes"] * 60:
                        blocked_actions.append({"action": "WHATSAPP", "reason": "whatsapp_cooldown_active"})
                except ValueError:
                    pass

            if state.current_state in {"STOPPED", "ABSTAINED", "EXHAUSTED"}:
                reasons.append("state_stop")

            if outcome_status == "FAILED":
                reasons.append("payment_failed")

            allowed_actions = ["NONE", "RETRY", "WHATSAPP"]
            blocked_map = {str(item["action"]).upper() for item in blocked_actions}
            allowed_actions = [action for action in allowed_actions if action not in blocked_map]

            should_stop = bool(reasons) or len(allowed_actions) == 0
            if should_stop:
                if current_state not in {"STOPPED", "ABSTAINED", "EXHAUSTED"}:
                    if any(r in {"max_total_interventions_reached", "no_valid_action_remains"} for r in reasons):
                        new_state = "EXHAUSTED"
                    elif "causal_abstention" in reasons:
                        new_state = "ABSTAINED"
                    else:
                        new_state = "STOPPED"
                    _append_recovery_transition(session, event_id, current_state, new_state, "STOP_RULES", "; ".join(reasons) or "no_valid_action_remains", {"reasons": reasons, "allowed_actions": allowed_actions, "blocked_actions": blocked_actions})
                    state.current_state = new_state
                    state.outcome_status = "FAILED" if not reasons and not any(r == "recovery_success" for r in reasons) else outcome_status
                session.commit()
                response = {
                    "event_id": str(event_id),
                    "current_state": state.current_state,
                    "recommended_action": "STOP",
                    "allowed_actions": ["NONE"],
                    "blocked_actions": blocked_actions,
                    "stopping_rules": {"should_stop": True, "reasons": reasons or ["no_valid_action_remains"]},
                    "decision": {
                        "prob_none": None,
                        "prob_retry": None,
                        "prob_whatsapp": None,
                        "incremental_effects": {},
                        "eniv": {},
                        "uncertainty": {},
                    },
                    "action_history": action_history,
                    "recovery_lifecycle": state_history,
                }
                _redis_set_json(cache_key, response, ttl_seconds=600)
                return response

            candidate_payload = _event_input_payload(event)
            candidate_payload.pop("event_id", None)
            candidate_payload.pop("amount", None)
            decision_response = make_decision(DecisionRequest(
                event_id=str(event_id),
                amount=float(_event_input_payload(event).get("amount", 0.0)),
                **candidate_payload,
            ))

            if getattr(decision_response, "is_abstain", False):
                reasons.append("causal_abstention")
                should_stop = True

            if should_stop:
                if current_state not in {"STOPPED", "ABSTAINED", "EXHAUSTED"}:
                    if "causal_abstention" in reasons:
                        new_state = "ABSTAINED"
                    elif any(r in {"max_total_interventions_reached", "no_valid_action_remains"} for r in reasons):
                        new_state = "EXHAUSTED"
                    else:
                        new_state = "STOPPED"
                    _append_recovery_transition(session, event_id, current_state, new_state, "STOP_RULES", "; ".join(reasons) or "no_valid_action_remains", {"reasons": reasons, "allowed_actions": allowed_actions, "blocked_actions": blocked_actions})
                    state.current_state = new_state
                    state.outcome_status = "PENDING"
                session.commit()
                response = {
                    "event_id": str(event_id),
                    "current_state": state.current_state,
                    "recommended_action": "STOP",
                    "allowed_actions": ["NONE"],
                    "blocked_actions": blocked_actions,
                    "stopping_rules": {"should_stop": True, "reasons": reasons},
                    "decision": {
                        "prob_none": decision_response.prob_none,
                        "prob_retry": decision_response.prob_retry,
                        "prob_whatsapp": decision_response.prob_whatsapp,
                        "incremental_effects": {"retry": decision_response.inc_prob_retry, "whatsapp": decision_response.inc_prob_whatsapp},
                        "eniv": {"none": decision_response.eniv_none, "retry": decision_response.eniv_retry, "whatsapp": decision_response.eniv_whatsapp},
                        "uncertainty": {"retry": None, "whatsapp": None},
                    },
                    "action_history": _load_json_list(state.action_history),
                    "recovery_lifecycle": _load_json_list(state.state_history),
                }
                _redis_set_json(cache_key, response, ttl_seconds=600)
                return response

            recommendation = str(decision_response.recommended_action).upper()
            if recommendation not in {"NONE", "RETRY", "WHATSAPP"}:
                recommendation = "NONE"
            if recommendation not in allowed_actions:
                if "RETRY" in allowed_actions:
                    recommendation = "RETRY"
                elif "NONE" in allowed_actions:
                    recommendation = "NONE"
                else:
                    recommendation = allowed_actions[0]

            chosen_action = recommendation
            if chosen_action == "WHATSAPP" and event.whatsapp_opted_in is False:
                chosen_action = "NONE"

            action_state = "DECIDED"
            if chosen_action == "RETRY":
                action_state = "RETRY_SCHEDULED"
            elif chosen_action == "WHATSAPP":
                action_state = "WHATSAPP_SCHEDULED"
            elif chosen_action == "NONE":
                action_state = "DECIDED"

            previous_state = state.current_state
            state.current_state = action_state
            state.outcome_status = "PENDING"
            latest_decision = get_event_decision(session, event_id)
            _append_recovery_action_history(session, event_id, chosen_action, "scheduled", decision_id=latest_decision.decision_id if latest_decision else None, reason=decision_response.explanation, outcome=None)
            _append_recovery_transition(session, event_id, previous_state, action_state, chosen_action, decision_response.explanation, {"recommended_action": chosen_action, "decision_id": latest_decision.decision_id if latest_decision else None})

            workflow = {"required": False, "status": "NOT_REQUIRED", "mode": "local", "workflow_id": None, "decision_id": None, "failure_reason": None}
            if chosen_action in {"RETRY", "WHATSAPP"}:
                workflow = dispatch_workflow(
                    event_id=str(event_id),
                    action=chosen_action,
                    decision_id=latest_decision.decision_id if latest_decision else None,
                    context={
                        "payment_id": event.payment_id,
                        "customer_id": event.customer_id,
                        "amount": float(event.amount or 0.0),
                        "recovery_state": action_state,
                        "attempt_number": 1 + sum(1 for entry in _load_json_list(state.action_history) if str(entry.get("action", "")).upper() in {"RETRY", "WHATSAPP"}),
                        "source": "next_action",
                    },
                )
                workflow = {**workflow, "required": True}
                _append_recovery_transition(
                    session,
                    event_id,
                    action_state,
                    action_state,
                    "WORKFLOW",
                    workflow.get("failure_reason") or "workflow_dispatch",
                    {"workflow_id": workflow.get("workflow_id"), "status": workflow.get("status"), "mode": workflow.get("mode")},
                )
                _append_recovery_action_history(
                    session,
                    event_id,
                    "WORKFLOW",
                    str(workflow.get("status", "DISPATCHED")),
                    decision_id=latest_decision.decision_id if latest_decision else None,
                    reason=workflow.get("failure_reason") or "workflow_dispatch",
                    outcome=None,
                )

            session.commit()

            response = {
                "event_id": str(event_id),
                "current_state": action_state,
                "recommended_action": chosen_action,
                "allowed_actions": allowed_actions,
                "blocked_actions": blocked_actions,
                "stopping_rules": {"should_stop": False, "reasons": []},
                "decision": {
                    "prob_none": decision_response.prob_none,
                    "prob_retry": decision_response.prob_retry,
                    "prob_whatsapp": decision_response.prob_whatsapp,
                    "incremental_effects": {
                        "retry": decision_response.inc_prob_retry,
                        "whatsapp": decision_response.inc_prob_whatsapp,
                    },
                    "eniv": {
                        "none": decision_response.eniv_none,
                        "retry": decision_response.eniv_retry,
                        "whatsapp": decision_response.eniv_whatsapp,
                    },
                    "uncertainty": {
                        "retry": None,
                        "whatsapp": None,
                    },
                },
                "workflow": {
                    "required": bool(workflow.get("required", False)),
                    "status": workflow.get("status", "NOT_REQUIRED"),
                    "mode": workflow.get("mode", "local"),
                    "workflow_id": workflow.get("workflow_id"),
                    "decision_id": workflow.get("decision_id"),
                    "failure_reason": workflow.get("failure_reason"),
                    "last_updated": workflow.get("last_updated"),
                },
                "action_history": _load_json_list(state.action_history),
                "recovery_lifecycle": _load_json_list(state.state_history),
            }
            _redis_set_json(cache_key, response, ttl_seconds=600)
            return response
        finally:
            _redis_unlock(f"next_action:lock:{event_id}", lock_token)


@app.get("/api/events", response_model=EventListResponse)
def list_events(limit: int = 30):
    """Browse seeded demo events and persisted dynamic events without exposing hidden outcomes."""
    if events_df is None:
        raise HTTPException(status_code=503, detail="Data not ready")
    limit = max(1, min(limit, 100))

    seeded_rows = [_event_payload(row) for _, row in events_df.iterrows()]
    for item in seeded_rows:
        item["source"] = "seeded_demo"

    dynamic_rows = []
    with SessionLocal() as session:
        for event in session.query(PaymentEvent).order_by(PaymentEvent.timestamp.desc()).all():
            decision = get_event_decision(session, event.event_id)
            payload = _event_payload_from_dict(_build_dynamic_event_response(event, decision), source="dynamic")
            dynamic_rows.append(payload)

    seen = set()
    merged_rows = []
    for row in dynamic_rows + seeded_rows:
        event_id = str(row.get("event_id"))
        if event_id in seen:
            continue
        seen.add(event_id)
        merged_rows.append(row)

    merged_rows = merged_rows[:limit]
    return EventListResponse(events=merged_rows, total=len(merged_rows))

@app.get("/api/diagnostics")
def get_diagnostics():
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluation", "results")
    pattern = os.path.join(results_dir, "diagnostic_summary_*.json")
    files = sorted(glob.glob(pattern), reverse=True)
    if not files:
        raise HTTPException(status_code=404, detail="No diagnostic file found")
    latest = files[0]
    with open(latest, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
