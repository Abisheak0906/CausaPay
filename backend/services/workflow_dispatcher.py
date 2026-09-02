import json
import os
import uuid
from datetime import datetime
from urllib import error, request

from sqlalchemy.exc import SQLAlchemyError

from db import SessionLocal, WorkflowExecution, json_safe


WORKFLOW_ACTIONS = {"RETRY", "WHATSAPP"}


def _n8n_enabled() -> bool:
    raw = os.getenv("N8N_ENABLED", "false")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _n8n_webhook_url() -> str:
    return str(os.getenv("N8N_WEBHOOK_URL", "")).strip()


def workflow_status_snapshot(event_id: str, decision_id: str | None = None, action: str | None = None) -> dict:
    with SessionLocal() as session:
        rows = (
            session.query(WorkflowExecution)
            .filter(WorkflowExecution.event_id == str(event_id))
            .order_by(WorkflowExecution.dispatch_timestamp.desc().nullslast())
            .all()
        )
        if not rows:
            return {"required": False, "status": "NOT_REQUIRED", "mode": "disabled", "workflow_id": None, "failure_reason": None}

        latest = rows[0]
        return {
            "required": bool(action is None or str(action).upper() in WORKFLOW_ACTIONS),
            "status": latest.execution_status,
            "mode": latest.mode,
            "workflow_id": latest.workflow_id,
            "decision_id": latest.decision_id,
            "action": latest.action,
            "failure_reason": latest.failure_reason,
            "last_updated": latest.last_updated.isoformat() if latest.last_updated else None,
        }


def _persist_workflow_record(
    *,
    event_id: str,
    decision_id: str | None,
    action: str,
    execution_status: str,
    mode: str,
    workflow_id: str,
    failure_reason: str | None = None,
    metadata: dict | None = None,
) -> dict:
    with SessionLocal() as session:
        execution = session.query(WorkflowExecution).filter(WorkflowExecution.workflow_id == workflow_id).first()
        if execution is None:
            execution = WorkflowExecution(
                workflow_id=workflow_id,
                event_id=str(event_id),
                decision_id=decision_id,
                action=str(action).upper(),
                execution_status=execution_status,
                mode=mode,
                dispatch_timestamp=datetime.utcnow(),
                completion_timestamp=None,
                failure_reason=failure_reason,
                callback_payload=None,
                workflow_metadata=json_safe(metadata or {}),
                last_updated=datetime.utcnow(),
            )
            session.add(execution)
        else:
            execution.event_id = str(event_id)
            execution.decision_id = decision_id
            execution.action = str(action).upper()
            execution.execution_status = execution_status
            execution.mode = mode
            execution.failure_reason = failure_reason
            execution.workflow_metadata = json_safe(metadata or {})
            execution.last_updated = datetime.utcnow()
            if execution.dispatch_timestamp is None:
                execution.dispatch_timestamp = datetime.utcnow()

        if execution_status in {"COMPLETED", "FAILED", "SIMULATED"}:
            execution.completion_timestamp = datetime.utcnow()
        try:
            session.commit()
        except SQLAlchemyError as exc:
            session.rollback()
            return {
                "workflow_id": workflow_id,
                "event_id": str(event_id),
                "decision_id": decision_id,
                "action": str(action).upper(),
                "status": "SIMULATED" if mode == "local" else "FAILED",
                "mode": mode,
                "failure_reason": f"database_commit_error:{type(exc).__name__}:{exc}",
                "last_updated": datetime.utcnow().isoformat(),
                "database_error": True,
            }

    return {
        "workflow_id": workflow_id,
        "event_id": str(event_id),
        "decision_id": decision_id,
        "action": str(action).upper(),
        "status": execution_status,
        "mode": mode,
        "failure_reason": failure_reason,
        "last_updated": datetime.utcnow().isoformat(),
    }


def dispatch_workflow(event_id: str, action: str, decision_id: str | None, context: dict | None = None) -> dict:
    action_name = str(action or "").upper()
    context = context or {}
    if action_name not in WORKFLOW_ACTIONS:
        return {
            "required": False,
            "status": "NOT_REQUIRED",
            "mode": "disabled",
            "workflow_id": None,
            "decision_id": decision_id,
            "failure_reason": None,
            "last_updated": datetime.utcnow().isoformat(),
        }

    workflow_id = str(context.get("workflow_id") or uuid.uuid4())
    enabled = _n8n_enabled()
    webhook_url = _n8n_webhook_url()

    if not enabled:
        return _persist_workflow_record(
            event_id=event_id,
            decision_id=decision_id,
            action=action_name,
            execution_status="SIMULATED",
            mode="local",
            workflow_id=workflow_id,
            failure_reason=None,
            metadata={"dispatch_mode": "simulated", "context": context},
        )

    if not webhook_url:
        record = _persist_workflow_record(
            event_id=event_id,
            decision_id=decision_id,
            action=action_name,
            execution_status="FAILED",
            mode="n8n",
            workflow_id=workflow_id,
            failure_reason="missing_n8n_webhook_url",
            metadata={"dispatch_mode": "n8n", "context": context},
        )
        return {**record, "required": True, "status": "FAILED", "mode": "n8n"}

    payload = {
        "event_id": str(event_id),
        "payment_id": str(context.get("payment_id") or f"pay_{event_id}"),
        "customer_id": str(context.get("customer_id") or "unknown"),
        "decision_id": decision_id,
        "action": action_name,
        "amount": context.get("amount"),
        "workflow_context": {
            "recovery_state": context.get("recovery_state", "NEW"),
            "attempt_number": int(context.get("attempt_number", 1)),
            "source": context.get("source", "decision_engine"),
        },
    }

    try:
        req = request.Request(
            webhook_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "CausaPay-Workflow-Dispatcher"},
            method="POST",
        )
        with request.urlopen(req, timeout=5) as response:
            status_code = getattr(response, "status", 200)
            if 200 <= status_code < 300:
                record = _persist_workflow_record(
                    event_id=event_id,
                    decision_id=decision_id,
                    action=action_name,
                    execution_status="DISPATCHED",
                    mode="n8n",
                    workflow_id=workflow_id,
                    failure_reason=None,
                    metadata={"dispatch_mode": "n8n", "payload": payload, "context": context},
                )
                return {**record, "required": True, "status": "DISPATCHED", "mode": "n8n"}
            record = _persist_workflow_record(
                event_id=event_id,
                decision_id=decision_id,
                action=action_name,
                execution_status="FAILED",
                mode="n8n",
                workflow_id=workflow_id,
                failure_reason=f"http_status_{status_code}",
                metadata={"dispatch_mode": "n8n", "payload": payload, "context": context},
            )
            return {**record, "required": True, "status": "FAILED", "mode": "n8n"}
    except error.HTTPError as exc:
        record = _persist_workflow_record(
            event_id=event_id,
            decision_id=decision_id,
            action=action_name,
            execution_status="FAILED",
            mode="n8n",
            workflow_id=workflow_id,
            failure_reason=f"http_error_{exc.code}",
            metadata={"dispatch_mode": "n8n", "payload": payload, "context": context},
        )
        return {**record, "required": True, "status": "FAILED", "mode": "n8n"}
    except Exception as exc:
        record = _persist_workflow_record(
            event_id=event_id,
            decision_id=decision_id,
            action=action_name,
            execution_status="FAILED",
            mode="n8n",
            workflow_id=workflow_id,
            failure_reason=str(exc),
            metadata={"dispatch_mode": "n8n", "payload": payload, "context": context},
        )
        return {**record, "required": True, "status": "FAILED", "mode": "n8n"}
