import json
import os
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String, Text, create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


class PaymentEvent(Base):
    __tablename__ = "payment_events"

    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    payment_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    customer_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    amount: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    failure_reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    whatsapp_opted_in: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    input_payload: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    decisions: Mapped[list["CausaPayDecision"]] = relationship(back_populates="event")
    recovery_state: Mapped[Optional["PaymentRecoveryState"]] = relationship(back_populates="event")


class CausaPayDecision(Base):
    __tablename__ = "causapay_decisions"

    decision_id: Mapped[str] = mapped_column(String, primary_key=True)
    event_id: Mapped[str] = mapped_column(String, ForeignKey("payment_events.event_id"), index=True, nullable=False)
    recommended_action: Mapped[str] = mapped_column(String, nullable=False)
    baseline_action: Mapped[str] = mapped_column(String, nullable=False, default="none")
    probability_none: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    probability_retry: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    probability_whatsapp: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    incremental_retry: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    incremental_whatsapp: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    eniv_retry: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    eniv_whatsapp: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    uncertainty_retry: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    uncertainty_whatsapp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    abstained: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    decision_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    model_version: Mapped[str] = mapped_column(String, nullable=False, default="causapay-v1")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    audit_payload: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    event: Mapped[PaymentEvent] = relationship(back_populates="decisions")


class InterventionOutcome(Base):
    __tablename__ = "intervention_outcomes"

    outcome_id: Mapped[str] = mapped_column(String, primary_key=True)
    event_id: Mapped[str] = mapped_column(String, ForeignKey("payment_events.event_id"), index=True, nullable=False)
    action_taken: Mapped[str] = mapped_column(String, nullable=False)
    outcome: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    recovered_amount: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class PaymentRecoveryState(Base):
    __tablename__ = "payment_recovery_states"

    event_id: Mapped[str] = mapped_column(String, ForeignKey("payment_events.event_id"), primary_key=True, nullable=False)
    current_state: Mapped[str] = mapped_column(String, nullable=False, default="NEW")
    outcome_status: Mapped[str] = mapped_column(String, nullable=False, default="PENDING")
    action_history: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    state_history: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    last_action: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    event: Mapped[PaymentEvent] = relationship(back_populates="recovery_state")


class RecoveryTransitionLog(Base):
    __tablename__ = "recovery_transition_logs"

    transition_id: Mapped[str] = mapped_column(String, primary_key=True)
    event_id: Mapped[str] = mapped_column(String, ForeignKey("payment_events.event_id"), index=True, nullable=False)
    previous_state: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    new_state: Mapped[str] = mapped_column(String, nullable=False)
    triggering_action: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class WorkflowExecution(Base):
    __tablename__ = "workflow_executions"

    workflow_id: Mapped[str] = mapped_column(String, primary_key=True)
    event_id: Mapped[str] = mapped_column(String, ForeignKey("payment_events.event_id"), index=True, nullable=False)
    decision_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    action: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    execution_status: Mapped[str] = mapped_column(String, nullable=False, default="PENDING")
    mode: Mapped[str] = mapped_column(String, nullable=False, default="local")
    dispatch_timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completion_timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    failure_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    callback_payload: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    workflow_metadata: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


def _database_url() -> str:
    return os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL") or "sqlite:///./causapay.db"


engine = create_engine(
    _database_url(),
    future=True,
    echo=os.getenv("SQLALCHEMY_ECHO", "false").lower() == "true",
    connect_args={"check_same_thread": False} if _database_url().startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    inspector = inspect(engine)

    if inspector.has_table("payment_events"):
        payment_columns = {column["name"] for column in inspector.get_columns("payment_events")}
        for column_name, column_type in {"input_payload": "TEXT"}.items():
            if column_name not in payment_columns:
                with engine.begin() as connection:
                    connection.execute(text(f"ALTER TABLE payment_events ADD COLUMN {column_name} {column_type}"))

    if inspector.has_table("causapay_decisions"):
        decision_columns = {column["name"] for column in inspector.get_columns("causapay_decisions")}
        if "audit_payload" not in decision_columns:
            with engine.begin() as connection:
                connection.execute(text("ALTER TABLE causapay_decisions ADD COLUMN audit_payload TEXT"))


def json_safe(value: Any) -> str:
    return json.dumps(value, default=str)


def load_audit_payload(raw: Optional[str]) -> dict[str, Any]:
    if raw in (None, ""):
        return {}
    try:
        loaded = json.loads(raw)
        return loaded if isinstance(loaded, dict) else {"value": loaded}
    except (TypeError, ValueError):
        return {"raw": raw}


def load_json_payload(raw: Optional[str]) -> dict[str, Any]:
    if raw in (None, ""):
        return {}
    try:
        loaded = json.loads(raw)
        return loaded if isinstance(loaded, dict) else {"value": loaded}
    except (TypeError, ValueError):
        return {"raw": raw}


def get_event_decision(session: Session, event_id: str) -> Optional[CausaPayDecision]:
    return (
        session.query(CausaPayDecision)
        .filter(CausaPayDecision.event_id == str(event_id))
        .order_by(CausaPayDecision.created_at.desc())
        .first()
    )
