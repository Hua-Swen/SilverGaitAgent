"""
SQLite database layer using SQLAlchemy.
Stores patients and longitudinal assessments.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    Text,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from backend.models import Patient, Assessment

# --- DB location ---
DATA_DIR = Path(__file__).parent.parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "silvergait.db"

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


# --- ORM tables ---

class PatientRow(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    date_of_birth = Column(String, nullable=False)  # ISO format
    gender = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class AssessmentRow(Base):
    __tablename__ = "assessments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    assessed_at = Column(DateTime, default=datetime.utcnow)

    # Stored as JSON strings
    cfs_json = Column(Text)
    katz_json = Column(Text)
    sppb_json = Column(Text)
    contributing_json = Column(Text)

    history_summary = Column(Text)
    frailty_tier = Column(String)
    risk_explanation = Column(Text)
    education_plan = Column(Text)
    exercise_plan = Column(Text)
    sleep_plan = Column(Text)
    monitoring_notes = Column(Text)


class ConversationRow(Base):
    """Persists every chat message exchanged during coaching sessions."""

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    role = Column(String, nullable=False)   # "user" | "assistant"
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(engine)


# --- Patient CRUD ---

def create_patient(patient: Patient) -> Patient:
    with SessionLocal() as session:
        row = PatientRow(
            name=patient.name,
            date_of_birth=patient.date_of_birth.isoformat(),
            gender=patient.gender,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        patient.id = row.id
        patient.created_at = row.created_at.isoformat()
        return patient


def get_patient(patient_id: int) -> Optional[Patient]:
    with SessionLocal() as session:
        row = session.get(PatientRow, patient_id)
        if not row:
            return None
        return Patient(
            id=row.id,
            name=row.name,
            date_of_birth=row.date_of_birth,
            gender=row.gender,
            created_at=row.created_at.isoformat(),
        )


def list_patients() -> list[Patient]:
    with SessionLocal() as session:
        rows = session.query(PatientRow).order_by(PatientRow.name).all()
        return [
            Patient(
                id=r.id,
                name=r.name,
                date_of_birth=r.date_of_birth,
                gender=r.gender,
                created_at=r.created_at.isoformat(),
            )
            for r in rows
        ]


# --- Assessment CRUD ---

def save_assessment(assessment: Assessment) -> Assessment:
    with SessionLocal() as session:
        row = AssessmentRow(
            patient_id=assessment.patient_id,
            cfs_json=assessment.cfs.model_dump_json() if assessment.cfs else None,
            katz_json=assessment.katz.model_dump_json() if assessment.katz else None,
            sppb_json=assessment.sppb.model_dump_json() if assessment.sppb else None,
            contributing_json=assessment.contributing.model_dump_json() if assessment.contributing else None,
            history_summary=assessment.history_summary,
            frailty_tier=assessment.frailty_tier,
            risk_explanation=assessment.risk_explanation,
            education_plan=assessment.education_plan,
            exercise_plan=assessment.exercise_plan,
            sleep_plan=assessment.sleep_plan,
            monitoring_notes=assessment.monitoring_notes,
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        assessment.id = row.id
        assessment.assessed_at = row.assessed_at
        return assessment


# --- Conversation CRUD ---

def save_message(patient_id: int, role: str, content: str) -> None:
    """Append a single chat message to the conversation log."""
    with SessionLocal() as session:
        session.add(ConversationRow(patient_id=patient_id, role=role, content=content))
        session.commit()


def get_conversation(patient_id: int) -> list[dict]:
    """Return all chat messages for a patient, oldest first."""
    with SessionLocal() as session:
        rows = (
            session.query(ConversationRow)
            .filter(ConversationRow.patient_id == patient_id)
            .order_by(ConversationRow.timestamp.asc())
            .all()
        )
        return [{"role": r.role, "content": r.content} for r in rows]


# --- Assessment CRUD ---

def get_assessments_for_patient(patient_id: int) -> list[Assessment]:
    from backend.models import CFSScore, KatzScore, SPPBScore, ContributingConditionsScore

    with SessionLocal() as session:
        rows = (
            session.query(AssessmentRow)
            .filter(AssessmentRow.patient_id == patient_id)
            .order_by(AssessmentRow.assessed_at.desc())
            .all()
        )

        result = []
        for r in rows:
            a = Assessment(
                id=r.id,
                patient_id=r.patient_id,
                assessed_at=r.assessed_at,
                history_summary=r.history_summary,
                frailty_tier=r.frailty_tier,
                risk_explanation=r.risk_explanation,
                education_plan=r.education_plan,
                exercise_plan=r.exercise_plan,
                sleep_plan=r.sleep_plan,
                monitoring_notes=r.monitoring_notes,
            )
            if r.cfs_json:
                a.cfs = CFSScore.model_validate_json(r.cfs_json)
            if r.katz_json:
                a.katz = KatzScore.model_validate_json(r.katz_json)
            if r.sppb_json:
                a.sppb = SPPBScore.model_validate_json(r.sppb_json)
            if r.contributing_json:
                a.contributing = ContributingConditionsScore.model_validate_json(r.contributing_json)
            result.append(a)

        return result
