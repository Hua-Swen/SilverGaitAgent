"""
SilverGait — Elderly Frailty Recognition & Management System
CLI Entry Point
"""

from datetime import date

from backend.config import get_llm
from backend.database.db import (
    init_db,
    create_patient,
    get_patient,
    list_patients,
    save_assessment,
    get_assessments_for_patient,
    get_conversation,
)
from backend.models import Patient
from backend.graph.workflow import run_full_assessment
from backend.agents.chat_agent import run_chat_session


BANNER = """
╔══════════════════════════════════════════════════════════╗
║           SilverGait — Frailty Assessment System         ║
║     Recognise · Assess · Manage · Monitor · Protect      ║
╚══════════════════════════════════════════════════════════╝
"""


def prompt_new_patient() -> Patient:
    print("\n--- New Patient Registration ---")
    name = input("Full name: ").strip()
    dob_str = input("Date of birth (YYYY-MM-DD): ").strip()
    dob = date.fromisoformat(dob_str)
    gender = input("Gender (male/female/other): ").strip().lower()
    return Patient(name=name, date_of_birth=dob, gender=gender)


def select_patient() -> Patient | None:
    patients = list_patients()
    if not patients:
        print("No patients found.")
        return None

    print("\n--- Existing Patients ---")
    for p in patients:
        print(f"  [{p.id}] {p.name} — Age {p.age}, {p.gender}")

    pid = input("\nEnter patient ID: ").strip()
    return get_patient(int(pid))


def view_history(patient: Patient):
    assessments = get_assessments_for_patient(patient.id)
    if not assessments:
        print(f"\nNo assessment history for {patient.name}.")
        return

    print(f"\n{'=' * 60}")
    print(f"  Assessment History — {patient.name}  (Age {patient.age}, {patient.gender})")
    print(f"{'=' * 60}")

    for i, a in enumerate(assessments, 1):
        date_str = a.assessed_at.strftime("%Y-%m-%d %H:%M") if a.assessed_at else "unknown date"
        tier = (a.frailty_tier or "not classified").upper()

        print(f"\n  Assessment #{i}  —  {date_str}")
        print(f"  {'─' * 54}")

        # Frailty tier
        print(f"  Frailty Tier    : {tier}")

        # CFS
        if a.cfs:
            print(f"  CFS Score       : {a.cfs.score}/9  —  {a.cfs.label}")
            if a.cfs.notes:
                print(f"                    {a.cfs.notes}")
        else:
            print(f"  CFS Score       : not recorded")

        # Katz ADL
        if a.katz:
            adl_fields = ["bathing", "dressing", "toileting", "transferring", "continence", "feeding"]
            dependent = [f for f in adl_fields if not getattr(a.katz, f, True)]
            print(f"  Katz ADL        : {a.katz.total}/6  —  {a.katz.label}")
            if dependent:
                print(f"                    Needs help with: {', '.join(dependent)}")
        else:
            print(f"  Katz ADL        : not recorded")

        # SPPB
        if a.sppb:
            print(f"  SPPB Score      : {a.sppb.total}/12  —  {a.sppb.label}")
            print(f"                    Balance {a.sppb.balance_score}/4 | Gait {a.sppb.gait_speed_score}/4 | Chair stand {a.sppb.chair_stand_score}/4")
        else:
            print(f"  SPPB Score      : not recorded")

        # Contributing conditions
        if a.contributing:
            c = a.contributing
            print(f"  Contributing    : Cognitive {c.cognitive_risk} | Mood {c.mood_risk} | "
                  f"Sleep {c.sleep_risk} | Social {c.social_isolation_risk}")
        else:
            print(f"  Contributing    : not recorded")

        # History summary
        if a.history_summary:
            print(f"\n  History Summary :")
            for line in a.history_summary.strip().splitlines():
                print(f"    {line}")

        # Risk explanation
        if a.risk_explanation:
            print(f"\n  Risk Explanation:")
            for line in a.risk_explanation.strip().splitlines():
                print(f"    {line}")

        # Management plans (collapsed to first 2 lines each)
        for label, text in [
            ("Education Plan", a.education_plan),
            ("Exercise Plan",  a.exercise_plan),
            ("Sleep Plan",     a.sleep_plan),
            ("Monitoring",     a.monitoring_notes),
        ]:
            if text:
                lines = text.strip().splitlines()
                preview = lines[0] if lines else ""
                more = f"  (+{len(lines)-1} more lines)" if len(lines) > 1 else ""
                print(f"\n  {label:<16}: {preview}{more}")

    print(f"\n{'=' * 60}\n")


def _select_provider() -> str:
    """Ask the user which LLM provider to use for this session."""
    from backend.config import SUPPORTED_PROVIDERS
    import os

    # If already set via env, use it silently
    env_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if env_provider in SUPPORTED_PROVIDERS:
        print(f"Using provider from environment: {env_provider.upper()}")
        return env_provider

    print("\nSelect LLM provider:")
    for i, p in enumerate(SUPPORTED_PROVIDERS, 1):
        print(f"  [{i}] {p.capitalize()}")
    choice = input("\nProvider (1/2/3) or name: ").strip().lower()

    if choice in ("1", "claude"):
        return "claude"
    elif choice in ("2", "openai"):
        return "openai"
    elif choice in ("3", "gemini"):
        return "gemini"
    else:
        print(f"Unrecognised choice '{choice}', defaulting to Claude.")
        return "claude"


def main():
    print(BANNER)
    init_db()
    provider = _select_provider()
    llm = get_llm(provider)
    print(f"Provider ready: {provider.upper()}\n")

    while True:
        print("\n=== Main Menu ===")
        print("  [1] New patient assessment")
        print("  [2] Continue assessment for existing patient")
        print("  [3] View patient history")
        print("  [4] Register new patient (without assessment)")
        print("  [q] Quit")

        choice = input("\nChoice: ").strip().lower()

        if choice == "q":
            print("\nGoodbye.\n")
            break

        elif choice == "1":
            patient = prompt_new_patient()
            patient = create_patient(patient)
            print(f"\n✓ Patient registered: {patient.name} (ID: {patient.id})")
            print("\nStarting full assessment workflow...")
            assessment = run_full_assessment(patient=patient, llm=llm)
            assessment = save_assessment(assessment)
            print(f"\n✓ Assessment saved (ID: {assessment.id})")
            # Enter coaching chat with a fresh conversation
            history = get_conversation(patient.id)
            run_chat_session(patient=patient, assessment=assessment, llm=llm, history=history)

        elif choice == "2":
            patient = select_patient()
            if not patient:
                continue
            assessments = get_assessments_for_patient(patient.id)
            completed = [a for a in assessments if a.frailty_tier]
            if completed:
                # Patient has a finished assessment — resume coaching chat
                assessment = completed[0]  # most recent
                history = get_conversation(patient.id)
                session_label = "Resuming coaching session" if history else "Starting coaching session"
                print(f"\n{session_label} for {patient.name} (tier: {assessment.frailty_tier})...")
                run_chat_session(patient=patient, assessment=assessment, llm=llm, history=history)
            else:
                # No completed assessment yet — run one, then chat
                print(f"\nNo completed assessment found for {patient.name}. Running full assessment...")
                assessment = run_full_assessment(patient=patient, llm=llm)
                assessment = save_assessment(assessment)
                print(f"\n✓ Assessment saved (ID: {assessment.id})")
                history = get_conversation(patient.id)
                run_chat_session(patient=patient, assessment=assessment, llm=llm, history=history)

        elif choice == "3":
            patient = select_patient()
            if patient:
                view_history(patient)

        elif choice == "4":
            patient = prompt_new_patient()
            patient = create_patient(patient)
            print(f"\n✓ Patient registered: {patient.name} (ID: {patient.id})")

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
