"""
Management Router Agent
-----------------------
After frailty risk is classified, this agent reviews the full clinical
summary and decides which of the four management agents to activate.

All four agents are considered regardless of frailty tier. The decision
is driven by the patient's individual needs as reflected in the assessment.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from backend.models import Assessment, Patient


AGENT_DESCRIPTIONS = {
    "education": (
        "Physical Education Agent — frailty education, fall prevention, "
        "and mobility preservation strategies."
    ),
    "exercise": (
        "Exercise Agent — structured 4-week exercise program (strength, "
        "balance, mobility) tailored to the patient's physical performance."
    ),
    "sleep": (
        "Sleep Agent — sleep hygiene coaching and CBT-I interventions for "
        "patients with poor sleep quality or insomnia."
    ),
    "monitoring": (
        "Monitoring Agent — longitudinal trend tracking, scheduled "
        "reassessments, and deterioration alerts."
    ),
}


class ManagementRoutingDecision(BaseModel):
    """Structured output from the management router LLM call."""

    agents_to_activate: list[str] = Field(
        description=(
            "List of management agents to activate. "
            "Valid values: 'education', 'exercise', 'sleep', 'monitoring'. "
            "Include only those clearly warranted by the clinical picture."
        )
    )
    rationale: str = Field(
        description=(
            "A concise clinical rationale (2-4 sentences) explaining why "
            "each selected agent was chosen and why any were excluded."
        )
    )


def run_management_router_agent(
    patient: Patient,
    assessment: Assessment,
    llm: BaseChatModel,
) -> Assessment:
    print("\n" + "=" * 60)
    print("MANAGEMENT ROUTER AGENT — Personalised Pathway Selection")
    print("=" * 60)

    # Build a rich clinical summary for the LLM
    contributing_summary = "Not assessed"
    if assessment.contributing:
        c = assessment.contributing
        contributing_summary = (
            f"Cognitive risk: {c.cognitive_risk} | "
            f"Mood risk: {c.mood_risk} | "
            f"Sleep risk: {c.sleep_risk} | "
            f"Social isolation risk: {c.social_isolation_risk}"
        )
        if c.notes:
            contributing_summary += f"\nNotes: {c.notes}"

    sppb_detail = "Not assessed"
    if assessment.sppb:
        s = assessment.sppb
        sppb_detail = (
            f"Total {s.total}/12 ({s.label}) — "
            f"Balance: {s.balance_score}/4, "
            f"Gait speed: {s.gait_speed_score}/4, "
            f"Chair stand: {s.chair_stand_score}/4"
        )
        if s.notes:
            sppb_detail += f"\nNotes: {s.notes}"

    agent_menu = "\n".join(
        f"- {name}: {desc}" for name, desc in AGENT_DESCRIPTIONS.items()
    )

    prompt = f"""You are a geriatric care coordinator deciding which management agents to activate for a patient.

PATIENT
  Name : {patient.name}
  Age  : {patient.age}

FRAILTY CLASSIFICATION
  Tier        : {assessment.frailty_tier or "unknown"}
  Explanation : {assessment.risk_explanation or "N/A"}

CLINICAL SCORES
  CFS  : {f"{assessment.cfs.score}/9 — {assessment.cfs.label}" if assessment.cfs else "Not assessed"}
  Katz : {f"{assessment.katz.total}/6 — {assessment.katz.label}" if assessment.katz else "Not assessed"}
  SPPB : {sppb_detail}

CONTRIBUTING CONDITIONS
  {contributing_summary}

HISTORY SUMMARY
  {assessment.history_summary or "Not available"}

AVAILABLE MANAGEMENT AGENTS
{agent_menu}

TASK
Review the full clinical picture above and select the management agents this patient needs.
Consider all four agents regardless of frailty tier — choose based on what the individual patient's data indicates.
Return your decision in structured JSON.
"""

    structured_llm = llm.with_structured_output(ManagementRoutingDecision)
    decision: ManagementRoutingDecision = structured_llm.invoke(
        [HumanMessage(content=prompt)]
    )

    # Validate: keep only recognised agent names
    valid = set(AGENT_DESCRIPTIONS.keys())
    chosen = [a for a in decision.agents_to_activate if a in valid]

    if not chosen:
        print("[Router] LLM returned no valid agents — defaulting to all four.")
        chosen = list(valid)

    print(f"\n[Router] Agents activated : {', '.join(chosen)}")
    print(f"[Router] Rationale:\n{decision.rationale}\n")

    assessment.management_routes = chosen
    assessment.management_routing_rationale = decision.rationale

    return assessment
