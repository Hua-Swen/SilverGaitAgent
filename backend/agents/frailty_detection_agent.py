"""
Frailty Detection Agent — Risk Stratification Engine
------------------------------------------------------
Combines CFS, Katz, SPPB, and contributing conditions scores.
Produces a frailty tier classification and activates management pathways.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from backend.models import Assessment, Patient
from backend.tools.scoring import classify_frailty

TIER_DISPLAY = {
    "robust": "✅ ROBUST",
    "pre-frail": "⚠️  PRE-FRAIL",
    "frail": "🔶 FRAIL",
    "severely-frail": "🔴 SEVERELY FRAIL",
}

MANAGEMENT_TRIGGERS = {
    "robust": ["education"],
    "pre-frail": ["education", "exercise", "monitoring"],
    "frail": ["education", "exercise", "sleep", "monitoring"],
    "severely-frail": ["education", "exercise", "sleep", "monitoring"],
}


def run_frailty_detection_agent(
    patient: Patient,
    assessment: Assessment,
    llm: BaseChatModel,
) -> Assessment:
    print("\n" + "=" * 60)
    print("FRAILTY DETECTION AGENT — Risk Stratification")
    print("=" * 60)

    if not assessment.cfs or not assessment.katz or not assessment.sppb:
        missing = [n for n, v in [("CFS", assessment.cfs), ("Katz", assessment.katz), ("SPPB", assessment.sppb)] if not v]
        print(f"[Warning] Incomplete data — missing: {', '.join(missing)}. Defaulting to pre-frail.")
        assessment.frailty_tier = "pre-frail"
        assessment.risk_explanation = "Incomplete data — defaulting to pre-frail for safety."
        return assessment

    # Print raw scores going into classification
    print(f"\n[Scores] CFS       : {assessment.cfs.score}/9 — {assessment.cfs.label}")
    print(f"[Scores] Katz ADL  : {assessment.katz.total}/6 — {assessment.katz.label}")
    print(f"[Scores] SPPB      : {assessment.sppb.total}/12 — {assessment.sppb.label}")
    if assessment.contributing:
        c = assessment.contributing
        print(f"[Scores] Contributing — Cognitive: {c.cognitive_risk} | Mood: {c.mood_risk} | "
              f"Sleep: {c.sleep_risk} | Social: {c.social_isolation_risk}")
    print("[Classifier] Running deterministic frailty classification...")

    # Deterministic classification
    tier, explanation = classify_frailty(assessment.cfs, assessment.katz, assessment.sppb)
    print(f"[Classifier] Initial tier from scores: {tier.upper()}")

    # Adjust upward if contributing conditions are severe
    if assessment.contributing:
        high_risks = [
            assessment.contributing.cognitive_risk == "high",
            assessment.contributing.mood_risk == "high",
            assessment.contributing.sleep_risk == "high",
            assessment.contributing.social_isolation_risk == "high",
        ]
        if sum(high_risks) >= 2 and tier == "robust":
            tier = "pre-frail"
            explanation += " Elevated psychosocial risk factors upgraded tier to pre-frail."
            print(f"[Classifier] Tier upgraded to PRE-FRAIL due to {sum(high_risks)} high psychosocial risks")
        elif sum(high_risks) >= 2 and tier == "pre-frail":
            tier = "frail"
            explanation += " Elevated psychosocial risk factors upgraded tier to frail."
            print(f"[Classifier] Tier upgraded to FRAIL due to {sum(high_risks)} high psychosocial risks")

    assessment.frailty_tier = tier
    assessment.risk_explanation = explanation

    # Use LLM to generate a human-readable narrative
    prompt = f"""You are a geriatric specialist summarizing a frailty risk assessment.

Patient: {patient.name}, Age: {patient.age}
Frailty tier: {tier}
Scoring explanation: {explanation}
CFS: {assessment.cfs.score} ({assessment.cfs.label})
Katz ADL: {assessment.katz.total}/6 ({assessment.katz.label})
SPPB: {assessment.sppb.total}/12 ({assessment.sppb.label})
Contributing conditions: {assessment.contributing.model_dump() if assessment.contributing else "Not assessed"}

Write a 3-4 sentence clinical summary explaining the risk tier to the patient in plain language.
Be empathetic and constructive. Focus on what can be done, not just the risk.
"""

    narrative = llm.invoke([HumanMessage(content=prompt)])
    print(f"\n{TIER_DISPLAY[tier]}\n")
    print(f"Assessment: {explanation}\n")
    print(f"Summary for patient:\n{narrative.content}\n")

    # Append narrative to risk explanation
    assessment.risk_explanation = explanation + "\n\n" + narrative.content

    # Show which management agents will be activated
    triggered = MANAGEMENT_TRIGGERS.get(tier, [])
    print(f"Management pathways activated: {', '.join(triggered)}")

    return assessment
