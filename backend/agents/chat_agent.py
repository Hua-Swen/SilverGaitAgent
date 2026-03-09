"""
Post-Assessment Health Coach Chat Agent
----------------------------------------
Disseminates personalised management plan information across multiple conversation
turns rather than dumping everything at once.  Conversation history is persisted
to SQLite so returning patients can continue where they left off.
"""

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from backend.models import Patient, Assessment


# Session trigger tokens — the LLM is told what these mean in the system prompt
_TRIGGER_START = "[SESSION_START: Patient has just completed their frailty assessment. Greet them warmly, briefly acknowledge that the assessment is done, and gently introduce their results. Do NOT cover everything — just open the conversation.]"
_TRIGGER_RETURN = "[RETURNING_SESSION: Patient is logging back in. Welcome them back warmly by name. Briefly recap the last topic you discussed, then ask how they have been and invite them to continue.]"


def _build_system_prompt(patient: Patient, assessment: Assessment) -> str:
    tier = (assessment.frailty_tier or "not classified").upper()

    cfs = (
        f"CFS score {assessment.cfs.score}/9 ({assessment.cfs.label})"
        if assessment.cfs
        else "not recorded"
    )
    katz = (
        f"Katz Index {assessment.katz.total}/6 ({assessment.katz.label})"
        if assessment.katz
        else "not recorded"
    )
    sppb = (
        f"SPPB {assessment.sppb.total}/12 ({assessment.sppb.label})"
        if assessment.sppb
        else "not recorded"
    )

    history_line = assessment.history_summary or "No summary recorded."
    risk_line = assessment.risk_explanation or "No explanation recorded."

    plan_sections: list[str] = []
    if assessment.education_plan:
        plan_sections.append(f"EDUCATION PLAN:\n{assessment.education_plan}")
    if assessment.exercise_plan:
        plan_sections.append(f"EXERCISE PLAN:\n{assessment.exercise_plan}")
    if assessment.sleep_plan:
        plan_sections.append(f"SLEEP PLAN:\n{assessment.sleep_plan}")
    if assessment.monitoring_notes:
        plan_sections.append(f"MONITORING PLAN:\n{assessment.monitoring_notes}")

    plans_text = (
        "\n\n".join(plan_sections)
        if plan_sections
        else "No specific management plans were generated for this patient."
    )

    return f"""You are a compassionate health coach for {patient.name}, age {patient.age}.

FRAILTY ASSESSMENT RESULTS:
- Frailty classification: {tier}
- {cfs}
- {katz}
- {sppb}
- History summary: {history_line}
- Risk explanation: {risk_line}

MANAGEMENT PLANS — YOUR KNOWLEDGE BASE (share these gradually, not all at once):
{plans_text}

YOUR ROLE:
- You are a warm, supportive health coach. Use plain language; avoid clinical jargon.
- Disseminate the management plan GRADUALLY across multiple conversation turns.
- Suggested order of topics: (1) explain frailty results in simple terms, (2) frailty education and fall prevention, (3) exercise recommendations, (4) sleep advice if applicable, (5) monitoring schedule.
- After each topic, check in — ask how the patient feels, invite questions — before moving on.
- NEVER repeat topics already well-covered in the conversation history.
- Keep each response focused and digestible (3–5 sentences or a short list). Do not overwhelm.
- When you see {_TRIGGER_START!r}, greet the patient warmly and open with their results.
- When you see {_TRIGGER_RETURN!r}, welcome them back and continue from the last topic in the history.
- If the patient asks a direct question, answer it first before returning to the structured plan."""


def run_chat_session(
    patient: Patient,
    assessment: Assessment,
    llm: BaseChatModel,
    history: list[dict],
) -> None:
    """
    Run an interactive coaching chat loop.

    Parameters
    ----------
    patient    : current patient
    assessment : most recent completed assessment
    llm        : LangChain chat model
    history    : list of {"role": "user"|"assistant", "content": str} dicts
                 loaded from the database (may be empty for first session)
    """
    from backend.database.db import save_message

    system_prompt = _build_system_prompt(patient, assessment)
    is_first_session = len(history) == 0

    # Reconstruct LangChain message list from persisted history
    messages: list = [SystemMessage(content=system_prompt)]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    print(f"\n{'═' * 60}")
    print(f"  HEALTH COACH SESSION — {patient.name}")
    print(f"  Type 'exit' or 'quit' to end the session")
    print(f"{'═' * 60}\n")

    # --- Opening message -------------------------------------------------
    # Send a hidden trigger so the AI knows whether to greet or welcome back.
    # The trigger is saved to DB so future sessions see a coherent history.
    trigger_text = _TRIGGER_START if is_first_session else _TRIGGER_RETURN
    trigger_msg = HumanMessage(content=trigger_text)

    opening_response = llm.invoke(messages + [trigger_msg])
    opening_text = opening_response.content

    # Persist trigger + opening
    save_message(patient.id, "user", trigger_text)
    save_message(patient.id, "assistant", opening_text)

    # Update in-memory message list
    messages.append(trigger_msg)
    messages.append(AIMessage(content=opening_text))

    print(f"Coach: {opening_text}\n")

    # --- Main conversation loop ------------------------------------------
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSession interrupted. Your progress has been saved.\n")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "bye", "q"):
            farewell = "Take care! Everything we discussed has been saved — see you next time."
            print(f"\nCoach: {farewell}\n")
            save_message(patient.id, "user", user_input)
            save_message(patient.id, "assistant", farewell)
            break

        messages.append(HumanMessage(content=user_input))
        save_message(patient.id, "user", user_input)

        response = llm.invoke(messages)
        response_text = response.content

        messages.append(AIMessage(content=response_text))
        save_message(patient.id, "assistant", response_text)

        print(f"\nCoach: {response_text}\n")
