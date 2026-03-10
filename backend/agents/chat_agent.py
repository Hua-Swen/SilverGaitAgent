"""
Post-Assessment Health Coach Chat Agent
----------------------------------------
Interactive coaching session that disseminates personalised management plans
across multiple conversation turns via tool calling.  The LLM decides which
plan to generate based on what the patient asks about.  Conversation history
is persisted to SQLite so returning patients can continue where they left off.
"""

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool

from backend.models import Patient, Assessment


# Session trigger tokens
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

    wearables_section = ""
    if assessment.wearables_summary or assessment.wearables_clinical_notes:
        wearables_section = f"""
WEARABLES DATA (from Apple Health / step tracker):
- Summary: {assessment.wearables_summary or "not available"}
- Clinical notes: {assessment.wearables_clinical_notes or "not available"}
- Needs exercise intervention: {assessment.wearables_needs_exercise}
- Needs sleep intervention: {assessment.wearables_needs_sleep}"""

    return f"""You are a compassionate health coach for {patient.name}, age {patient.age}.

FRAILTY ASSESSMENT RESULTS:
- Frailty classification: {tier}
- {cfs}
- {katz}
- {sppb}
- History summary: {history_line}
- Risk explanation: {risk_line}{wearables_section}

YOUR ROLE:
- You are a warm, supportive health coach. Use plain language; avoid clinical jargon.
- Guide the conversation gradually across multiple turns — do not overwhelm the patient.
- Suggested order of topics: (1) explain frailty results in simple terms, (2) frailty education and fall prevention, (3) exercise recommendations, (4) sleep advice if applicable, (5) monitoring schedule.
- After each topic, check in — ask how the patient feels, invite questions — before moving on.
- NEVER repeat topics already well-covered in the conversation history.
- Keep each response focused and digestible (3–5 sentences or a short list).
- When you see {_TRIGGER_START!r}, greet the patient warmly and open with their results.
- When you see {_TRIGGER_RETURN!r}, welcome them back and continue from the last topic in the history.
- If the patient asks a direct question, answer it first before returning to the structured plan.
- IMPORTANT: When the patient asks about ANY topic covered by your tools, you MUST call the relevant tool — never say you lack access. The tools generate the content on demand.

YOUR TOOLS — ALWAYS call these when the patient asks about the relevant topic:
- get_education_plan: call when the patient asks about frailty education or fall prevention
- get_exercise_plan: call when the patient asks about exercise or physical activity
- get_sleep_plan: call when the patient asks about sleep
- get_monitoring_plan: call when the patient asks about follow-up, monitoring, wearables data, step count, activity tracking, or health data
Each tool generates a personalised plan and saves it to the database automatically."""


def _make_tools(patient: Patient, assessment: Assessment, llm: BaseChatModel) -> list:
    """Build the 4 management plan tools, closed over patient/assessment/llm."""

    from backend.agents.physical_education_agent import run_physical_education_agent
    from backend.agents.exercise_agent import run_exercise_agent
    from backend.agents.sleep_agent import run_sleep_agent
    from backend.agents.monitoring_agent import run_monitoring_agent
    from backend.database.db import save_message

    @tool
    def get_education_plan() -> str:
        """Generate and return the personalised frailty education and fall prevention plan."""
        updated = run_physical_education_agent(patient=patient, assessment=assessment, llm=llm)
        assessment.education_plan = updated.education_plan
        return updated.education_plan or "Education plan could not be generated."

    @tool
    def get_exercise_plan() -> str:
        """Generate and return the personalised exercise program."""
        updated = run_exercise_agent(patient=patient, assessment=assessment, llm=llm)
        assessment.exercise_plan = updated.exercise_plan
        return updated.exercise_plan or "Exercise plan could not be generated."

    @tool
    def get_sleep_plan() -> str:
        """Generate and return the personalised sleep hygiene and CBT-I plan."""
        updated = run_sleep_agent(patient=patient, assessment=assessment, llm=llm)
        assessment.sleep_plan = updated.sleep_plan
        return updated.sleep_plan or "Sleep plan could not be generated."

    @tool
    def get_monitoring_plan() -> str:
        """Generate and return the personalised longitudinal monitoring and follow-up plan."""
        updated = run_monitoring_agent(patient=patient, assessment=assessment, llm=llm)
        assessment.monitoring_notes = updated.monitoring_notes
        return updated.monitoring_notes or "Monitoring plan could not be generated."

    return [get_education_plan, get_exercise_plan, get_sleep_plan, get_monitoring_plan]


def run_chat_session(
    patient: Patient,
    assessment: Assessment,
    llm: BaseChatModel,
    history: list[dict],
) -> None:
    """
    Run an interactive coaching chat loop with tool calling.

    Parameters
    ----------
    patient    : current patient
    assessment : most recent completed assessment
    llm        : LangChain chat model
    history    : list of {"role": "user"|"assistant", "content": str} dicts
                 loaded from the database (may be empty for first session)
    """
    from backend.database.db import save_message

    tools = _make_tools(patient, assessment, llm)
    tool_map = {t.name: t for t in tools}
    tool_llm = llm.bind_tools(tools)

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
    trigger_text = _TRIGGER_START if is_first_session else _TRIGGER_RETURN
    trigger_msg = HumanMessage(content=trigger_text)

    opening_response = tool_llm.invoke(messages + [trigger_msg])
    opening_text = opening_response.content

    save_message(patient.id, "user", trigger_text)
    save_message(patient.id, "assistant", opening_text)

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

        response = tool_llm.invoke(messages)

        # --- Handle tool calls -------------------------------------------
        while response.tool_calls:
            messages.append(response)
            for tc in response.tool_calls:
                tool_name = tc["name"]
                print(f"\n[Tool] Calling {tool_name}...", flush=True)
                tool_result = tool_map[tool_name].invoke(tc["args"])
                messages.append(
                    ToolMessage(content=tool_result, tool_call_id=tc["id"])
                )
            response = tool_llm.invoke(messages)

        response_text = response.content
        messages.append(AIMessage(content=response_text))
        save_message(patient.id, "assistant", response_text)

        print(f"\nCoach: {response_text}\n")
