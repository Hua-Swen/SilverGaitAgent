"""
History Agent
-------------
Conducts a structured functional history interview with the patient.
Outputs: history summary, CFS score, Katz ADL score.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from backend.models import Assessment, Patient, CFSScore, KatzScore
from backend.tools.scoring import score_cfs, score_katz


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # drop opening fence line (e.g. "```json")
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]  # drop closing fence line
        text = "\n".join(lines)
    return text.strip()
#1. Energy and Fatigue: Are they experiencing new or worsening exhaustion?
#2. Activity Levels: Has their physical activity or walking reduced recently?
SYSTEM_PROMPT = """You are a geriatric clinical assistant conducting a structured history intake for elderly frailty assessment.

You must gather specific information across 3 domains:
1. Basic ADLs: Do they need physical help or supervision with bathing, dressing, toileting, transferring (getting in/out of bed), continence, or feeding?
2. Instrumental ADLs (iADLs): Do they need help managing finances, medications, transport, cooking, or housework?
3. Falls: Have they had any falls, slips, or trips in the past 6 months?

CRITICAL INTERVIEW RULES:
- ONE AT A TIME: NEVER ask a list of questions. Ask exactly ONE question at a time. Wait for the patient's response.
- DO NOT ASSUME: If a patient says "I take care of myself just fine," DO NOT assume they are independent in all ADLs. You must gently probe for specifics (e.g., "That's wonderful to hear! Just to be thorough, do you need any help at all getting in and out of the shower, or with your medications?").
- COMPASSIONATE REDIRECTION: Elderly patients may share long stories or get distracted. Validate their feelings warmly, but gently guide the conversation back to the assessment.
- PLAIN LANGUAGE: Speak warmly and plainly. Avoid clinical jargon like "ADLs", "continence", or "transferring." Use everyday phrases like "getting to the bathroom" or "getting out of a chair."
- NO DIAGNOSING: You are gathering information only. Do not provide medical advice.

COMPLETION RULE:
When—and ONLY when—you have confidently gathered details covering all 3 domains, thank the patient for their time and output exactly this phrase: "[INTAKE_COMPLETE]". Do not use this phrase until the assessment is thoroughly finished.
"""

SYSTEM_PROMPT = """You are a geriatric clinical assistant conducting a structured history intake for elderly frailty assessment.

RULE: 
You do not need to ask any questions. Pretend that you have interviewed an elderly patient, generate the following based on this interview:
  "history_summary": "2-3 sentence clinical summary of functional status and key concerns",
  "cfs_score": <integer 1-9 based on the Clinical Frailty Scale>,
  "cfs_notes": "brief justification for the CFS score",
  "bathing": <true if independent, false if dependent>,
  "dressing": <true if independent, false if dependent>,
  "toileting": <true if independent, false if dependent>,
  "transferring": <true if independent, false if dependent>,
  "continence": <true if independent, false if dependent>,
  "feeding": <true if independent, false if dependent>
Ask one question from the patient, then proceed on to the completion rule below

COMPLETION RULE:
After generating the above, thank the patient for their time and output exactly this phrase: "[INTAKE_COMPLETE]". 
"""

EXTRACTION_PROMPT = """Based on the conversation history below, extract the following as JSON.

Conversation:
{conversation}

Patient name: {patient_name}

Extract:
{{
  "history_summary": "2-3 sentence clinical summary of functional status and key concerns",
  "cfs_score": <integer 1-9 based on the Clinical Frailty Scale>,
  "cfs_notes": "brief justification for the CFS score",
  "bathing": <true if independent, false if dependent>,
  "dressing": <true if independent, false if dependent>,
  "toileting": <true if independent, false if dependent>,
  "transferring": <true if independent, false if dependent>,
  "continence": <true if independent, false if dependent>,
  "feeding": <true if independent, false if dependent>
}}

Return only valid JSON with no extra text.
"""


def run_history_agent(
    patient: Patient,
    assessment: Assessment,
    llm: BaseChatModel,
    interactive: bool = True,
) -> Assessment:
    """
    Runs the history agent.

    In interactive mode: conducts a multi-turn conversation with the user (CLI).
    Returns the assessment updated with history_summary, cfs, and katz.
    """
    import json

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    print("\n" + "=" * 60)
    print("HISTORY AGENT — Functional History Interview")
    print("=" * 60)
    print(f"Patient: {patient.name}, Age: {patient.age}\n")

    # Opening message from the agent
    opening = llm.invoke(
        messages + [HumanMessage(content=f"Please greet {patient.name} and begin the history intake.")]
    )
    print(f"Agent: {opening.content}\n")
    messages.append(opening)

    conversation_log = [f"Agent: {opening.content}"]

    # Multi-turn conversation
    while True:
        user_input = input("Patient: ").strip()
        if not user_input:
            continue

        conversation_log.append(f"Patient: {user_input}")
        messages.append(HumanMessage(content=user_input))

        response = llm.invoke(messages)
        print(f"\nAgent: {response.content}\n")
        messages.append(response)
        conversation_log.append(f"Agent: {response.content}")

        # Check if the agent signals completion
        if "[intake_complete]" in response.content.lower():
            print("\n[History intake complete. Extracting structured data...]\n")
            break

    # Extract structured data from the conversation
    extraction_prompt = EXTRACTION_PROMPT.format(
        conversation="\n".join(conversation_log),
        patient_name=patient.name,
    )
    print("[Extraction] Sending conversation to LLM for structured data extraction...")
    extraction_response = llm.invoke([HumanMessage(content=extraction_prompt)])
    print(f"[Extraction] Response received ({len(extraction_response.content)} chars)")

    try:
        data = json.loads(_strip_json_fences(extraction_response.content))
        assessment.history_summary = data["history_summary"]
        assessment.cfs = score_cfs(
            score=int(data["cfs_score"]),
            notes=data.get("cfs_notes", ""),
        )
        assessment.katz = score_katz(
            bathing=data["bathing"],
            dressing=data["dressing"],
            toileting=data["toileting"],
            transferring=data["transferring"],
            continence=data["continence"],
            feeding=data["feeding"],
        )
        adl_count = sum([data.get(k, False) for k in
                         ["bathing", "dressing", "toileting", "transferring", "continence", "feeding"]])
        print(f"[Extraction] OK — CFS: {data['cfs_score']}, ADLs independent: {adl_count}/6")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[Extraction] FAILED — {type(e).__name__}: {e}")
        print(f"[Extraction] Raw LLM response:\n{extraction_response.content[:600]}")
        assessment.history_summary = "History collected but structured extraction failed."

    print(f"History Summary: {assessment.history_summary}")
    if assessment.cfs:
        print(f"CFS Score: {assessment.cfs.score} — {assessment.cfs.label}")
    if assessment.katz:
        print(f"Katz ADL Score: {assessment.katz.total}/6 — {assessment.katz.label}")

    return assessment
