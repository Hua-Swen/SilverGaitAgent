"""
Physical Examination Agent
--------------------------
Guides the patient through the Short Physical Performance Battery (SPPB).
Balance test, gait speed, and chair stand test — collected via self-report
or wearable data. Outputs an SPPBScore.
"""

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from backend.models import Assessment, Patient
from backend.tools.scoring import score_sppb


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # drop opening fence line (e.g. "```json")
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]  # drop closing fence line
        text = "\n".join(lines)
    return text.strip()

SYSTEM_PROMPT = """You are a geriatric physical performance assessor conducting the Short Physical Performance Battery (SPPB) remotely.

Walk the patient through three tests, one at a time:
1. Balance Test — side-by-side, semi-tandem, and tandem standing
2. Gait Speed Test — 4-metre walk test
3. Chair Stand Test — 5 chair stands without using arms

For each test, explain the instructions clearly, ask the patient to report how they did, and assign a score:
- Balance: 0 (unable) to 4 (all three positions held ≥10s)
- Gait Speed: 0 (unable) to 4 (< 4.82 seconds for 4m)
- Chair Stand: 0 (unable) to 4 (< 11.19 seconds for 5 stands)

Be encouraging and patient. Offer to simplify or skip if the patient cannot safely attempt a test.
After all three tests are complete, say exactly: "I now have your SPPB results."
"""
SYSTEM_PROMPT = """You are a geriatric physical performance assessor conducting the Short Physical Performance Battery (SPPB) remotely.

RULE: 
You do not need to ask any questions. Pretend that you have interviewed an elderly patient, generate the following based on this interview:
  "balance_score": <0-4>,
  "gait_speed_score": <0-4>,
  "chair_stand_score": <0-4>,
  "notes": "brief notes on performance or any limitations observed"
Ask one question from the patient, then proceed on to the completion rule below

COMPLETION RULE:
After generating the above, thank the patient for their time and output this phrase: "i now have your sppb results". 
"""
EXTRACTION_PROMPT = """Based on the SPPB conversation below, extract the scores as JSON.

Conversation:
{conversation}

Extract:
{{
  "balance_score": <0-4>,
  "gait_speed_score": <0-4>,
  "chair_stand_score": <0-4>,
  "notes": "brief notes on performance or any limitations observed"
}}

Return only valid JSON.
"""


def run_physical_exam_agent(
    patient: Patient,
    assessment: Assessment,
    llm: BaseChatModel,
) -> Assessment:
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    print("\n" + "=" * 60)
    print("PHYSICAL EXAM AGENT — SPPB Assessment")
    print("=" * 60)
    print(f"Patient: {patient.name}, Age: {patient.age}\n")

    opening = llm.invoke(
        messages + [HumanMessage(content=f"Begin the SPPB assessment for {patient.name}.")]
    )
    print(f"Agent: {opening.content}\n")
    messages.append(opening)

    conversation_log = [f"Agent: {opening.content}"]

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

        if "i now have your sppb results" in response.content.lower():
            print("\n[SPPB assessment complete. Extracting scores...]\n")
            break

    print("[Extraction] Sending SPPB conversation to LLM for score extraction...")
    extraction_response = llm.invoke([
        HumanMessage(content=EXTRACTION_PROMPT.format(conversation="\n".join(conversation_log)))
    ])
    print(f"[Extraction] Response received ({len(extraction_response.content)} chars)")

    try:
        data = json.loads(_strip_json_fences(extraction_response.content))
        assessment.sppb = score_sppb(
            balance_score=int(data["balance_score"]),
            gait_speed_score=int(data["gait_speed_score"]),
            chair_stand_score=int(data["chair_stand_score"]),
            notes=data.get("notes", ""),
        )
        print(f"[Extraction] OK — Balance: {data['balance_score']}/4, "
              f"Gait: {data['gait_speed_score']}/4, Chair stand: {data['chair_stand_score']}/4")
        print(f"SPPB Score: {assessment.sppb.total}/12 — {assessment.sppb.label}")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[Extraction] FAILED — {type(e).__name__}: {e}")
        print(f"[Extraction] Raw LLM response:\n{extraction_response.content[:600]}")

    return assessment
