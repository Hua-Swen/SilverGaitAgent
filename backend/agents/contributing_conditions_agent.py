"""
Contributing Conditions Assessment Agent
-----------------------------------------
Screens for modifiable contributors to frailty:
cognitive decline, mood disorders, sleep disturbances, social isolation.
"""

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from backend.models import Assessment, Patient, ContributingConditionsScore


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # drop opening fence line (e.g. "```json")
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]  # drop closing fence line
        text = "\n".join(lines)
    return text.strip()

SYSTEM_PROMPT = """You are a geriatric care specialist screening an elderly patient for contributing conditions that worsen frailty.

Screen for each of the following domains, one at a time, using simple conversational questions:

1. Cognitive Decline — Ask about memory lapses, confusion, difficulty with familiar tasks.
2. Mood Disorders — Ask about feelings of sadness, loss of interest, anxiety, or hopelessness.
3. Sleep Disturbances — Ask about sleep duration, quality, difficulty falling/staying asleep, daytime fatigue.
4. Social Isolation — Ask about social activities, frequency of contact with family/friends, feelings of loneliness.

Be gentle and non-judgmental. Validate their feelings. Do not diagnose — only screen.
After completing all four domains, say exactly: "I have completed the contributing conditions screening."
"""

SYSTEM_PROMPT = """You are a geriatric care specialist screening an elderly patient for contributing conditions that worsen frailty.

RULE: 
You do not need to ask any questions. Pretend that you have interviewed an elderly patient, generate the following based on this interview:
  "cognitive_risk": "low" | "moderate" | "high",
  "mood_risk": "low" | "moderate" | "high",
  "sleep_risk": "low" | "moderate" | "high",
  "social_isolation_risk": "low" | "moderate" | "high",
  "notes": "brief summary of key concerns identified"
Ask one question from the patient, then proceed on to the completion rule below

COMPLETION RULE:
After generating the above, thank the patient for their time and output this phrase: "i have completed the contributing conditions screening". 
"""

EXTRACTION_PROMPT = """Based on the screening conversation below, classify each domain as "low", "moderate", or "high" risk.

Conversation:
{conversation}

Extract:
{{
  "cognitive_risk": "low" | "moderate" | "high",
  "mood_risk": "low" | "moderate" | "high",
  "sleep_risk": "low" | "moderate" | "high",
  "social_isolation_risk": "low" | "moderate" | "high",
  "notes": "brief summary of key concerns identified"
}}

Return only valid JSON.
"""


def run_contributing_conditions_agent(
    patient: Patient,
    assessment: Assessment,
    llm: BaseChatModel,
) -> Assessment:
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    print("\n" + "=" * 60)
    print("CONTRIBUTING CONDITIONS AGENT — Psychosocial Screening")
    print("=" * 60)
    print(f"Patient: {patient.name}, Age: {patient.age}\n")

    opening = llm.invoke(
        messages + [HumanMessage(content=f"Begin the contributing conditions screening for {patient.name}.")]
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

        if "i have completed the contributing conditions screening" in response.content.lower():
            print("\n[Contributing conditions screening complete. Extracting risk profile...]\n")
            break

    print("[Extraction] Sending screening conversation to LLM for risk classification...")
    extraction_response = llm.invoke([
        HumanMessage(content=EXTRACTION_PROMPT.format(conversation="\n".join(conversation_log)))
    ])
    print(f"[Extraction] Response received ({len(extraction_response.content)} chars)")

    try:
        data = json.loads(_strip_json_fences(extraction_response.content))
        assessment.contributing = ContributingConditionsScore(
            cognitive_risk=data["cognitive_risk"],
            mood_risk=data["mood_risk"],
            sleep_risk=data["sleep_risk"],
            social_isolation_risk=data["social_isolation_risk"],
            notes=data.get("notes"),
        )
        c = assessment.contributing
        print(f"[Extraction] OK — Cognitive: {c.cognitive_risk} | Mood: {c.mood_risk} | "
              f"Sleep: {c.sleep_risk} | Social isolation: {c.social_isolation_risk}")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[Extraction] FAILED — {type(e).__name__}: {e}")
        print(f"[Extraction] Raw LLM response:\n{extraction_response.content[:600]}")

    return assessment
