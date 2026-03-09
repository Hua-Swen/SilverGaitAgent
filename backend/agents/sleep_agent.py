"""
Sleep Agent
-----------
Provides sleep hygiene education and behavioral interventions
based on the contributing conditions sleep risk score.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from backend.models import Assessment, Patient


def run_sleep_agent(
    patient: Patient,
    assessment: Assessment,
    llm: BaseChatModel,
) -> Assessment:
    print("\n" + "=" * 60)
    print("SLEEP AGENT — Sleep Hygiene & Intervention Plan")
    print("=" * 60)

    sleep_risk = "unknown"
    sleep_notes = ""
    if assessment.contributing:
        sleep_risk = assessment.contributing.sleep_risk
        sleep_notes = assessment.contributing.notes or ""

    prompt = f"""You are a geriatric sleep specialist helping {patient.name}, age {patient.age}.

Their sleep risk level from the assessment: {sleep_risk}
Additional context: {sleep_notes}

Create a personalized sleep improvement plan that includes:

**Understanding Your Sleep**
- Why sleep quality changes with age (brief, reassuring explanation)
- How poor sleep worsens frailty and physical performance

**Sleep Hygiene Recommendations** (tailored to risk level {sleep_risk})
- 5-7 specific, actionable tips for improving sleep quality
- Evening wind-down routine (step by step)
- Morning light exposure guidance

**Behavioral Interventions**
- If risk is moderate or high: include brief CBT-I (Cognitive Behavioral Therapy for Insomnia) techniques
  - Sleep restriction basics
  - Stimulus control instructions
  - Relaxation techniques (progressive muscle relaxation or breathing)

**When to Seek Help**
- Signs that warrant speaking to a doctor (sleep apnoea symptoms, severe insomnia)

Use warm, practical language. Format with clear headers. Aim for 300-400 words.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    assessment.sleep_plan = response.content

    print(response.content)
    return assessment
