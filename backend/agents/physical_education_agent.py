"""
Physical Education Agent
------------------------
Educates the patient about frailty, mobility preservation, and fall prevention.
Tailored to their specific frailty tier.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from backend.models import Assessment, Patient


def run_physical_education_agent(
    patient: Patient,
    assessment: Assessment,
    llm: BaseChatModel,
) -> Assessment:
    print("\n" + "=" * 60)
    print("PHYSICAL EDUCATION AGENT — Frailty Education")
    print("=" * 60)

    tier = assessment.frailty_tier or "pre-frail"
    sppb_label = assessment.sppb.label if assessment.sppb else "unknown"
    cfs_label = assessment.cfs.label if assessment.cfs else "unknown"

    prompt = f"""You are a geriatric health educator speaking directly to {patient.name}, age {patient.age}.

Their frailty classification is: {tier}
Their physical performance level (SPPB): {sppb_label}
Their frailty severity (CFS): {cfs_label}

Write a personalized education plan covering:
1. What frailty means and why it matters (briefly, without being alarming)
2. Why staying active and maintaining muscle strength is important at their age
3. 3 practical tips for fall prevention specific to their level
4. An encouraging message about maintaining independence

Use warm, simple language appropriate for an elderly adult. Use short paragraphs.
Format with clear section headers. Aim for 300-400 words.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    assessment.education_plan = response.content

    print(response.content)
    return assessment
