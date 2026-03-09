"""
Exercise Agent
--------------
Provides a structured, personalized exercise program based on frailty tier and SPPB score.
Includes strength, balance, and mobility exercises.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from backend.models import Assessment, Patient

# Exercise intensity guidance by frailty tier
INTENSITY_GUIDE = {
    "robust": "moderate intensity — focus on maintaining and building strength",
    "pre-frail": "light to moderate — emphasize balance and functional strength",
    "frail": "gentle, seated or supported exercises — prioritize safety and consistency",
    "severely-frail": "very gentle, primarily seated or bed-based — comfort and minimal exertion",
}


def run_exercise_agent(
    patient: Patient,
    assessment: Assessment,
    llm: BaseChatModel,
) -> Assessment:
    print("\n" + "=" * 60)
    print("EXERCISE AGENT — Personalized Exercise Program")
    print("=" * 60)

    tier = assessment.frailty_tier or "pre-frail"
    sppb_total = assessment.sppb.total if assessment.sppb else 6
    sppb_label = assessment.sppb.label if assessment.sppb else "unknown"
    intensity = INTENSITY_GUIDE.get(tier, "light")

    prompt = f"""You are a geriatric physiotherapist creating a personalized exercise program for {patient.name}, age {patient.age}.

Frailty tier: {tier}
SPPB score: {sppb_total}/12 ({sppb_label})
Recommended intensity: {intensity}

Design a 4-week exercise program with the following structure:

**Week 1-2: Foundation Phase**
- 3 specific exercises (with sets, reps, and safety tips)
- Duration: aim for 20-30 minutes per session
- Frequency: 3 days per week

**Week 3-4: Progressive Phase**
- 3 exercises building on Week 1-2 (slight progression)
- Duration: 30 minutes
- Frequency: 3-4 days per week

For each exercise include:
- Name and brief description
- How to do it safely (step by step)
- Modification if it feels too hard

Also include:
- A warm-up routine (5 minutes)
- A cool-down routine (5 minutes)
- Warning signs to stop exercising immediately

Use simple, encouraging language. Format with clear headers and bullet points.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    assessment.exercise_plan = response.content

    print(response.content)
    return assessment
