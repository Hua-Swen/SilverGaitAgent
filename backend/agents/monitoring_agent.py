"""
Monitoring Agent (30-Day Trend & Deterioration Detector)
-------------------------------------------------------
Reads Apple Health XML from data/exports.xml, analyses 30 days of step and
sleep data, and writes a patient summary + clinical flags to the Assessment.
"""

import os
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import timedelta
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

from backend.models import Patient, Assessment


# ==========================================
# 1. OUTPUT SCHEMA
# ==========================================
class WearableInsights(BaseModel):
    patient_message: str = Field(
        description="A 150-word empathetic message to the patient explaining their 30-day trends and highlighting any deterioration."
    )
    clinical_handoff_notes: str = Field(
        description="A brief 2-sentence summary of the objective data to be passed to the doctors."
    )
    needs_exercise_intervention: bool = Field(
        description="True if steps are chronically low or deteriorating."
    )
    needs_sleep_intervention: bool = Field(
        description="True if sleep is chronically low (< 6 hours) or deteriorating."
    )


# ==========================================
# 2. DATA EXTRACTION & ANALYSIS
# ==========================================
def _parse_apple_health(xml_path: str) -> pd.DataFrame:
    print(f"\n[System] Loading Apple Health data from {xml_path}...")
    if not os.path.exists(xml_path):
        print(f"[Warning] Could not find {xml_path} — skipping wearable analysis.")
        return pd.DataFrame()

    tree = ET.parse(xml_path)
    root = tree.getroot()

    target_types = {
        "HKQuantityTypeIdentifierStepCount",
        "HKCategoryTypeIdentifierSleepAnalysis",
        "HKQuantityTypeIdentifierRestingHeartRate",
    }

    records = []
    for record in root.findall("Record"):
        r_type = record.attrib.get("type")
        if r_type in target_types:
            records.append({
                "type": r_type,
                "value": record.attrib.get("value"),
                "date": record.attrib.get("startDate"),
            })

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    return df


def _analyze_30_day_trends(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}

    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    latest_date = df["date"].max()
    thirty_days_ago = latest_date - timedelta(days=30)
    midpoint = latest_date - timedelta(days=15)

    df_30 = df[df["date"] >= thirty_days_ago].copy()
    df_30["day"] = df_30["date"].dt.date

    first_15 = df_30[df_30["date"] < midpoint]
    last_15 = df_30[df_30["date"] >= midpoint]

    def get_metrics(data, metric_type):
        subset = data[data["type"] == metric_type]
        if subset.empty:
            return 0
        if "Step" in metric_type:
            return round(subset.groupby("day")["value"].sum().mean(), 0)
        return round(subset["value"].mean(), 1)

    return {
        "overall_30_day_steps": get_metrics(df_30, "HKQuantityTypeIdentifierStepCount"),
        "first_15_steps": get_metrics(first_15, "HKQuantityTypeIdentifierStepCount"),
        "last_15_steps": get_metrics(last_15, "HKQuantityTypeIdentifierStepCount"),
        "overall_30_day_sleep": get_metrics(df_30, "HKCategoryTypeIdentifierSleepAnalysis"),
        "first_15_sleep": get_metrics(first_15, "HKCategoryTypeIdentifierSleepAnalysis"),
        "last_15_sleep": get_metrics(last_15, "HKCategoryTypeIdentifierSleepAnalysis"),
    }


# ==========================================
# 3. CORE AGENT LOGIC
# ==========================================
def run_monitoring_agent(patient: Patient, assessment: Assessment, llm) -> Assessment:
    print("\n" + "=" * 60)
    print("MONITORING AGENT — 30-Day Deterioration Analysis")
    print("=" * 60)

    # Locate export file relative to project root (backend/agents/ → ../../data/)
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    xml_path = os.path.join(_project_root, "data", "export.xml")
    df = _parse_apple_health(xml_path)
    trends = _analyze_30_day_trends(df)

    if not trends:
        assessment.monitoring_notes = (
            "No Apple Health data found. Wearable trend analysis could not be completed."
        )
        assessment.wearables_needs_exercise = False
        assessment.wearables_needs_sleep = False
        return assessment

    prompt = f"""You are a clinical AI analyzing 30 days of continuous wearable data for {patient.name} (Age: {patient.age}).

### 30-DAY WEARABLE DATA TRENDS
**Activity (Steps):**
- 30-Day Average: {trends.get('overall_30_day_steps', 0)} steps/day
- First 15 Days Average: {trends.get('first_15_steps', 0)} steps/day
- Last 15 Days Average: {trends.get('last_15_steps', 0)} steps/day

**Sleep:**
- 30-Day Average: {trends.get('overall_30_day_sleep', 0)} hours/night
- First 15 Days Average: {trends.get('first_15_sleep', 0)} hours/night
- Last 15 Days Average: {trends.get('last_15_sleep', 0)} hours/night

### YOUR OBJECTIVE
Analyze this data to detect if their health is chronically low (e.g., < 3000 steps or < 6 hours sleep) OR if it is deteriorating (Last 15 days are noticeably worse than the First 15 days).

Generate a structured output with:
1. A direct, empathetic message to the patient highlighting these specific trends. If things are dropping, point it out gently.
2. Clinical handoff notes.
3. True/False flags if this data warrants sending them to the Exercise or Sleep intervention agents.
"""

    print("[System] Analyzing trends and generating cross-agent flags...\n")

    structured_llm = llm.with_structured_output(WearableInsights)
    output: WearableInsights = structured_llm.invoke([HumanMessage(content=prompt)])

    # Write results into the shared Assessment
    assessment.wearables_summary = output.patient_message
    assessment.monitoring_notes = (
        f"{output.patient_message}\n\nClinical Handoff: {output.clinical_handoff_notes}"
    )
    assessment.wearables_clinical_notes = output.clinical_handoff_notes
    assessment.wearables_needs_exercise = output.needs_exercise_intervention
    assessment.wearables_needs_sleep = output.needs_sleep_intervention

    print(f"MESSAGE TO PATIENT:\n{output.patient_message}\n")
    print(f"CLINICAL HANDOFF:\n{output.clinical_handoff_notes}\n")
    print(
        f"FLAGS FOR OTHER AGENTS:\n"
        f"- Needs Exercise Agent: {output.needs_exercise_intervention}\n"
        f"- Needs Sleep Agent: {output.needs_sleep_intervention}"
    )

    return assessment
