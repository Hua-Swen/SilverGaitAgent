# Claude.md
## Elderly Frailty Recognition & Management AI Application

---

# 1. Application Overview

This application is an agentic AI system designed to recognize, assess, and manage frailty in elderly adults over weeks to months.

The system mirrors a structured clinical diagnostic and management approach and provides longitudinal monitoring, risk stratification, and personalized intervention strategies.

The architecture is modular and built using multiple specialized AI agents.

---

# 2. Core Objectives

1. Detect early signs of frailty.
2. Monitor progression over time.
3. Identify contributing risk factors.
4. Stratify frailty risk using validated tools.
5. Provide personalized intervention strategies.
6. Support elderly adults in maintaining independence.

---

# 3. System Architecture (Agent-Based Design)

The system consists of two major layers:

- Assessment Layer
- Management Layer

A central Risk Stratification Engine coordinates outputs across agents.

---

# 4. Assessment Layer

## 4.1 History Agent

Purpose:
Collect structured history and detect functional decline.

Responsibilities:
- Assess energy levels and fatigue
- Detect new or worsening fatigue
- Identify reduced activity levels (step tracker integration)
- Evaluate functional status:
  - ADLs (Activities of Daily Living)
  - iADLs (Instrumental Activities of Daily Living)
- Perform frailty screening using:
  - Clinical Frailty Scale (CFS)
  - Katz Index

Outputs:
- Structured history summary
- CFS score
- Katz ADL score
- Longitudinal trend analysis (improving / stable / declining)

---

## 4.2 Physical Examination Agent

Purpose:
Conduct frailty screening using SPPB (Short Physical Performance Battery).

SPPB Components:
- Chair stand test
- Balance test
- Walking speed test

Integrations:
- Computer vision module using gemini for sit-to-stand detection
- Wearable sensor data for movement analysis

Outputs:
- SPPB score
- Physical performance classification
- Functional mobility profile

---

## 4.3 Contributing Conditions Assessment Agent

Purpose:
Identify modifiable contributors to frailty.

Screens for:
- Cognitive decline
- Mood disorders
- Sleep disturbances
- Social isolation

Integrations:
- Sleep tracker agent
- Questionnaire-based scoring
- Social activity tracking

Outputs:
- Risk factor profile
- Domain-specific risk scores
- Contributing factor summary

---

# 5. Risk Stratification Engine

## Frailty Detection Agent

Purpose:
Determine overall frailty risk and activate management workflows.

Logic:
- Combine CFS, Katz Index, SPPB, and contributing condition scores
- Flag high-risk individuals (e.g., CFS ≥ 7)
- Generate frailty classification:
  - Robust
  - Pre-frail
  - Frail
  - Severely frail

Outputs:
- Risk tier classification
- Alert triggers
- Personalized intervention pathway

---

# 6. Management Layer

Once frailty risk is classified, personalized management agents are activated.

---

## 6.1 Physical Education Agent

Function:
- Educate elderly adults about frailty
- Explain mobility preservation strategies
- Promote independence and fall prevention

Content Areas:
- Muscle strengthening importance
- Fall prevention strategies
- Functional independence principles

---

## 6.2 Exercise Agent

Function:
- Provide structured exercise programs
- Deliver workout videos from curated database
- Tailor recommendations based on SPPB and frailty level

Includes:
- Strength training
- Balance training
- Mobility exercises
- Progressive tracking

---

## 6.3 Sleep Agent

Function:
- Provide sleep hygiene education
- Interpret sleep tracker data
- Offer behavioral sleep interventions

Includes:
- Sleep routine coaching
- Insomnia behavior guidance
- Sleep trend monitoring

---

## 6.4 Monitoring & Follow-Up Agent

Function:
- Track longitudinal changes
- Detect deterioration over weeks to months
- Trigger reassessment or escalation

Includes:
- Trend analysis engine
- Threshold-based alerts
- Scheduled reassessments

---

# 7. Data Inputs

The system may integrate:

- Step trackers
- Wearable fitness devices
- Sleep monitors
- Questionnaire responses
- Computer vision (sit-to-stand detection)
- Manual clinician inputs

---

# 8. Workflow Summary

1. Collect structured history.
2. Perform frailty screening (CFS + Katz Index).
3. Conduct SPPB physical screening.
4. Assess contributing conditions.
5. Compute frailty risk tier.
6. Activate personalized management agents.
7. Monitor changes longitudinally.
8. Escalate care if deterioration detected.

---

# 9. Design Philosophy

- Modular agentic architecture
- Clinically grounded assessment tools
- Longitudinal monitoring over time
- Preventive and proactive care
- Explainable outputs
- Scalable to clinician dashboard integration

---

# 10. Future Extensions

- Clinician-facing dashboard
- Predictive frailty progression modeling
- EHR integration
- Automated referral system
- Multi-agent orchestration using LLM frameworks