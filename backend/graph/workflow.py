"""
LangGraph Workflow — SilverGait Agent Orchestration
-----------------------------------------------------
Defines the multi-agent graph for frailty assessment and management.

Graph flow:
  START
    → history_node
    → physical_exam_node
    → contributing_conditions_node
    → frailty_detection_node
    → management_router_node          ← LLM decides which agents are needed
    → [conditional] management nodes (education, exercise, sleep, monitoring)
  END
"""

from typing import TypedDict, Annotated
import operator
import time

from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseChatModel

from backend.models import Patient, Assessment
from backend.agents import (
    run_history_agent,
    run_physical_exam_agent,
    run_contributing_conditions_agent,
    run_frailty_detection_agent,
    run_management_router_agent,
    run_physical_education_agent,
    run_exercise_agent,
    run_sleep_agent,
    run_monitoring_agent,
)


# --- Progress helpers ---

def _print_node_banner(step: str, label: str, description: str) -> None:
    print(f"\n{'─' * 60}", flush=True)
    print(f"  GRAPH STEP {step}  |  {label}", flush=True)
    print(f"  {description}", flush=True)
    print(f"{'─' * 60}", flush=True)


def _print_node_done(label: str, elapsed: float) -> None:
    print(f"[✓] {label} completed in {elapsed:.1f}s", flush=True)


# --- Reducers ---

def _merge_assessment(current: Assessment, update: Assessment) -> Assessment:
    """
    Merge reducer for Assessment.  Safe for parallel management nodes: each
    node only populates its own field (education_plan / exercise_plan / etc.),
    so non-None fields from `update` are layered on top of `current` without
    clobbering work done by sibling nodes.
    """
    merged = current.model_dump()
    for k, v in update.model_dump().items():
        if v is not None:
            merged[k] = v
    return Assessment(**merged)


# --- Graph state ---

class AgentState(TypedDict):
    patient: Patient
    assessment: Annotated[Assessment, _merge_assessment]
    llm: BaseChatModel
    completed_nodes: Annotated[list[str], operator.add]


# --- Node functions ---

def history_node(state: AgentState) -> AgentState:
    _print_node_banner("1 of 9", "HISTORY AGENT", "Collecting functional history → CFS score + Katz ADL score")
    t = time.time()
    assessment = run_history_agent(
        patient=state["patient"],
        assessment=state["assessment"],
        llm=state["llm"],
    )
    _print_node_done("History Agent", time.time() - t)
    return {**state, "assessment": assessment, "completed_nodes": ["history"]}


def physical_exam_node(state: AgentState) -> AgentState:
    _print_node_banner("2 of 9", "PHYSICAL EXAM AGENT", "SPPB assessment → balance, gait speed, chair stand scores")
    t = time.time()
    assessment = run_physical_exam_agent(
        patient=state["patient"],
        assessment=state["assessment"],
        llm=state["llm"],
    )
    _print_node_done("Physical Exam Agent", time.time() - t)
    return {**state, "assessment": assessment, "completed_nodes": ["physical_exam"]}


def contributing_conditions_node(state: AgentState) -> AgentState:
    _print_node_banner("3 of 9", "CONTRIBUTING CONDITIONS AGENT", "Screening cognitive, mood, sleep, and social isolation risk")
    t = time.time()
    assessment = run_contributing_conditions_agent(
        patient=state["patient"],
        assessment=state["assessment"],
        llm=state["llm"],
    )
    _print_node_done("Contributing Conditions Agent", time.time() - t)
    return {**state, "assessment": assessment, "completed_nodes": ["contributing_conditions"]}


def frailty_detection_node(state: AgentState) -> AgentState:
    _print_node_banner("4 of 9", "FRAILTY DETECTION AGENT", "Combining scores → frailty tier classification + routing decision")
    t = time.time()
    assessment = run_frailty_detection_agent(
        patient=state["patient"],
        assessment=state["assessment"],
        llm=state["llm"],
    )
    _print_node_done("Frailty Detection Agent", time.time() - t)
    return {**state, "assessment": assessment, "completed_nodes": ["frailty_detection"]}


def management_router_node(state: AgentState) -> AgentState:
    _print_node_banner("5 of 9", "MANAGEMENT ROUTER AGENT", "Reviewing clinical summary → selecting personalised management pathways")
    t = time.time()
    assessment = run_management_router_agent(
        patient=state["patient"],
        assessment=state["assessment"],
        llm=state["llm"],
    )
    _print_node_done("Management Router Agent", time.time() - t)
    return {**state, "assessment": assessment, "completed_nodes": ["management_router"]}


def education_node(state: AgentState) -> dict:
    _print_node_banner("6 of 9", "PHYSICAL EDUCATION AGENT", "Generating personalised frailty education + fall prevention plan")
    t = time.time()
    assessment = run_physical_education_agent(
        patient=state["patient"],
        assessment=state["assessment"],
        llm=state["llm"],
    )
    _print_node_done("Physical Education Agent", time.time() - t)
    return {"assessment": assessment, "completed_nodes": ["education"]}


def exercise_node(state: AgentState) -> dict:
    _print_node_banner("7 of 9", "EXERCISE AGENT", "Building 4-week personalised exercise program")
    t = time.time()
    assessment = run_exercise_agent(
        patient=state["patient"],
        assessment=state["assessment"],
        llm=state["llm"],
    )
    _print_node_done("Exercise Agent", time.time() - t)
    return {"assessment": assessment, "completed_nodes": ["exercise"]}


def sleep_node(state: AgentState) -> dict:
    _print_node_banner("8 of 9", "SLEEP AGENT", "Generating sleep hygiene plan + CBT-I interventions")
    t = time.time()
    assessment = run_sleep_agent(
        patient=state["patient"],
        assessment=state["assessment"],
        llm=state["llm"],
    )
    _print_node_done("Sleep Agent", time.time() - t)
    return {"assessment": assessment, "completed_nodes": ["sleep"]}


def monitoring_node(state: AgentState) -> dict:
    _print_node_banner("9 of 9", "MONITORING AGENT", "Analysing 30-day Apple Health trends + generating follow-up plan")
    t = time.time()
    assessment = run_monitoring_agent(
        patient=state["patient"],
        assessment=state["assessment"],
        llm=state["llm"],
    )
    _print_node_done("Monitoring Agent", time.time() - t)
    return {"assessment": assessment, "completed_nodes": ["monitoring"]}


# --- Conditional routing ---

def route_management(state: AgentState) -> list[str]:
    """
    Translate management_routes set by the Management Router Agent into graph node names.
    Falls back to all four agents if the router produced no output.
    """
    chosen = state["assessment"].management_routes or ["education", "exercise", "sleep", "monitoring"]

    agent_to_node = {
        "education": "education_node",
        "exercise": "exercise_node",
        "sleep": "sleep_node",
        "monitoring": "monitoring_node",
    }

    routes = [agent_to_node[a] for a in chosen if a in agent_to_node]

    if not routes:
        routes = list(agent_to_node.values())

    readable = [r.replace("_node", "").title() for r in routes]
    all_agents = {"Education", "Exercise", "Sleep", "Monitoring"}
    skipped = all_agents - set(readable)
    print(f"\n[GRAPH] Activating: {', '.join(readable)}")
    print(f"[GRAPH] Skipped   : {', '.join(skipped) or 'none'}")

    return routes


# --- Build graph ---

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Assessment layer nodes
    graph.add_node("history_node", history_node)
    graph.add_node("physical_exam_node", physical_exam_node)
    graph.add_node("contributing_conditions_node", contributing_conditions_node)
    graph.add_node("frailty_detection_node", frailty_detection_node)

    # Management router node
    graph.add_node("management_router_node", management_router_node)

    # Management layer nodes
    graph.add_node("education_node", education_node)
    graph.add_node("exercise_node", exercise_node)
    graph.add_node("sleep_node", sleep_node)
    graph.add_node("monitoring_node", monitoring_node)

    # Assessment layer edges (sequential)
    graph.set_entry_point("history_node")
    graph.add_edge("history_node", "physical_exam_node")
    graph.add_edge("physical_exam_node", "contributing_conditions_node")
    graph.add_edge("contributing_conditions_node", "frailty_detection_node")
    graph.add_edge("frailty_detection_node", "management_router_node")

    # Conditional routing after management router
    graph.add_conditional_edges(
        "management_router_node",
        route_management,
        {
            "education_node": "education_node",
            "exercise_node": "exercise_node",
            "sleep_node": "sleep_node",
            "monitoring_node": "monitoring_node",
        },
    )

    # All management nodes → END
    graph.add_edge("education_node", END)
    graph.add_edge("exercise_node", END)
    graph.add_edge("sleep_node", END)
    graph.add_edge("monitoring_node", END)

    return graph


def run_full_assessment(patient: Patient, llm: BaseChatModel) -> Assessment:
    """
    Run the complete frailty assessment workflow for a patient.
    Returns the fully populated Assessment object.
    """
    assessment = Assessment(patient_id=patient.id)

    print(f"\n{'=' * 60}")
    print(f"  SILVERGAIT WORKFLOW STARTED")
    print(f"  Patient : {patient.name}  |  Age: {patient.age}  |  ID: {patient.id}")
    print(f"  Model   : {llm.__class__.__name__}")
    print(f"{'=' * 60}")

    graph = build_graph()
    app = graph.compile()

    initial_state: AgentState = {
        "patient": patient,
        "assessment": assessment,
        "llm": llm,
        "completed_nodes": [],
    }

    workflow_start = time.time()
    final_state = app.invoke(initial_state)
    elapsed = time.time() - workflow_start

    completed = final_state.get("completed_nodes", [])
    print(f"\n{'=' * 60}")
    print(f"  WORKFLOW COMPLETE  |  Total time: {elapsed:.1f}s")
    print(f"  Nodes run: {', '.join(completed)}")
    print(f"  Frailty tier: {final_state['assessment'].frailty_tier or 'not classified'}")
    print(f"{'=' * 60}\n")

    return final_state["assessment"]
