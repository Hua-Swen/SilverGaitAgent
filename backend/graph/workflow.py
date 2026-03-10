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
    → save_assessment_node          ← persists assessment to SQLite
    → chat_node                     ← interactive health coach with tool calling
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
    Merge reducer for Assessment.  Non-None fields from `update` are layered
    on top of `current` without clobbering existing values.
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
    _print_node_banner("1 of 6", "HISTORY AGENT", "Collecting functional history → CFS score + Katz ADL score")
    t = time.time()
    assessment = run_history_agent(
        patient=state["patient"],
        assessment=state["assessment"],
        llm=state["llm"],
    )
    _print_node_done("History Agent", time.time() - t)
    return {**state, "assessment": assessment, "completed_nodes": ["history"]}


def physical_exam_node(state: AgentState) -> AgentState:
    _print_node_banner("2 of 6", "PHYSICAL EXAM AGENT", "SPPB assessment → balance, gait speed, chair stand scores")
    t = time.time()
    assessment = run_physical_exam_agent(
        patient=state["patient"],
        assessment=state["assessment"],
        llm=state["llm"],
    )
    _print_node_done("Physical Exam Agent", time.time() - t)
    return {**state, "assessment": assessment, "completed_nodes": ["physical_exam"]}


def contributing_conditions_node(state: AgentState) -> AgentState:
    _print_node_banner("3 of 6", "CONTRIBUTING CONDITIONS AGENT", "Screening cognitive, mood, sleep, and social isolation risk")
    t = time.time()
    assessment = run_contributing_conditions_agent(
        patient=state["patient"],
        assessment=state["assessment"],
        llm=state["llm"],
    )
    _print_node_done("Contributing Conditions Agent", time.time() - t)
    return {**state, "assessment": assessment, "completed_nodes": ["contributing_conditions"]}


def frailty_detection_node(state: AgentState) -> AgentState:
    _print_node_banner("4 of 6", "FRAILTY DETECTION AGENT", "Combining scores → frailty tier classification")
    t = time.time()
    assessment = run_frailty_detection_agent(
        patient=state["patient"],
        assessment=state["assessment"],
        llm=state["llm"],
    )
    _print_node_done("Frailty Detection Agent", time.time() - t)
    return {**state, "assessment": assessment, "completed_nodes": ["frailty_detection"]}


def save_assessment_node(state: AgentState) -> AgentState:
    from backend.database.db import save_assessment
    _print_node_banner("5 of 6", "SAVE ASSESSMENT", "Persisting assessment results to SQLite")
    t = time.time()
    assessment = save_assessment(state["assessment"])
    _print_node_done("Save Assessment", time.time() - t)
    return {**state, "assessment": assessment, "completed_nodes": ["saved"]}


def chat_node(state: AgentState) -> AgentState:
    from backend.database.db import get_conversation
    from backend.agents.chat_agent import run_chat_session
    _print_node_banner("6 of 6", "HEALTH COACH", "Starting interactive coaching session with tool calling")
    history = get_conversation(state["patient"].id)
    run_chat_session(
        patient=state["patient"],
        assessment=state["assessment"],
        llm=state["llm"],
        history=history,
    )
    return {**state, "completed_nodes": ["chat"]}


# --- Build graph ---

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Assessment layer nodes
    graph.add_node("history_node", history_node)
    graph.add_node("physical_exam_node", physical_exam_node)
    graph.add_node("contributing_conditions_node", contributing_conditions_node)
    graph.add_node("frailty_detection_node", frailty_detection_node)

    # Save + chat nodes
    graph.add_node("save_assessment_node", save_assessment_node)
    graph.add_node("chat_node", chat_node)

    # Linear edges
    graph.set_entry_point("history_node")
    graph.add_edge("history_node", "physical_exam_node")
    graph.add_edge("physical_exam_node", "contributing_conditions_node")
    graph.add_edge("contributing_conditions_node", "frailty_detection_node")
    graph.add_edge("frailty_detection_node", "save_assessment_node")
    graph.add_edge("save_assessment_node", "chat_node")
    graph.add_edge("chat_node", END)

    return graph


def run_full_assessment(patient: Patient, llm: BaseChatModel) -> None:
    """
    Run the complete frailty assessment workflow for a patient,
    including saving the assessment and launching the coaching chat.
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
