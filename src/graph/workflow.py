
"""
Purpose:
    Build the LangGraph workflow for candidate evaluation, HR approval,
    and approved-candidate email delivery.

A persistent checkpointer should be supplied by the application so the
graph can pause for HR input and resume the same workflow execution.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from src.graph.nodes import (
    record_rejection,
    request_hr_approval,
    retrieve_and_evaluate_candidates,
    send_approved_candidate_emails,
)
from src.graph.state import RecruitmentState


def route_after_retrieval(state: RecruitmentState) -> str:
    """Request HR approval only when candidate results are available."""

    if state.get("candidates"):
        return "request_approval"

    return "finish"


def route_after_approval(state: RecruitmentState) -> str:
    """Route according to the decision returned by HR."""

    decision = state.get("hr_decision", {}).get("decision")

    if decision == "approve":
        return "send_emails"

    return "rejected"


def build_recruitment_graph(
    checkpointer: BaseCheckpointSaver,
):
    """Compile the graph with checkpointing enabled."""

    builder = StateGraph(RecruitmentState)

    builder.add_node(
        "retrieve_and_evaluate",
        retrieve_and_evaluate_candidates,
    )
    builder.add_node("request_hr_approval", request_hr_approval)
    builder.add_node(
        "send_approved_candidate_emails",
        send_approved_candidate_emails,
    )
    builder.add_node("record_rejection", record_rejection)

    builder.add_edge(START, "retrieve_and_evaluate")

    builder.add_conditional_edges(
        "retrieve_and_evaluate",
        route_after_retrieval,
        {
            "request_approval": "request_hr_approval",
            "finish": END,
        },
    )

    builder.add_conditional_edges(
        "request_hr_approval",
        route_after_approval,
        {
            "send_emails": "send_approved_candidate_emails",
            "rejected": "record_rejection",
        },
    )

    builder.add_edge("send_approved_candidate_emails", END)
    builder.add_edge("record_rejection", END)

    return builder.compile(checkpointer=checkpointer)
