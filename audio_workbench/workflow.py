from __future__ import annotations

from typing import Any, TypedDict
from . import core

class MixState(TypedDict, total=False):
    project_root: str
    render_sha: str
    next_check: str
    finalizable: bool

def build_workflow():
    """Optional LangGraph wrapper. SQLite remains the source of truth."""
    try:
        from langgraph.graph import StateGraph, START, END
    except ImportError as exc:
        raise RuntimeError("Install optional dependency: langgraph") from exc

    def inspect(state: MixState):
        task=core.next_task(state["project_root"],state["render_sha"])
        return {"next_check":task.get("check",""),"finalizable":task.get("status")=="complete"}

    def route(state: MixState):
        return "done" if state.get("finalizable") else "needs_check"

    g=StateGraph(MixState)
    g.add_node("inspect",inspect)
    g.add_node("needs_check",lambda s:s)
    g.add_node("done",lambda s:s)
    g.add_edge(START,"inspect")
    g.add_conditional_edges("inspect",route,{"needs_check":"needs_check","done":"done"})
    g.add_edge("needs_check",END); g.add_edge("done",END)
    return g.compile()
