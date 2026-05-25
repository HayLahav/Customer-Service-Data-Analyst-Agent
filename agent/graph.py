import json
from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from agent.llm import get_llm
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from agent.memory import load_profile, save_profile
from agent.router import classify_query
from agent.tools import TOOLS

MAX_ITERATIONS = 12

_AGENT_SYSTEM = {
    "structured": (
        "You are a data analyst assistant for a customer support intent dataset. "
        "Answer questions using ONLY data retrieved from the available tools. "
        "Be precise — quote exact numbers, categories, and intent names from tool results. "
        "Never invent or estimate data."
    ),
    "unstructured": (
        "You are a data analyst assistant for a customer support intent dataset. "
        "Use tools to gather relevant examples and statistics, then synthesise them "
        "into a clear, qualitative summary. "
        "Ground every observation in actual data from the tools — never invent examples."
    ),
    "recommendation": (
        "You are in QUERY RECOMMENDER mode for a customer-support dataset analysis agent.\n\n"
        "Your mission: guide the user to their next useful query through a short dialogue.\n\n"
        "=== THREE-PHASE PROTOCOL ===\n\n"
        "PHASE 1 — SUGGEST (user just asked for a recommendation, no prior suggestion yet):\n"
        "  • Review the full conversation history and user profile for context.\n"
        "  • Identify the single most relevant and interesting follow-up query.\n"
        "  • Describe it in plain English — what it retrieves and why it is useful.\n"
        '  • End with: "Should I go ahead?" (or equivalent).\n'
        "  DO NOT call any tool in this phase.\n\n"
        "PHASE 2 — REFINE (your last message was a suggestion, user is modifying it):\n"
        "  • Acknowledge the change. Describe the updated query clearly.\n"
        '  • End with: "Should I go ahead?"\n'
        "  DO NOT call any tool in this phase.\n\n"
        "PHASE 3 — EXECUTE (user explicitly confirms: 'yes', 'go ahead', 'do it', 'sure', 'run it'):\n"
        "  • Call the appropriate tool(s) to execute the agreed query.\n"
        "  • Present the results clearly and completely.\n"
        "  ONLY call tools in this phase.\n\n"
        "=== HOW TO DETECT YOUR PHASE ===\n"
        "Look at the conversation history:\n"
        '• Your last message ended with "Should I go ahead?" → you are in PHASE 2 or 3.\n'
        "• The user confirmed → PHASE 3 (execute).\n"
        "• The user refined → PHASE 2 (update suggestion, ask again).\n"
        "• No prior suggestion from you → PHASE 1 (fresh suggestion).\n\n"
        "=== SUGGESTION GUIDELINES ===\n"
        "Draw from: categories/intents in the user profile, natural progressions "
        "(counts → examples → distributions), or unexplored areas of the dataset."
    ),
}

_FALLBACK_MESSAGE = (
    f"I've reached the maximum reasoning limit ({MAX_ITERATIONS} steps) without a complete answer. "
    "Please try rephrasing your question or splitting it into smaller parts."
)

_DECLINE_MESSAGE = (
    "I'm designed to answer questions about the customer support dataset I have access to. "
    "Your question appears to be outside that scope — I won't use my general knowledge to answer it. "
    "Feel free to ask about the dataset's categories, intents, example conversations, or statistics."
)


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # full conversation history
    query_type: str                            # structured | unstructured | out_of_scope | recommendation
    iterations: int                            # tool-call steps in the current turn
    user_profile: str                          # JSON: {"name":..., "interests":[...], "notes":[...]}


# ── Profile-update node ───────────────────────────────────────────────────────

class _ProfileData(BaseModel):
    name: Optional[str] = Field(None, description="User's first name if explicitly stated")
    interests: List[str] = Field(
        [],
        description=(
            "Dataset topics, categories, or intents the user mentions or cares about "
            "(e.g. 'REFUND', 'shipping issues', 'complaint intents')"
        ),
    )
    notes: List[str] = Field(
        [],
        description="Brief observations about user preferences or patterns (max 3 new items per turn)",
    )


_profile_llm = None


def _get_profile_llm():
    global _profile_llm
    if _profile_llm is None:
        _profile_llm = get_llm(temperature=0).with_structured_output(_ProfileData)
    return _profile_llm


def profile_update_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Runs silently after every final answer.
    Extracts new user facts from the last exchange and merges them into the
    per-session JSON profile file (separate from conversation checkpoints).
    """
    session_id = config.get("configurable", {}).get("thread_id", "default")
    existing = load_profile(session_id)

    msgs = state["messages"]
    last_human = next((m for m in reversed(msgs) if m.type == "human"), None)
    # last AI message that is NOT a tool-call step
    last_ai = next(
        (m for m in reversed(msgs) if m.type == "ai" and not getattr(m, "tool_calls", [])),
        None,
    )

    if not last_human or not last_ai:
        return {"user_profile": json.dumps(existing)}

    try:
        extracted: _ProfileData = _get_profile_llm().invoke([
            SystemMessage(content=(
                "You maintain a concise user profile for a customer-support data analysis assistant. "
                "Given the existing profile and a single conversation turn, extract any NEW facts "
                "about the user. Only include information that is explicitly stated. "
                "Return empty lists / null if there is nothing new to add."
            )),
            {"role": "human", "content": (
                f"Existing profile:\n{json.dumps(existing, indent=2)}\n\n"
                f"User said: {last_human.content}\n"
                f"Agent replied: {last_ai.content[:500]}\n\n"
                "Extract NEW facts (not already in the existing profile):"
            )},
        ])
        updated = {
            "name": extracted.name or existing.get("name"),
            # dict.fromkeys preserves insertion order and deduplicates
            "interests": list(
                dict.fromkeys(existing.get("interests", []) + extracted.interests)
            )[:10],
            "notes": list(
                dict.fromkeys(existing.get("notes", []) + extracted.notes)
            )[:8],
        }
    except Exception:
        updated = existing

    save_profile(session_id, updated)
    return {"user_profile": json.dumps(updated)}


# ── Router node ───────────────────────────────────────────────────────────────

def router_node(state: AgentState) -> dict:
    user_query = next(
        (m.content for m in reversed(state["messages"]) if m.type == "human"), ""
    )
    # Pass the last non-tool AI message so the router can detect whether we are
    # in the middle of a recommendation dialogue (pending "Should I go ahead?").
    last_ai = next(
        (m.content for m in reversed(state["messages"])
         if m.type == "ai" and not getattr(m, "tool_calls", [])),
        "",
    )
    result = classify_query(user_query, last_ai_message=last_ai)
    print(f"\n[Router] {result.category.upper()} — {result.reasoning}")
    return {"query_type": result.category}


# ── Agent node ────────────────────────────────────────────────────────────────

def _build_agent_node(llm_with_tools):
    def agent_node(state: AgentState) -> dict:
        if state["iterations"] >= MAX_ITERATIONS:
            return {"messages": [AIMessage(content=_FALLBACK_MESSAGE)]}

        query_type = state.get("query_type", "structured")

        # Build user-profile text for the system prompt
        try:
            profile = json.loads(state.get("user_profile", "{}"))
            parts = []
            if profile.get("name"):
                parts.append(f"Name: {profile['name']}")
            if profile.get("interests"):
                parts.append(f"Topics of interest: {', '.join(profile['interests'])}")
            if profile.get("notes"):
                parts.append(f"Observations: {'; '.join(profile['notes'])}")
            profile_section = ("\n\nUser profile (from past conversations):\n" + "\n".join(parts)) if parts else ""
        except Exception:
            profile_section = ""

        system_content = (
            _AGENT_SYSTEM.get(query_type, _AGENT_SYSTEM["structured"])
            + profile_section
            + (
                '\n\nIf asked "What do you remember about me?" or similar, '
                "answer based on the user profile above. If the profile is empty, say so honestly."
            )
        )

        response = llm_with_tools.invoke(
            [SystemMessage(content=system_content)] + state["messages"]
        )
        return {"messages": [response], "iterations": state["iterations"] + 1}

    return agent_node


# ── Decline node ──────────────────────────────────────────────────────────────

def decline_node(state: AgentState) -> dict:
    return {"messages": [AIMessage(content=_DECLINE_MESSAGE)]}


# ── Routing functions ─────────────────────────────────────────────────────────

def _route_after_router(state: AgentState) -> str:
    return state["query_type"]


def _route_after_agent(state: AgentState) -> str:
    if state["iterations"] >= MAX_ITERATIONS:
        return "profile_update"
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "profile_update"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(model: str | None = None, checkpointer=None):
    """
    Compile the LangGraph agent.

    Parameters
    ----------
    model       : Model name override.  Pass None to use the provider default
                  (Nebius: Llama-3.3-70B-Instruct, OpenAI: gpt-4o-mini).
    checkpointer: LangGraph BaseCheckpointSaver instance (e.g. SqliteSaver).
                  Pass None for a stateless graph (no persistence).
    """
    llm = get_llm(model=model, temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    agent_node = _build_agent_node(llm_with_tools)
    tool_node = ToolNode(TOOLS)

    g = StateGraph(AgentState)
    g.add_node("router", router_node)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.add_node("decline", decline_node)
    g.add_node("profile_update", profile_update_node)

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "structured": "agent",
            "unstructured": "agent",
            "recommendation": "agent",
            "out_of_scope": "decline",
        },
    )
    g.add_conditional_edges(
        "agent",
        _route_after_agent,
        {"tools": "tools", "profile_update": "profile_update"},
    )
    g.add_edge("tools", "agent")
    g.add_edge("decline", "profile_update")
    g.add_edge("profile_update", END)

    return g.compile(checkpointer=checkpointer)
