"""
Streamlit chat UI for the Customer Support Dataset Agent.

Run:
    streamlit run streamlit_app.py

Requirements: OPENAI_API_KEY must be set in the environment.
"""

import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st

# ── Page config (must be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="CS Dataset Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _render_step(step: dict) -> None:
    """Render one reasoning step (router / tool call / observation)."""
    kind = step["type"]
    if kind == "router":
        icons = {"structured": "📊", "unstructured": "📝", "out_of_scope": "🚫", "recommendation": "💡"}
        icon = icons.get(step.get("query_type", ""), "🤔")
        st.markdown(f"{icon} **Router:** query classified as `{step.get('query_type', '?')}`")
    elif kind == "tool_call":
        args_str = ", ".join(f"{k}={v!r}" for k, v in step.get("args", {}).items())
        st.markdown(f"🔧 **Tool call:** `{step['name']}({args_str})`")
    elif kind == "observation":
        content = step.get("content", "")
        if len(content) > 500:
            content = content[:500] + "\n… (truncated)"
        st.markdown(f"📋 **`{step['name']}`** returned:")
        st.code(content, language="text")


# ── Cached graph (app-wide singleton) ─────────────────────────────────────────

@st.cache_resource(show_spinner="Loading agent and dataset…")
def _get_graph():
    """
    Build the LangGraph agent with a SqliteSaver checkpointer.
    Called once per Streamlit server process — the connection stays open.
    """
    import sys
    root = str(Path(__file__).parent)
    if root not in sys.path:
        sys.path.insert(0, root)

    from agent.graph import build_graph
    from langgraph.checkpoint.sqlite import SqliteSaver
    import sqlite3

    db_path = str(Path(__file__).parent / "data" / "checkpoints.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    return build_graph(checkpointer=saver)


# ── API key guard ─────────────────────────────────────────────────────────────

if not os.getenv("NEBIUS_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    st.error(
        "**No LLM provider key found.**  \n\n"
        "Set one of the following before launching Streamlit:  \n"
        "```\n# Nebius (Llama 3.3 70B)\nset NEBIUS_API_KEY=<your-key>\n\n"
        "# OpenAI\nset OPENAI_API_KEY=sk-...\n```"
    )
    st.stop()


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Session")

    session_id: str = st.text_input(
        "Session ID",
        value=st.session_state.get("active_session", "default"),
        help=(
            "Same session ID = same conversation history and user profile. "
            "Change the ID to start a new session or resume a previous one."
        ),
    )

    col_switch, col_clear = st.columns(2)
    with col_switch:
        if st.button("▶ Load", use_container_width=True, help="Switch to this session"):
            if session_id != st.session_state.get("active_session"):
                st.session_state.active_session = session_id
                st.session_state.display_messages = []
            st.rerun()
    with col_clear:
        if st.button("🗑 Clear", use_container_width=True, help="Clear display (keeps server history)"):
            st.session_state.display_messages = []
            st.rerun()

    # ── User profile ───────────────────────────────────────────────────────────
    from agent.memory import load_profile

    profile = load_profile(st.session_state.get("active_session", session_id))
    has_profile = any(v for v in profile.values() if v)
    if has_profile:
        st.divider()
        st.subheader("👤 User Profile")
        if profile.get("name"):
            st.write(f"**Name:** {profile['name']}")
        if profile.get("interests"):
            st.write("**Interests:**")
            for item in profile["interests"][:6]:
                st.caption(f"• {item}")
        if profile.get("notes"):
            st.write("**Notes:**")
            for note in profile["notes"][:4]:
                st.caption(f"• {note}")

    # ── Example queries ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("💡 Try these")
    examples = [
        "What categories exist in the dataset?",
        "How many refund requests did we get?",
        "Show me 3 examples from SHIPPING",
        "Distribution of intents in ACCOUNT",
        "Summarise the FEEDBACK category",
        "How do agents respond to cancellations?",
        "Show me examples of people wanting their money back",
        "What do you remember about me?",
        "What should I query next?",        # recommendation demo
        "Who won the Champions League?",    # out-of-scope demo
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
            st.session_state.prefill_query = ex

    st.divider()
    st.caption("History persists via SqliteSaver · Profile saved per session as JSON")


# ── Session-state initialisation ──────────────────────────────────────────────

if "active_session" not in st.session_state:
    st.session_state.active_session = session_id
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

# Sync active session with sidebar input when user edits and presses Enter
# (without clicking Load — handles the case where they just type and submit)
if st.session_state.active_session != session_id and not st.session_state.get("prefill_query"):
    st.session_state.active_session = session_id
    st.session_state.display_messages = []

active_session = st.session_state.active_session


# ── Main chat area ─────────────────────────────────────────────────────────────

from agent.llm import active_provider
st.title("🤖 Customer Support Dataset Agent")
st.caption(
    f"Session: **{active_session}** · "
    "26,872 labelled customer-support conversations · "
    f"Powered by LangGraph + {active_provider()}"
)

# Replay persisted display messages
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        steps = msg.get("reasoning_steps", [])
        if steps:
            with st.expander("🔍 Reasoning steps", expanded=False):
                for step in steps:
                    _render_step(step)
                    st.divider()
        st.markdown(msg["content"])

# Welcome hint for empty sessions
if not st.session_state.display_messages:
    st.info(
        "👋 Ask me anything about the customer-support dataset — "
        "categories, intents, example conversations, statistics, or patterns. "
        "Pick a query from the sidebar or type your own below."
    )

# ── Query input ───────────────────────────────────────────────────────────────

# Sidebar example buttons set prefill_query; chat_input takes priority if typed
prefill: str | None = st.session_state.pop("prefill_query", None)
query: str | None = st.chat_input("Ask something about the dataset…") or prefill

if query:
    # ── Show user message ──────────────────────────────────────────────────
    st.session_state.display_messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # ── Run agent and stream reasoning ────────────────────────────────────
    graph = _get_graph()
    config = {"configurable": {"thread_id": active_session}}

    profile_data = load_profile(active_session)
    initial_input = {
        "messages": [{"role": "human", "content": query}],
        "iterations": 0,
        "user_profile": json.dumps(profile_data),
    }

    reasoning_steps: list[dict] = []
    final_answer = ""

    with st.chat_message("assistant"):
        # st.status: shows a live spinner with expandable reasoning while the
        # agent works, then collapses to "Done" when finished.
        with st.status("Agent is thinking…", expanded=True) as status:
            for event in graph.stream(
                initial_input, config=config, stream_mode="updates"
            ):
                for node_name, node_output in event.items():
                    msgs = node_output.get("messages", [])

                    if node_name == "router":
                        qt = node_output.get("query_type", "unknown")
                        step = {"type": "router", "query_type": qt}
                        reasoning_steps.append(step)
                        _render_step(step)
                        st.divider()

                    elif node_name in ("agent", "decline"):
                        for msg in msgs:
                            tool_calls = getattr(msg, "tool_calls", None)
                            if tool_calls:
                                for tc in tool_calls:
                                    step = {
                                        "type": "tool_call",
                                        "name": tc["name"],
                                        "args": tc["args"],
                                    }
                                    reasoning_steps.append(step)
                                    _render_step(step)
                                    st.divider()
                            elif msg.content:
                                final_answer = msg.content

                    elif node_name == "tools":
                        for msg in msgs:
                            content = getattr(msg, "content", "")
                            name = getattr(msg, "name", "tool")
                            step = {
                                "type": "observation",
                                "name": name,
                                "content": content,
                            }
                            reasoning_steps.append(step)
                            _render_step(step)
                            st.divider()

            # Update status label once complete
            status.update(
                label="✅ Reasoning complete — click to expand",
                state="complete",
                expanded=False,
            )

        # Final answer appears below the collapsed reasoning status
        if final_answer:
            st.markdown(final_answer)

    # Save to display history so it survives reruns
    st.session_state.display_messages.append({
        "role": "assistant",
        "content": final_answer,
        "reasoning_steps": reasoning_steps,
    })

    # Rerun to refresh sidebar profile after profile_update_node may have run
    st.rerun()
