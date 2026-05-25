#!/usr/bin/env python3
"""
Customer Support Dataset Agent — interactive CLI with persistent memory.

Usage
-----
  python main.py                        # default session
  python main.py --session alice        # named, persistent session
  python main.py --session demo --model gpt-4o

Session behaviour
-----------------
  • Conversation history is persisted in data/checkpoints.db via SqliteSaver.
    Restarting with the same --session ID restores the full conversation.
  • A per-session user profile (name, interests, notes) is maintained in
    data/profiles/<session_id>.json, separately from the conversation log.
"""

import argparse
import json
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── Environment check ─────────────────────────────────────────────────────────

def _check_env() -> None:
    if not os.getenv("NEBIUS_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("Error: no LLM provider key found.")
        print("  Nebius : set NEBIUS_API_KEY=<your-key>")
        print("  OpenAI : set OPENAI_API_KEY=sk-...")
        sys.exit(1)


# ── CLI arguments ─────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Customer Support Dataset Agent with persistent memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python main.py                         # default session (persists)
  python main.py --session alice         # named session — resumable after restart
  python main.py --session demo --model gpt-4o
""",
    )
    parser.add_argument(
        "--session",
        default="default",
        metavar="ID",
        help="Session ID for conversation memory (default: 'default'). "
             "The same ID restores history across restarts.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name override.  Defaults to the provider's default: "
            "Nebius → meta-llama/Llama-3.3-70B-Instruct, OpenAI → gpt-4o-mini"
        ),
    )
    return parser.parse_args()


# ── Query runner ──────────────────────────────────────────────────────────────

def run_query(graph, query: str, config: dict, session_id: str) -> str:
    """
    Stream one query through the graph, printing each reasoning step, and
    return the final answer string.

    The user profile is loaded from the JSON file and injected into the initial
    state so the agent can reference it even on the very first turn.
    """
    from agent.memory import load_profile

    print(f"\n{'─' * 64}")
    print(f"You: {query}")
    print('─' * 64)

    profile = load_profile(session_id)
    initial_input = {
        "messages": [{"role": "human", "content": query}],
        "iterations": 0,
        # Profile injected so the agent has it immediately; also serves as
        # fallback if the checkpoint DB is ever cleared while the JSON file
        # survives (or vice-versa for a brand-new session).
        "user_profile": json.dumps(profile),
    }

    final_answer = ""

    for event in graph.stream(initial_input, config=config, stream_mode="updates"):
        for node_name, node_output in event.items():
            msgs = node_output.get("messages", [])

            if node_name == "decline":
                if msgs:
                    print(f"\n[Agent — Out of scope]\n{msgs[-1].content}")
                    final_answer = msgs[-1].content

            elif node_name == "agent":
                if msgs:
                    last = msgs[-1]
                    tool_calls = getattr(last, "tool_calls", None)
                    if tool_calls:
                        for tc in tool_calls:
                            args_str = ", ".join(
                                f"{k}={v!r}" for k, v in tc["args"].items()
                            )
                            print(f"\n[Thought → Tool] {tc['name']}({args_str})")
                    else:
                        print(f"\n[Final Answer]\n{last.content}")
                        final_answer = last.content

            elif node_name == "tools":
                for msg in msgs:
                    content = getattr(msg, "content", str(msg))
                    name = getattr(msg, "name", "tool")
                    preview = (
                        content if len(content) <= 400 else content[:400] + "\n  … (truncated)"
                    )
                    print(f"\n[Observation — {name}]\n{preview}")

            elif node_name == "profile_update":
                # Silent — profile bookkeeping in the background
                pass

    return final_answer


# ── Session banner ────────────────────────────────────────────────────────────

def _print_banner(session_id: str, model: str) -> None:
    from agent.memory import load_profile

    print("\n" + "=" * 64)
    print("  Customer Support Dataset Agent")
    print(f"  Session : {session_id}")
    print(f"  Model   : {model}")

    profile = load_profile(session_id)
    if any(v for v in profile.values() if v):
        name_str = f" ({profile['name']})" if profile.get("name") else ""
        interests = ", ".join(profile.get("interests", [])[:5]) or "—"
        print(f"  Profile : known user{name_str}, interests: {interests}")
    else:
        print("  Profile : new user (no profile yet)")

    print("  Type your question and press Enter. 'quit' to exit.")
    print("=" * 64 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()   # parse first so --help works without API key
    _check_env()
    session_id = args.session

    from agent.llm import active_provider
    print(f"Initializing agent (session='{session_id}', provider={active_provider()})…")

    from agent.graph import build_graph
    from agent.memory import get_checkpointer

    config = {"configurable": {"thread_id": session_id}}

    with get_checkpointer() as checkpointer:
        graph = build_graph(model=args.model, checkpointer=checkpointer)
        _print_banner(session_id, args.model)

        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not query:
                continue
            if query.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break

            run_query(graph, query, config, session_id)
            print()


if __name__ == "__main__":
    main()
