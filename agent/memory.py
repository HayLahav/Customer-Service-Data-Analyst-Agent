"""
Persistent memory for the agent:
  - Conversation history  → LangGraph SqliteSaver (data/checkpoints.db)
  - User profile          → per-session JSON file  (data/profiles/<session_id>.json)

The two stores are intentionally separate so conversation logs and distilled
user facts can be managed, inspected, and cleared independently.
"""

import json
from contextlib import contextmanager
from pathlib import Path

_DATA_DIR = Path(__file__).parent.parent / "data"
_PROFILES_DIR = _DATA_DIR / "profiles"
_CHECKPOINTS_DB = _DATA_DIR / "checkpoints.db"


# ── User profile (JSON files) ─────────────────────────────────────────────────

def load_profile(session_id: str) -> dict:
    """Load the user profile for a session, or return an empty skeleton."""
    path = _PROFILES_DIR / f"{session_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"name": None, "interests": [], "notes": []}


def save_profile(session_id: str, profile: dict) -> None:
    """Persist the user profile to a JSON file."""
    _PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    path = _PROFILES_DIR / f"{session_id}.json"
    path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Conversation checkpointer (SQLite) ────────────────────────────────────────

@contextmanager
def get_checkpointer():
    """
    Context manager that yields a LangGraph SqliteSaver for durable conversation
    checkpointing.  Falls back to an in-memory saver (with a warning) if the
    SQLite package is somehow unavailable.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        with SqliteSaver.from_conn_string(str(_CHECKPOINTS_DB)) as saver:
            yield saver
    except ImportError:
        from langgraph.checkpoint.memory import MemorySaver
        print(
            "Warning: langgraph-checkpoint-sqlite not found. "
            "Conversation will NOT persist across restarts."
        )
        yield MemorySaver()
