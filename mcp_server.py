#!/usr/bin/env python3
"""
FastMCP server that exposes customer-support dataset tools over the
Model Context Protocol (MCP).

Transports
----------
stdio (default — Claude Desktop):
    python mcp_server.py

Streamable-HTTP (remote / curl / Python client):
    python mcp_server.py --http            # binds 0.0.0.0:8000
    python mcp_server.py --http --port 9000
"""

import argparse
from typing import Literal, Optional

from fastmcp import FastMCP

mcp = FastMCP(
    name="Customer Support Dataset",
    instructions=(
        "Tools for exploring a customer-support intent dataset "
        "(26 000+ labelled conversations across 11 categories). "
        "Use these tools to look up categories, intents, example conversations, "
        "counts, and keyword searches."
    ),
)


# ── helpers (imported lazily so the server starts fast) ──────────────────────

def _df():
    from agent.dataset import get_dataframe
    return get_dataframe()


# ── Tool 1: list all categories ───────────────────────────────────────────────

@mcp.tool()
def get_all_categories() -> str:
    """
    Return every category in the dataset with its record count.

    Use this as the first step when you want to know what high-level topics
    exist (e.g. ACCOUNT, SHIPPING, REFUND, FEEDBACK …).
    """
    df = _df()
    counts = df["category"].value_counts()
    lines = [f"  {cat}: {cnt} records" for cat, cnt in counts.items()]
    return (
        "Categories in the dataset:\n"
        + "\n".join(lines)
        + f"\n\nTotal: {len(df):,} records across {len(counts)} categories."
    )


# ── Tool 2: count records with optional filters ───────────────────────────────

@mcp.tool()
def count_records(
    category: Optional[str] = None,
    intent: Optional[str] = None,
) -> str:
    """
    Count records in the dataset, with optional category and/or intent filters.

    Examples
    --------
    count_records()                            → total dataset size
    count_records(category="REFUND")           → all refund records
    count_records(intent="cancel_order")       → all cancel-order records
    count_records(category="ORDER", intent="cancel_order")
    """
    df = _df()
    label_parts: list[str] = []
    if category:
        df = df[df["category"].str.upper() == category.upper()]
        label_parts.append(f"category='{category.upper()}'")
    if intent:
        df = df[df["intent"].str.lower() == intent.lower()]
        label_parts.append(f"intent='{intent.lower()}'")
    label = " AND ".join(label_parts) if label_parts else "the entire dataset"
    return f"Record count for {label}: {len(df):,}"


# ── Tool 3: sample example conversations ──────────────────────────────────────

@mcp.tool()
def get_examples(
    category: Optional[str] = None,
    intent: Optional[str] = None,
    n: int = 5,
) -> str:
    """
    Return N sample conversations (customer message + agent response).

    Parameters
    ----------
    category : Filter by category, case-insensitive (e.g. "SHIPPING").
               Leave blank to include all categories.
    intent   : Filter by intent, case-insensitive (e.g. "cancel_order").
               Leave blank to include all intents.
    n        : Number of examples to return (1–20, default 5).

    Examples
    --------
    get_examples(category="REFUND", n=3)
    get_examples(intent="track_refund", n=10)
    get_examples()   → 5 random records from the whole dataset
    """
    n = max(1, min(n, 20))
    df = _df()
    if category:
        df = df[df["category"].str.upper() == category.upper()]
    if intent:
        df = df[df["intent"].str.lower() == intent.lower()]
    if df.empty:
        return "No records match the specified filters."
    sample = df.sample(min(n, len(df)), random_state=42)
    parts = []
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        parts.append(
            f"[{i}] category={row['category']}  intent={row['intent']}\n"
            f"  Customer : {row['instruction']}\n"
            f"  Agent    : {row['response']}"
        )
    header = f"Showing {len(sample)} of {len(df):,} matching records:\n\n"
    return header + "\n\n".join(parts)


# ── Tool 4: intent distribution within a category ─────────────────────────────

@mcp.tool()
def get_intent_distribution(category: str) -> str:
    """
    Return the count and percentage of each intent inside a category.

    Use this for "what is the breakdown of X?" or "distribution of intents in Y"
    questions.

    Parameters
    ----------
    category : Category name, case-insensitive (e.g. "ACCOUNT", "ORDER").
    """
    df = _df()
    subset = df[df["category"].str.upper() == category.upper()]
    if subset.empty:
        available = ", ".join(sorted(df["category"].unique()))
        return f"No category '{category}' found. Available: {available}"
    total = len(subset)
    counts = subset["intent"].value_counts()
    lines = [
        f"  {intent}: {cnt:,} records ({cnt / total * 100:.1f}%)"
        for intent, cnt in counts.items()
    ]
    return (
        f"Intent distribution in '{category.upper()}' (total: {total:,} records):\n"
        + "\n".join(lines)
    )


# ── Tool 5: keyword search ─────────────────────────────────────────────────────

@mcp.tool()
def search_examples(
    query: str,
    n: int = 5,
    search_in: Literal["instruction", "response", "both"] = "both",
) -> str:
    """
    Search the dataset for conversations that contain specific keywords.

    Parameters
    ----------
    query     : Keyword(s) to search for (case-insensitive).
    n         : Number of matching examples to return (1–20, default 5).
    search_in : Which text field to search —
                "instruction" = customer messages only,
                "response"    = agent replies only,
                "both"        = either field (default).

    Examples
    --------
    search_examples("money back", n=3)
    search_examples("late delivery", search_in="instruction")
    search_examples("we apologise", search_in="response", n=10)
    """
    n = max(1, min(n, 20))
    df = _df()
    q = query.lower()
    if search_in == "instruction":
        mask = df["instruction"].str.lower().str.contains(q, na=False)
    elif search_in == "response":
        mask = df["response"].str.lower().str.contains(q, na=False)
    else:
        mask = (
            df["instruction"].str.lower().str.contains(q, na=False)
            | df["response"].str.lower().str.contains(q, na=False)
        )
    matches = df[mask]
    if matches.empty:
        return f"No records found containing '{query}'."
    sample = matches.sample(min(n, len(matches)), random_state=42)
    parts = []
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        parts.append(
            f"[{i}] category={row['category']}  intent={row['intent']}\n"
            f"  Customer : {row['instruction']}\n"
            f"  Agent    : {row['response']}"
        )
    header = f"Found {len(matches):,} records matching '{query}'. Showing {len(sample)}:\n\n"
    return header + "\n\n".join(parts)


# ── Entry point ────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Customer Support Dataset MCP Server")
    p.add_argument(
        "--http",
        action="store_true",
        help="Run as a streamable-HTTP server instead of stdio (default)",
    )
    p.add_argument("--host", default="0.0.0.0", help="HTTP bind host (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8000, help="HTTP bind port (default: 8000)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.http:
        print(f"Starting MCP server on http://{args.host}:{args.port}/mcp")
        mcp.run("http", host=args.host, port=args.port)
    else:
        # stdio — used by Claude Desktop and MCP Inspector
        mcp.run()
