#!/usr/bin/env python3
"""
Minimal FastMCP client demo for the Customer Support Dataset MCP server.

Two ways to run
---------------
1. In-process (no HTTP server needed — good for quick testing):
       python client_example.py

2. Against a running HTTP server:
       # Terminal 1: python mcp_server.py --http
       # Terminal 2: python client_example.py --url http://localhost:8000/mcp
"""

import argparse
import asyncio

from fastmcp import Client


async def demo(server) -> None:
    """Call every exposed tool and print a summary of the response."""
    async with Client(server) as client:

        # ── 1. Discover tools ─────────────────────────────────────────────
        tools = await client.list_tools()
        print("Available tools:", [t.name for t in tools])
        print()

        # ── 2. No-argument tool ───────────────────────────────────────────
        r = await client.call_tool("get_all_categories", {})
        print("=== get_all_categories ===")
        print(r.content[0].text)
        print()

        # ── 3. Count with category filter ─────────────────────────────────
        r = await client.call_tool("count_records", {"category": "REFUND"})
        print("=== count_records(category='REFUND') ===")
        print(r.content[0].text)
        print()

        # ── 4. Intent distribution ────────────────────────────────────────
        r = await client.call_tool(
            "get_intent_distribution", {"category": "ACCOUNT"}
        )
        print("=== get_intent_distribution(category='ACCOUNT') ===")
        print(r.content[0].text)
        print()

        # ── 5. Sample examples ────────────────────────────────────────────
        r = await client.call_tool(
            "get_examples", {"category": "SHIPPING", "n": 2}
        )
        print("=== get_examples(category='SHIPPING', n=2) ===")
        print(r.content[0].text[:600], "…")
        print()

        # ── 6. Keyword search ─────────────────────────────────────────────
        r = await client.call_tool(
            "search_examples", {"query": "money back", "n": 3}
        )
        print("=== search_examples(query='money back', n=3) ===")
        print(r.content[0].text[:600], "…")


def main() -> None:
    parser = argparse.ArgumentParser(description="FastMCP client demo")
    parser.add_argument(
        "--url",
        default=None,
        help=(
            "MCP server URL (e.g. http://localhost:8000/mcp). "
            "Omit to run against an in-process server (no HTTP needed)."
        ),
    )
    args = parser.parse_args()

    if args.url:
        print(f"Connecting to remote server: {args.url}\n")
        server = args.url
    else:
        print("Running against in-process server (no HTTP needed).\n")
        import mcp_server
        server = mcp_server.mcp

    asyncio.run(demo(server))


if __name__ == "__main__":
    main()
