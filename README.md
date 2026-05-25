# Customer Support Dataset Agent

A conversational AI agent built with **LangGraph** that lets you explore and analyse 26,872 labelled customer-support conversations through natural language — no SQL, no code.

The system combines a ReAct reasoning loop, persistent multi-session memory, a FastMCP tool server, and a Streamlit chat UI, all powered by the **Nebius API** running **nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B**.

![Streamlit UI demo](assets/demo.png)

---

## Features

| Capability | Details |
|---|---|
| **Natural language queries** | Ask in plain English — the agent picks the right tool automatically |
| **Smart query router** | Classifies every query as *structured*, *unstructured*, *recommendation*, or *out-of-scope* before acting |
| **7 dataset tools** | Categories, intents, examples, counts, distributions, keyword search, and overview |
| **Persistent conversation memory** | Full history stored in SQLite via LangGraph's `SqliteSaver` — resume any session by name |
| **User profile (semantic memory)** | Per-session JSON profile that the agent builds up over time (name, interests, patterns) |
| **Query recommender** | Interactive 3-phase dialogue: *Suggest → Refine → Execute* — the agent proposes, you approve |
| **Out-of-scope guard** | Off-topic questions (e.g. sports scores) are politely declined without using general knowledge |
| **Streamlit chat UI** | Full chat interface with live reasoning display and session switching |
| **FastMCP server** | Expose dataset tools over the Model Context Protocol for use with Claude Desktop or any MCP client |

---

## Demo

### CLI

```
You: How many refund requests did we get?

[Router] STRUCTURED — asking for a count of a specific record type
[Thought → Tool] count_records(category='REFUND')
[Observation] Record count for category='REFUND': 2,992
[Final Answer] There are 2,992 refund records in the dataset.
```

### Streamlit reasoning display

```
┌─ Agent is thinking… ────────────────────────────────────────────┐
│ 📊 Router: query classified as `structured`                     │
│ 🔧 Tool call: count_records(category='REFUND')                  │
│ 📋 count_records returned:                                      │
│   Record count for category='REFUND': 2,992                     │
└──────────────── ✅ Reasoning complete (click to expand) ────────┘

There are **2,992** refund records in the dataset.
```

### Query recommender

```
You:   What should I query next?
Agent: Based on your interest in refund data, I'd suggest looking at the intent
       distribution in the REFUND category — it shows which specific topics
       (get_refund vs check_refund_policy) dominate. Should I go ahead?
You:   I'd rather see some examples instead.
Agent: Then I'd suggest: show 5 examples from REFUND so you can read real
       customer messages and responses. Should I go ahead?
You:   Yes.
Agent: [calls get_examples and displays 5 records]
```

---

## Architecture

```
User query
    │
    ▼
┌─────────┐   query_type    ┌────────────────────────────────────┐
│  Router │ ──────────────► │           Agent (ReAct)            │
└─────────┘                 │  LangGraph StateGraph + SqliteSaver│
    │ out_of_scope           │  Conversation history persisted    │
    ▼                        └───────────┬──────────────┬─────────┘
┌─────────┐                             │ tool_calls   │ final answer
│ Decline │                             ▼              ▼
└─────────┘                    ┌──────────────┐  ┌─────────────────┐
                               │  Tool Node   │  │ Profile Update  │
                               │  (7 tools)   │  │  (JSON file)    │
                               └──────┬───────┘  └─────────────────┘
                                      │
                               ┌──────▼───────┐
                               │   Dataset    │
                               │ (26,872 rows │
                               │  cached CSV) │
                               └──────────────┘
```

---

## Dataset

[Bitext Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) — downloaded automatically from HuggingFace on first run and cached to `data/dataset.csv`.

| Stat | Value |
|---|---|
| Total records | 26,872 |
| Categories | 11 (ACCOUNT, ORDER, REFUND, SHIPPING, …) |
| Unique intents | 27 |
| Fields | `category`, `intent`, `instruction` (customer), `response` (agent) |

---

## Tools

| Tool | Description |
|---|---|
| `get_all_categories` | All categories with record counts |
| `get_intents_by_category` | Intents within a specific category |
| `get_examples` | Sample N conversations (filter by category / intent) |
| `count_records` | Exact count with optional filters |
| `get_intent_distribution` | Percentage breakdown of intents in a category |
| `search_examples` | Keyword search across customer messages and agent replies |
| `get_dataset_overview` | High-level dataset statistics |

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-username/customer-support-agent.git
cd customer-support-agent
pip install -r requirements.txt
```

### 2. Configure your API key

Copy `.env.example` to `.env` and add your key:

```bash
cp .env.example .env
```

```env
# .env
NEBIUS_API_KEY=your-key-here        # Nebius (Llama / Nemotron) — recommended
# OPENAI_API_KEY=sk-...             # OpenAI fallback
```

Get a free Nebius key at [studio.nebius.com](https://studio.nebius.com).

### 3. Run

**Streamlit UI** (recommended):
```bash
streamlit run streamlit_app.py
# Opens http://localhost:8501
```

**Interactive CLI:**
```bash
python main.py                        # default session
python main.py --session alice        # named, persistent session
python main.py --session demo --model meta-llama/Llama-3.3-70B-Instruct
```

---

## Persistent Sessions

Conversation history is stored in `data/checkpoints.db` (SQLite). Restart with the same session ID to resume exactly where you left off:

```bash
python main.py --session research
# Ask: "Show me 3 examples from REFUND"
# Ctrl-C to exit

python main.py --session research   # restart
# Ask: "What was the last thing we looked at?"  ← history is restored
```

A per-session user profile is maintained in `data/profiles/<session_id>.json`. The agent extracts your name, interests, and patterns after each turn and uses them in future responses.

---

## MCP Server

The FastMCP server exposes 5 dataset tools over the Model Context Protocol.

```bash
# stdio (for Claude Desktop / MCP Inspector)
python mcp_server.py

# HTTP (for remote clients)
python mcp_server.py --http              # http://0.0.0.0:8000/mcp
python mcp_server.py --http --port 9000
```

### Python client

```python
import asyncio
from fastmcp import Client

async def main():
    async with Client("http://localhost:8000/mcp") as client:
        result = await client.call_tool("count_records", {"category": "REFUND"})
        print(result.content[0].text)
        # → Record count for category='REFUND': 2,992

asyncio.run(main())
```

### Claude Desktop

Add to `%APPDATA%\Claude\claude_desktop_config.json` (Windows) or `~/Library/Application Support/Claude/claude_desktop_config.json` (Mac):

```json
{
  "mcpServers": {
    "customer-support-dataset": {
      "command": "python",
      "args": ["C:/path/to/project/mcp_server.py"]
    }
  }
}
```

---

## Project Structure

```
├── main.py              ← interactive CLI
├── streamlit_app.py     ← Streamlit chat UI
├── mcp_server.py        ← FastMCP server
├── client_example.py    ← MCP Python client demo
├── requirements.txt
├── .env.example
├── agent/
│   ├── llm.py           ← LLM factory (Nebius / OpenAI)
│   ├── dataset.py       ← HuggingFace dataset loader (cached locally)
│   ├── tools.py         ← 7 LangChain tools with Pydantic schemas
│   ├── router.py        ← LLM-based query classifier (4 categories)
│   ├── graph.py         ← LangGraph ReAct agent + profile-update node
│   └── memory.py        ← SqliteSaver checkpointer + JSON profile store
└── data/
    ├── dataset.csv               ← cached dataset (auto-downloaded)
    ├── checkpoints.db            ← conversation history
    └── profiles/<session>.json  ← per-session user profile
```

---

## Tech Stack

- **[LangGraph](https://github.com/langchain-ai/langgraph)** — stateful agent graph with SQLite checkpointing
- **[LangChain](https://github.com/langchain-ai/langchain)** — tool definitions, LLM integration
- **[FastMCP](https://github.com/jlowin/fastmcp)** — Model Context Protocol server
- **[Streamlit](https://streamlit.io)** — chat UI
- **[Nebius API](https://studio.nebius.com)** — `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` inference
- **[Pydantic](https://docs.pydantic.dev)** — tool input schemas
- **[pandas](https://pandas.pydata.org)** — dataset querying

---

## License

MIT
