from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from agent.llm import get_llm

_SYSTEM_PROMPT = """You are a query classifier for a customer support dataset analysis assistant.

The dataset contains labelled customer-support conversations with categories (ACCOUNT,
SHIPPING, ORDER, REFUND, FEEDBACK, etc.) and intents within each category.

Classify the user query into exactly one of four types:

**structured** — Has a concrete, data-driven answer fully retrievable from the dataset:
  "What categories exist?", "How many refund requests?", "Show me 5 SHIPPING examples",
  "Distribution of intents in ACCOUNT", "Count cancellations"

**unstructured** — Requires synthesis, summarisation, or qualitative analysis of dataset
  content, OR meta-questions about the agent's memory / user profile:
  "Summarise the FEEDBACK category", "How do agents typically respond to complaints?",
  "What do you remember about me?", "What topics have I been asking about?"

**recommendation** — The user wants a query recommendation, OR is continuing an active
  recommendation dialogue (refining or confirming a pending suggestion).

  Direct requests:
    "What should I query next?", "Suggest something interesting",
    "What else can I explore?", "What would you recommend?"

  Refinement or confirmation (ONLY when the last assistant message was a recommendation
  that ended with "Should I go ahead?" or similar):
    "I'd rather see examples instead", "Make it about SHIPPING",
    "Yes", "Go ahead", "Do it", "Sure", "Sounds good", "Not that, try refunds"

  KEY RULE: If the last assistant message was a pending suggestion awaiting confirmation,
  the user's response — even a simple "yes" or "show me something different" — is ALWAYS
  classified as "recommendation".

**out_of_scope** — Completely unrelated to the dataset and not a memory/profile/
  recommendation question (general world knowledge, current events, creative writing):
  "Who won the Champions League?", "Write a poem", "Capital of France?"

Return the classification and a brief one-sentence reason."""


class QueryClassification(BaseModel):
    category: Literal["structured", "unstructured", "out_of_scope", "recommendation"] = Field(
        ..., description="The query type."
    )
    reasoning: str = Field(..., description="One sentence explaining the classification.")


_router_llm = None


def _get_router_llm():
    global _router_llm
    if _router_llm is None:
        _router_llm = get_llm(temperature=0).with_structured_output(QueryClassification)
    return _router_llm


def classify_query(user_query: str, last_ai_message: str = "") -> QueryClassification:
    """
    Classify a user query as structured, unstructured, out_of_scope, or recommendation.

    Parameters
    ----------
    user_query      : The user's latest message.
    last_ai_message : The agent's most recent non-tool-call message.  Used to detect
                      whether we are in the middle of a recommendation dialogue.
    """
    context = ""
    if last_ai_message:
        # Truncate long messages so the router prompt stays focused
        preview = last_ai_message[:400] + ("…" if len(last_ai_message) > 400 else "")
        context = f'\nLast assistant message (context for recommendation detection):\n"""{preview}"""\n'

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"{context}\nUser query to classify: {user_query}"),
    ]
    return _get_router_llm().invoke(messages)
