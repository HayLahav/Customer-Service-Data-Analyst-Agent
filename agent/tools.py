from typing import Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from agent.dataset import get_dataframe


# ── Tool 1: List all categories ──────────────────────────────────────────────

class GetAllCategoriesInput(BaseModel):
    pass  # no parameters — returns all categories unconditionally


@tool("get_all_categories", args_schema=GetAllCategoriesInput)
def get_all_categories() -> str:
    """Lists every category in the customer support dataset with its record count.
    Use this as the first step when asked what categories exist, or to understand
    the dataset's top-level structure. No parameters required."""
    df = get_dataframe()
    counts = df["category"].value_counts()
    lines = [f"  {cat}: {cnt} records" for cat, cnt in counts.items()]
    return (
        "Categories in dataset:\n"
        + "\n".join(lines)
        + f"\n\nTotal: {len(df)} records across {len(counts)} categories."
    )


# ── Tool 2: Intents within a category ────────────────────────────────────────

class GetIntentsByCategoryInput(BaseModel):
    category: str = Field(
        ...,
        description="Category name, case-insensitive (e.g. 'ACCOUNT', 'SHIPPING', 'REFUND').",
    )


@tool("get_intents_by_category", args_schema=GetIntentsByCategoryInput)
def get_intents_by_category(category: str) -> str:
    """Returns every intent type within a specific category, with record counts.
    Use this when asked what intents a category contains, or before drilling deeper
    into a category's data. Example: 'what intents exist in ACCOUNT?'"""
    df = get_dataframe()
    subset = df[df["category"].str.upper() == category.upper()]
    if subset.empty:
        available = ", ".join(sorted(df["category"].unique()))
        return f"No category '{category}' found. Available categories: {available}"
    counts = subset["intent"].value_counts()
    lines = [f"  {intent}: {cnt} records" for intent, cnt in counts.items()]
    return f"Intents in '{category.upper()}' ({len(subset)} total records):\n" + "\n".join(lines)


# ── Tool 3: Sample examples ───────────────────────────────────────────────────

class GetExamplesInput(BaseModel):
    category: Optional[str] = Field(
        None,
        description="Filter by category, case-insensitive. Omit to include all categories.",
    )
    intent: Optional[str] = Field(
        None,
        description="Filter by intent, case-insensitive (e.g. 'cancel_order'). Omit for all intents.",
    )
    n: int = Field(5, description="Number of examples to return (1–20).", ge=1, le=20)


@tool("get_examples", args_schema=GetExamplesInput)
def get_examples(
    category: Optional[str] = None,
    intent: Optional[str] = None,
    n: int = 5,
) -> str:
    """Retrieves N sample records from the dataset, optionally filtered by category and/or intent.
    Each record shows the customer's message and the agent's response.
    Use for 'show me X examples of Y', demonstrating what a category/intent looks like,
    or gathering raw data before writing a summary."""
    df = get_dataframe()
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
            f"Example {i} [category={row['category']}, intent={row['intent']}]:\n"
            f"  Customer: {row['instruction']}\n"
            f"  Agent:    {row['response']}"
        )
    return f"Showing {len(sample)} of {len(df)} matching records:\n\n" + "\n\n".join(parts)


# ── Tool 4: Count records ─────────────────────────────────────────────────────

class CountRecordsInput(BaseModel):
    category: Optional[str] = Field(
        None,
        description="Filter by category, case-insensitive. Omit to count across all categories.",
    )
    intent: Optional[str] = Field(
        None,
        description="Filter by intent, case-insensitive. Omit to count across all intents.",
    )


@tool("count_records", args_schema=CountRecordsInput)
def count_records(
    category: Optional[str] = None,
    intent: Optional[str] = None,
) -> str:
    """Counts the number of records in the dataset with optional category and/or intent filters.
    Use for 'how many X' questions: 'how many refund requests', 'how many shipping complaints',
    'how many records are in the ACCOUNT category'. Returns an exact integer count."""
    df = get_dataframe()
    label_parts = []
    if category:
        df = df[df["category"].str.upper() == category.upper()]
        label_parts.append(f"category='{category.upper()}'")
    if intent:
        df = df[df["intent"].str.lower() == intent.lower()]
        label_parts.append(f"intent='{intent.lower()}'")
    label = " AND ".join(label_parts) if label_parts else "the entire dataset"
    return f"Record count for {label}: {len(df)}"


# ── Tool 5: Intent distribution within a category ────────────────────────────

class GetIntentDistributionInput(BaseModel):
    category: str = Field(
        ...,
        description="The category to analyze, case-insensitive (e.g. 'ACCOUNT', 'ORDER').",
    )


@tool("get_intent_distribution", args_schema=GetIntentDistributionInput)
def get_intent_distribution(category: str) -> str:
    """Returns the distribution of intents within a category: count and percentage for each intent.
    Use for 'what is the distribution of intents in X', 'breakdown of X category',
    or 'what are the most common intents in X'. Shows proportions at a glance."""
    df = get_dataframe()
    subset = df[df["category"].str.upper() == category.upper()]
    if subset.empty:
        available = ", ".join(sorted(df["category"].unique()))
        return f"No category '{category}' found. Available: {available}"
    total = len(subset)
    counts = subset["intent"].value_counts()
    lines = [
        f"  {intent}: {cnt} records ({cnt / total * 100:.1f}%)"
        for intent, cnt in counts.items()
    ]
    return (
        f"Intent distribution in '{category.upper()}' (total: {total} records):\n"
        + "\n".join(lines)
    )


# ── Tool 6: Keyword search ────────────────────────────────────────────────────

class SearchExamplesInput(BaseModel):
    query: str = Field(
        ...,
        description="Keyword(s) to search for, case-insensitive (e.g. 'refund', 'password reset', 'late delivery').",
    )
    n: int = Field(5, description="Number of matching examples to return (1–20).", ge=1, le=20)
    search_in: Literal["instruction", "response", "both"] = Field(
        "both",
        description=(
            "Which text field to search: "
            "'instruction' = customer messages only, "
            "'response' = agent replies only, "
            "'both' = either field."
        ),
    )


@tool("search_examples", args_schema=SearchExamplesInput)
def search_examples(
    query: str,
    n: int = 5,
    search_in: str = "both",
) -> str:
    """Searches the dataset for records whose customer message or agent response contains keywords.
    Use when the user describes a topic in natural language rather than an exact intent name —
    e.g. 'people wanting their money back', 'angry customers', 'delivery problems',
    'how agents handle complaints'. Also useful for finding examples for summarization."""
    df = get_dataframe()
    q = query.lower()
    if search_in == "instruction":
        mask = df["instruction"].str.lower().str.contains(q, na=False)
    elif search_in == "response":
        mask = df["response"].str.lower().str.contains(q, na=False)
    else:
        mask = df["instruction"].str.lower().str.contains(q, na=False) | df[
            "response"
        ].str.lower().str.contains(q, na=False)
    matches = df[mask]
    if matches.empty:
        return f"No records found containing '{query}'."
    sample = matches.sample(min(n, len(matches)), random_state=42)
    parts = []
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        parts.append(
            f"Example {i} [category={row['category']}, intent={row['intent']}]:\n"
            f"  Customer: {row['instruction']}\n"
            f"  Agent:    {row['response']}"
        )
    header = f"Found {len(matches)} records matching '{query}'. Showing {len(sample)}:\n\n"
    return header + "\n\n".join(parts)


# ── Tool 7: Dataset overview ──────────────────────────────────────────────────

class GetDatasetOverviewInput(BaseModel):
    pass  # no parameters — returns global statistics


@tool("get_dataset_overview", args_schema=GetDatasetOverviewInput)
def get_dataset_overview() -> str:
    """Returns high-level statistics about the entire customer support dataset:
    total record count, number of categories, number of unique intents, and top intents by volume.
    Use when the user asks for a general overview, wants to understand the dataset's scope,
    or before deciding which category/intent to explore."""
    df = get_dataframe()
    n_cats = df["category"].nunique()
    n_intents = df["intent"].nunique()
    top_intents = df["intent"].value_counts().head(5)
    lines = [f"  {intent}: {cnt}" for intent, cnt in top_intents.items()]
    return (
        f"Dataset Overview:\n"
        f"  Total records : {len(df):,}\n"
        f"  Categories    : {n_cats}\n"
        f"  Unique intents: {n_intents}\n\n"
        f"Top 5 intents by volume:\n" + "\n".join(lines)
    )


TOOLS = [
    get_all_categories,
    get_intents_by_category,
    get_examples,
    count_records,
    get_intent_distribution,
    search_examples,
    get_dataset_overview,
]
