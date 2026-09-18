"""The agent itself.

Kept separate from the web server so the notebook and the server can share
exactly the same agent definition.
"""

import os
from typing import Literal

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from tavily import TavilyClient

from testing.tools import write_docx, write_text_file

load_dotenv()

SYSTEM_PROMPT = """
You are a helpful assistant.

Use the internet_search tool when the question needs fresh or factual
information. Answer clearly and concisely.

Saving files:
- `write_docx` saves a real Word (.docx) file the user can open. Use it whenever
  the user asks for a Word document or a .docx report.
- `write_text_file` saves a real .txt or .md file.
- Your other file tools (`write_file`, `edit_file`, ...) write to a scratch
  workspace that the user CANNOT see or open. Never tell the user a file is
  ready unless you saved it with `write_docx` or `write_text_file`.
"""

_tavily_client: TavilyClient | None = None


def _get_tavily() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is not set in .env")
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search."""
    try:
        return _get_tavily().search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
    except Exception as exc:  # surface the failure to the model instead of crashing the run
        return {"error": f"search failed: {type(exc).__name__}: {exc}"}


def build_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ.get("MODEL_NAME", "gpt-4o-mini"),
        api_key=os.environ.get("MODEL_API_KEY", "not-needed"),
        base_url=os.environ.get("MODEL_BASE_URL") or None,
    )


def build_agent(checkpointer=None):
    """Build the agent.

    Args:
        checkpointer: where conversation history lives. Defaults to in-memory,
            which means history is lost when the process restarts.
    """
    return create_deep_agent(
        model=build_model(),
        tools=[internet_search, write_docx, write_text_file],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
    )
