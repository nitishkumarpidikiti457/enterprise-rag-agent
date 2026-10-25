"""Shared test setup: fully offline (hash embeddings, lexical re-ranker)."""


import os

# Must be set before app modules read settings.
os.environ.setdefault("EMBEDDING_BACKEND", "hash")
os.environ.setdefault("RERANKER_BACKEND", "lexical")
os.environ.setdefault("JWT_SECRET", "test-secret-with-at-least-32-bytes!!")
os.environ.setdefault("DEMO_USERNAME", "admin")
os.environ.setdefault("DEMO_PASSWORD", "admin123")
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.ingestion.embed import HashEmbedder, set_embedder  # noqa: E402
from app.ingestion.pipeline import ingest_path  # noqa: E402
from app.retrieval.rerank import LexicalReranker, set_reranker  # noqa: E402
from app.retrieval.retriever import invalidate_bm25  # noqa: E402
from app.retrieval.vectorstore import ChromaStore, set_store  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
get_settings.cache_clear()


@pytest.fixture(scope="session")
def store():
    """A Chroma store with all 21 sample documents ingested, installed as the global store."""
    set_embedder(HashEmbedder())
    set_reranker(LexicalReranker())
    s = ChromaStore(ephemeral=True, collection="session_docs")
    set_store(s)
    invalidate_bm25()
    ingest_path(ROOT / "data" / "sample_docs", store=s)
    return s




from app.llm.providers import LLMProvider, LLMResponse  # noqa: E402


class FakeProvider(LLMProvider):
    """Offline stand-in for a real LLM: records calls, can be told to fail."""

    name = "fake"

    def __init__(self, reply: str = "fake answer [1]", fail: bool = False) -> None:
        super().__init__("fake-1")
        self.reply, self.fail = reply, fail
        self.calls: list[str] = []

    async def complete(self, messages, temperature=0.1, max_tokens=1024):
        if self.fail:
            raise RuntimeError("provider down")
        self.calls.append("\n".join(m.content for m in messages))
        return LLMResponse(self.reply, self.name, self.model, 10, 5)




from app.agent import graph as agent_graph  # noqa: E402
from app.llm.cache import get_cache  # noqa: E402
from app.llm.providers import ExtractiveProvider  # noqa: E402
from app.llm.router import LLMRouter, set_router  # noqa: E402


class ScriptedProvider(LLMProvider):
    """Fake LLM that returns a realistic canned reply for each agent prompt."""

    name = "scripted"

    def __init__(self) -> None:
        super().__init__("scripted-1")
        self.calls: list[str] = []

    async def complete(self, messages, temperature=0.1, max_tokens=1024):
        text = "\n".join(m.content for m in messages)
        self.calls.append(text)
        if "Respond with ONLY one word" in text:            # classify node
            q = text.rsplit("Question:", 1)[-1].lower()
            if "sales" in q and "recall" in q:
                out = "both"
            elif "sales" in q or "units" in q:
                out = "data"
            elif q.strip().startswith(("hi", "hello")):
                out = "chat"
            else:
                out = "docs"
        elif "read-only SQLite SELECT" in text:              # sql_tool node
            out = ("SELECT r.sku, p.name, SUM(s.units) AS units FROM recalls r "
                   "JOIN products p ON p.sku = r.sku JOIN sales s ON s.sku = r.sku "
                   "WHERE s.sale_date BETWEEN '2026-04-01' AND '2026-06-30' "
                   "GROUP BY r.sku ORDER BY units DESC")
        elif "Database results [DB]" in text:                # synthesize node
            out = "The recalled SKUs HF-1042, HF-2210 and HF-3307 are listed in the policy [1] and had Q2 sales [DB]."
        else:                                                # chat node
            out = "Hello!"
        return LLMResponse(out, self.name, self.model, 10, 10)


@pytest.fixture
def agent_llm():
    """Install the scripted LLM and a fresh in-memory agent graph for one test."""
    scripted = ScriptedProvider()
    set_router(LLMRouter({"scripted": scripted, "extractive": ExtractiveProvider("x")}))
    get_cache().clear()
    agent_graph.set_graph(agent_graph.build_graph())  # MemorySaver: nothing written to disk
    yield scripted
    set_router(None)
    agent_graph.set_graph(None)




from app.api.auth import reset_rate_limits  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_rate_limits():
    reset_rate_limits()  # every test starts with an empty rate-limit window