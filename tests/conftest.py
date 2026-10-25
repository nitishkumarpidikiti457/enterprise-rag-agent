"""Shared test setup: fully offline (hash embeddings, lexical re-ranker)."""

import os

# Must be set before app modules read settings.
os.environ.setdefault("EMBEDDING_BACKEND", "hash")
os.environ.setdefault("RERANKER_BACKEND", "lexical")

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