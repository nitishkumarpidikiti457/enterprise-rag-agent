"""Tests for grounded RAG: retrieval + prompt + answer + citation extraction."""

import pytest

from app.llm.cache import get_cache
from app.llm.providers import ExtractiveProvider
from app.llm.router import LLMRouter, set_router
from app.rag import IDK, answer_question, build_messages, extract_citations, stream_answer
from app.retrieval.retriever import RetrievalConfig, retrieve
from app.retrieval.vectorstore import SearchHit
from tests.conftest import FakeProvider


@pytest.fixture
def offline_router():
    set_router(LLMRouter({"extractive": ExtractiveProvider("x")}))
    get_cache().clear()
    yield
    set_router(None)  # restore the default router for other tests


def test_build_messages_has_rules_and_numbered_context(store):
    msgs = build_messages("What is the PTO carryover?", retrieve("PTO carryover"))
    assert msgs[0].role == "system" and "cite" in msgs[0].content.lower()
    assert "Question: What is the PTO carryover?" in msgs[1].content
    assert "[1] (source: " in msgs[1].content


def test_extract_citations_maps_numbers_to_sources():
    hits = [SearchHit("a", "text A", {"source": "a.pdf", "page": 2, "section": "S1"}, 1.0),
            SearchHit("b", "text B", {"source": "b.md", "page": -1, "section": "S2"}, 0.9)]
    cites = extract_citations("Fact one [1]. Fact two [2][1]. Bogus [9].", hits)
    assert [c["ref"] for c in cites] == [1, 2]        # deduplicated, sorted, [9] ignored
    assert cites[0]["source"] == "a.pdf" and cites[0]["page"] == 2


async def test_answer_is_grounded_and_cited(store, offline_router):
    res = await answer_question("How quickly must a Class I recall be pulled from shelves?")
    assert "2 hours" in res.answer
    assert res.citations and res.citations[0]["source"] == "product_recall_policy.pdf"
    assert res.provider == "extractive"


async def test_no_hits_returns_idk_without_calling_llm(store):
    fake = FakeProvider()
    set_router(LLMRouter({"fake": fake}))
    try:
        res = await answer_question("anything", cfg=RetrievalConfig(source="does_not_exist.pdf"))
        assert res.answer == IDK and fake.calls == []    # LLM never called
    finally:
        set_router(None)


async def test_stream_emits_sources_tokens_done(store, offline_router):
    events = [e async for e in stream_answer("What is the daily meal allowance for travel?")]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "sources" and kinds[-1] == "done" and "token" in kinds
    assert events[-1]["data"]["citations"][0]["source"] == "travel_and_expense_policy.docx"