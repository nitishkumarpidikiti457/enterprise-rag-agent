"""Tests for the LLM router: fallback, pinning, caching, streaming and prompt templates."""

import pytest

from app.llm import prompts
from app.llm.cache import get_cache
from app.llm.providers import ExtractiveProvider, Message
from app.llm.router import LLMRouter
from tests.conftest import FakeProvider


@pytest.fixture(autouse=True)
def clean_cache():
    get_cache().clear()  # every test starts with an empty cache


async def test_falls_back_when_provider_fails():
    router = LLMRouter({"groq": FakeProvider(fail=True), "gemini": FakeProvider(reply="from gemini"),
                        "extractive": ExtractiveProvider("x")})
    resp = await router.complete([Message("user", "hello")])
    assert resp.text == "from gemini"


async def test_pinned_provider_is_tried_first():
    groq, gemini = FakeProvider(), FakeProvider()
    router = LLMRouter({"groq": groq, "gemini": gemini, "extractive": ExtractiveProvider("x")})
    await router.complete([Message("user", "hi")], provider="gemini")
    assert gemini.calls and not groq.calls


async def test_cache_hit_skips_provider():
    p = FakeProvider()
    router = LLMRouter({"groq": p, "extractive": ExtractiveProvider("x")})
    msgs = [Message("user", "same question")]
    await router.complete(msgs)
    await router.complete(msgs)
    assert len(p.calls) == 1  # second call served from cache


async def test_all_providers_fail_raises():
    router = LLMRouter({"groq": FakeProvider(fail=True)})
    with pytest.raises(RuntimeError, match="All LLM providers failed"):
        await router.complete([Message("user", "x")], use_cache=False)


async def test_extractive_fallback_works_offline():
    router = LLMRouter({"extractive": ExtractiveProvider("x")})
    resp = await router.complete([Message("user", "Context:\n[1] (source: a.md; section: S)\nRefunds take 7 days.")])
    assert resp.text.startswith("Refunds take 7 days.") and "[1]" in resp.text


async def test_stream_falls_back_before_first_token():
    router = LLMRouter({"groq": FakeProvider(fail=True), "extractive": ExtractiveProvider("x")})
    chunks = [d async for _, d in router.stream([Message("user", "no context here")])]
    assert "don't know" in "".join(chunks)


def test_classify_task_prefers_fast_model():
    router = LLMRouter({"groq": FakeProvider(), "groq-fast": FakeProvider(), "extractive": ExtractiveProvider("x")})
    assert router.chain("classify")[0] is router.providers["groq-fast"]


def test_prompt_templates_render():
    text = prompts.render("rag_user", question="Q?", context="[1] ctx")
    assert "Q?" in text and "[1] ctx" in text
    assert prompts.version() == "1.0"