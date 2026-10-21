"""Tests for configuration loading and observability helpers."""

import json
import logging
import time

from app.config import Settings, get_settings
from app.observability.logging import JsonFormatter, Metrics


def test_settings_defaults():
    # Build Settings directly (not via the cached getter) with no .env file.
    s = Settings(_env_file=None)
    assert s.retrieve_top_k == 20
    assert s.rerank_top_n == 5
    assert s.groq_api_key is None  # no key -> provider stays off -> free by default


def test_env_var_overrides_default(monkeypatch):
    # monkeypatch sets an env var only for this test, then restores it.
    monkeypatch.setenv("RERANK_TOP_N", "7")
    get_settings.cache_clear()  # force get_settings() to re-read the environment
    try:
        assert get_settings().rerank_top_n == 7  # text "7" converted to int 7
    finally:
        get_settings.cache_clear()


def test_metrics_counters_and_percentiles():
    m = Metrics()
    m.inc("queries")
    m.inc("queries")
    for ms in range(1, 101):  # 100 latencies: 1ms ... 100ms
        m.observe("query", ms / 1000)
    snap = m.snapshot()
    assert snap["counters"]["queries"] == 2
    assert 49 <= snap["latency_ms"]["query"]["p50"] <= 52
    assert 94 <= snap["latency_ms"]["query"]["p95"] <= 97


def test_metrics_timer_records_duration():
    m = Metrics()
    with m.timer("work"):
        time.sleep(0.01)
    assert m.latencies["work"][0] >= 0.01


def test_json_formatter_includes_extra_fields():
    record = logging.LogRecord("app.test", logging.INFO, __file__, 1, "llm_call", None, None)
    record.extra_fields = {"provider": "groq", "latency_ms": 812}
    payload = json.loads(JsonFormatter().format(record))
    assert payload["msg"] == "llm_call"
    assert payload["provider"] == "groq"
    assert payload["level"] == "INFO"