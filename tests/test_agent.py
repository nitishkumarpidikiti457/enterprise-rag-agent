"""Tests for the SQL guard, intent routing and the LangGraph agent."""

import sqlite3

import pytest

from app.agent import tools
from app.agent.graph import heuristic_intent, run_agent, stream_agent


@pytest.mark.parametrize("sql", [
    "DELETE FROM sales",
    "SELECT 1; DROP TABLE sales",
    "UPDATE products SET unit_price = 0",
    "PRAGMA table_info(sales)",
    "",
])
def test_sql_guard_blocks_dangerous_queries(sql):
    with pytest.raises(tools.SQLGuardError):
        tools.validate_sql(sql)


def test_sql_guard_strips_markdown_and_adds_limit():
    assert tools.validate_sql("```sql\nSELECT * FROM products;\n```") == "SELECT * FROM products LIMIT 50"


def test_sql_runs_read_only():
    assert tools.sql_query("SELECT COUNT(*) AS n FROM recalls")["rows"][0][0] == 3


def test_database_rejects_writes_even_if_guard_is_bypassed():
    conn = tools._connect(tools.get_settings().sql_db_path)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM recalls")
    finally:
        conn.close()


def test_heuristic_intent():
    assert heuristic_intent("hello") == "chat"
    assert heuristic_intent("What were total sales in Q2?") == "data"
    assert heuristic_intent("What is the return policy?") == "docs"
    assert heuristic_intent("Which products in the recall policy had sales in Q2?") == "both"


async def test_docs_question_uses_vector_search_and_cites(store, agent_llm):
    state = await run_agent("How quickly must Class I recalls be pulled?", "t-docs")
    assert state["intent"] == "docs"
    assert [s.get("tool") for s in state["steps"] if s["node"] == "tool"] == ["vector_search"]
    assert state["citations"][0]["source"] == "product_recall_policy.pdf"


async def test_multi_source_question_uses_both_tools(store, agent_llm):
    state = await run_agent("Which products in the recall policy had sales last quarter?", "t-both")
    assert state["intent"] == "both"
    assert [s.get("tool") for s in state["steps"] if s["node"] == "tool"] == ["vector_search", "sql_query"]
    assert state["sql_result"]["rows"]                        # SQL really ran
    refs = {c["ref"] for c in state["citations"]}
    assert 1 in refs and "DB" in refs                         # cites a document AND the database


async def test_agent_remembers_conversation(store, agent_llm):
    await run_agent("How quickly must Class I recalls be pulled?", "t-mem")
    state = await run_agent("And Class III?", "t-mem")
    assert len(state["history"]) == 4                         # 2 turns x (user + assistant)


async def test_threads_are_isolated(store, agent_llm):
    await run_agent("How quickly must Class I recalls be pulled?", "thread-a")
    state = await run_agent("hello", "thread-b")
    assert len(state["history"]) == 2                         # thread-b doesn't see thread-a


async def test_stream_emits_steps_tokens_done(store, agent_llm):
    events = [e async for e in stream_agent("hello", "t-stream")]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "step" and "token" in kinds and kinds[-1] == "done"   

async def test_stream_does_not_replay_previous_turn_steps(store, agent_llm):
    _ = [e async for e in stream_agent("hello", "t-replay")]
    events = [e async for e in stream_agent("How quickly must Class I recalls be pulled?", "t-replay")]
    classify_steps = [e for e in events if e["event"] == "step" and e["data"]["node"] == "classify"]
    assert len(classify_steps) == 1 and classify_steps[0]["data"]["intent"] == "docs"