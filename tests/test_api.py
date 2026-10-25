"""Tests for the FastAPI service: auth, validation, endpoints, streaming, uploads, rate limit."""

import io

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.config import get_settings


@pytest.fixture
def client(store, agent_llm):
    return TestClient(app)


@pytest.fixture
def auth(client):
    r = client.post("/auth/token", json={"username": "admin", "password": "admin123"})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_health_is_public(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["documents_indexed_chunks"] > 0


def test_bad_login_is_rejected(client):
    assert client.post("/auth/token", json={"username": "admin", "password": "nope"}).status_code == 401


def test_endpoints_require_a_valid_token(client):
    assert client.post("/query", json={"question": "anything here"}).status_code == 401
    bad = {"Authorization": "Bearer not-a-real-token"}
    assert client.post("/query", json={"question": "anything here"}, headers=bad).status_code == 401


def test_query_returns_grounded_answer(client, auth):
    r = client.post("/query", json={"question": "How quickly must Class I recalls be pulled?",
                                    "provider": "extractive"}, headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert "2 hours" in body["answer"]
    assert body["citations"][0]["source"] == "product_recall_policy.pdf"


@pytest.mark.parametrize("payload", [
    {"question": "x"},                                         # too short
    {"question": "   "},                                       # only spaces
    {"question": "valid question", "top_k": 0},                # out of range
    {"question": "valid question", "provider": "not-real"},    # unknown provider
    {"question": "a" * 2001},                                  # too long
])
def test_invalid_input_returns_422(client, auth, payload):
    assert client.post("/query", json=payload, headers=auth).status_code == 422


def test_query_streaming(client, auth):
    with client.stream("POST", "/query", json={"question": "Class I recall timing?", "stream": True,
                                               "provider": "extractive"}, headers=auth) as r:
        body = "".join(r.iter_text())
    assert "event: sources" in body and "event: token" in body and "event: done" in body


def test_chat_streams_agent_steps(client, auth):
    with client.stream("POST", "/chat", json={"message": "Which products in the recall policy had sales last quarter?",
                                              "thread_id": "api1"}, headers=auth) as r:
        body = "".join(r.iter_text())
    assert "event: step" in body and "sql_query" in body and "event: done" in body


def test_chat_non_streaming(client, auth):
    r = client.post("/chat", json={"message": "What is the PTO carryover?", "thread_id": "api2", "stream": False},
                    headers=auth)
    assert r.status_code == 200 and r.json()["intent"] == "docs"


def test_invalid_thread_id_rejected(client, auth):
    r = client.post("/chat", json={"message": "hi", "thread_id": "../etc"}, headers=auth)
    assert r.status_code == 422


def test_upload_ingests_and_checks_type(client, auth):
    good = {"files": ("parking_policy.md", io.BytesIO(b"# Parking Policy\n\n## Spaces\n\nEmployees park in rows F and G."),
                      "text/markdown")}
    r = client.post("/ingest", files=good, headers=auth)
    assert r.status_code == 200 and r.json()[0]["chunks"] >= 1

    bad = {"files": ("evil.exe", io.BytesIO(b"MZ"), "application/octet-stream")}
    assert client.post("/ingest", files=bad, headers=auth).status_code == 415


def test_rate_limit(client, auth, monkeypatch):
    monkeypatch.setattr(get_settings(), "rate_limit_per_minute", 2)
    codes = [client.get("/metrics", headers=auth).status_code for _ in range(3)]
    assert codes == [200, 200, 429]