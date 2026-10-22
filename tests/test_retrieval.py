"""Tests for RRF fusion, hybrid retrieval and re-ranking."""

import json
from pathlib import Path

from app.retrieval.retriever import RetrievalConfig, format_context, retrieve, rrf_fuse
from app.retrieval.vectorstore import SearchHit

GOLDEN = Path(__file__).resolve().parents[1] / "eval" / "golden_set.jsonl"


def hit(cid: str) -> SearchHit:
    return SearchHit(cid, "", {}, 1.0)


def test_rrf_rewards_agreement():
    # "y" is 2nd in one list and 1st in the other -> beats items that appear only once.
    fused = rrf_fuse([[hit("x"), hit("y")], [hit("y"), hit("z")]], [0.5, 0.5])
    assert fused[0].chunk_id == "y"


def test_finds_correct_document_and_section(store):
    hits = retrieve("How quickly must a Class I recall be pulled from shelves?")
    assert hits[0].metadata["source"] == "product_recall_policy.pdf"
    assert hits[0].metadata["section"] == "Recall Classes"


def test_exact_sku_found_via_keywords(store):
    # BM25 shines on exact identifiers like SKU codes.
    hits = retrieve("HF-2210")
    assert any("HF-2210" in h.text for h in hits)


def test_source_filter(store):
    hits = retrieve("refund", RetrievalConfig(source="return_and_refund_policy.docx"))
    assert hits and all(h.metadata["source"] == "return_and_refund_policy.docx" for h in hits)


def test_top_n_limits_results(store):
    assert len(retrieve("policy", RetrievalConfig(top_n=3))) == 3


def test_format_context_numbers_passages(store):
    ctx = format_context(retrieve("PTO carryover", RetrievalConfig(top_n=2)))
    assert ctx.startswith("[1] (source: ") and "\n\n[2] (source: " in ctx


def test_golden_set_recall(store):
    golden = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines()]
    found = sum(any(h.metadata["source"] == g["source"] for h in retrieve(g["question"])) for g in golden)
    assert found / len(golden) >= 0.9  # at least 90% of the 50 questions find the right document