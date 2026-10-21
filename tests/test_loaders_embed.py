"""Tests for document loaders and embedders (offline, no model downloads)."""

import math
from pathlib import Path

import pytest

from app.ingestion.embed import HashEmbedder, tokenize
from app.ingestion.loaders import load_document

DOCS = Path(__file__).resolve().parents[1] / "data" / "sample_docs"


def test_pdf_loader_keeps_pages_and_sections():
    doc = load_document(DOCS / "product_recall_policy.pdf")
    assert len({b.page for b in doc.blocks}) >= 2          # multi-page PDF
    assert "Recall Classes" in {b.section for b in doc.blocks}
    assert "Class I" in doc.text


def test_docx_loader_uses_heading_styles():
    doc = load_document(DOCS / "return_and_refund_policy.docx")
    assert {"General Returns", "Exclusions"} <= {b.section for b in doc.blocks}


def test_markdown_loader():
    doc = load_document(DOCS / "paid_time_off_policy.md")
    assert "Carryover" in {b.section for b in doc.blocks}


def test_doc_id_is_stable():
    a = load_document(DOCS / "paid_time_off_policy.md").doc_id
    b = load_document(DOCS / "paid_time_off_policy.md").doc_id
    assert a == b  # same file -> same id -> safe re-ingest


def test_unsupported_type_raises(tmp_path):
    f = tmp_path / "evil.exe"
    f.write_bytes(b"MZ")
    with pytest.raises(ValueError):
        load_document(f)


def test_tokenize_stems_and_drops_stopwords():
    assert tokenize("The Passwords are rotated") == ["password", "rotat"]


def test_hash_embedder_shape_and_similarity():
    e = HashEmbedder()
    v1, v2, v3 = e.embed_documents(["refund policy for returns", "return and refund policy", "freezer temperature"])
    assert len(v1) == 384
    assert math.isclose(sum(x * x for x in v1), 1.0, rel_tol=1e-6)   # normalised
    cos = lambda a, b: sum(x * y for x, y in zip(a, b, strict=True))
    assert cos(v1, v2) > cos(v1, v3)  # related texts are closer than unrelated ones