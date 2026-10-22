"""Tests for semantic chunking, the vector store and the ingestion pipeline."""

import uuid
from pathlib import Path

from app.ingestion.chunking import chunk_document, recursive_split, split_sentences
from app.ingestion.embed import HashEmbedder
from app.ingestion.loaders import LoadedDocument, PageBlock
from app.ingestion.pipeline import ingest_path
from app.retrieval.vectorstore import ChromaStore

DOCS = Path(__file__).resolve().parents[1] / "data" / "sample_docs"


def new_store() -> ChromaStore:
    # In-memory store with a unique collection name so tests don't share data.
    return ChromaStore(ephemeral=True, collection=f"test_{uuid.uuid4().hex[:8]}")


def test_split_sentences():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_recursive_split_respects_max():
    text = " ".join(["word"] * 1000)
    parts = recursive_split(text, 300, overlap=0)
    assert all(len(p) <= 300 for p in parts)
    assert "".join(parts).replace(" ", "") == text.replace(" ", "")  # nothing lost


def test_semantic_chunking_splits_topics_and_sets_metadata():
    text = ("Refrigerators must stay cold at 41 degrees. Freezers must stay at zero degrees. "
            "Cold cases are checked often. The loyalty program gives points for purchases. "
            "Points can be redeemed for discounts. Members earn rewards on every dollar.")
    doc = LoadedDocument("d1", "x.md", [PageBlock(text, page=3, section="Mixed")])
    chunks = chunk_document(doc, HashEmbedder(), max_chars=400, min_chars=10, percentile=70)
    assert len(chunks) >= 2
    md = chunks[0].metadata
    assert md["page"] == 3 and md["section"] == "Mixed" and md["source"] == "x.md"
    assert len({c.chunk_id for c in chunks}) == len(chunks)  # ids are unique


def test_vectorstore_upsert_query_and_filter():
    store, emb = new_store(), HashEmbedder()
    doc = LoadedDocument("d2", "temps.md", [PageBlock("Refrigerated cases hold 41 degrees or below.", section="Cold")])
    chunks = chunk_document(doc, emb)
    store.upsert(chunks, emb.embed_documents([c.text for c in chunks]))
    hits = store.query(emb.embed_query("refrigerated case temperature"), k=3)
    assert hits[0].metadata["source"] == "temps.md"
    assert store.query(emb.embed_query("x"), k=3, where={"source": "other.md"}) == []
    assert store.doc_ids() == {"d2"}


def test_pipeline_ingests_all_docs_and_is_idempotent():
    store, emb = new_store(), HashEmbedder()
    first = ingest_path(DOCS, store=store, embedder=emb)
    assert len(first) == 21 and not any(r.skipped for r in first)
    count = store.count()
    assert count > 21  # several chunks per document

    second = ingest_path(DOCS, store=store, embedder=emb)
    assert all(r.skipped for r in second)   # unchanged files are skipped
    assert store.count() == count            # no duplicates