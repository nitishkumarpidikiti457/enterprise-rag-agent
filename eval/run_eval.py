"""Evaluation harness.

Retrieval (always, no API key needed):
  hit@1, recall@5, MRR@5 and doc precision@5 for dense / hybrid / hybrid+rerank.
Generation (with --generation; uses your LLM keys):
  answer accuracy (LLM-as-judge), citation rate, refusals on unanswerable questions,
  p50/p95 latency, per provider.

    python eval\\run_eval.py                                  # retrieval only
    python eval\\run_eval.py --generation                     # + every configured provider
    python eval\\run_eval.py --generation --providers groq    # one provider
    python eval\\run_eval.py --generation --limit 10 --sleep 2  # stay inside free-tier rate limits
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from app.config import get_settings
from app.ingestion.pipeline import ingest_path
from app.llm.cache import get_cache
from app.llm.providers import Message
from app.llm.router import get_router
from app.rag import answer_question
from app.retrieval.retriever import RetrievalConfig, retrieve

GOLDEN = Path(__file__).with_name("golden_set.jsonl")
RESULTS = Path(__file__).with_name("results.md")
UNANSWERABLE = [
    "What is the CEO's home address?",
    "How many parking spaces does the Tampa distribution center have?",
    "What is the stock price target for next year?",
]
CONFIGS = {
    "dense only": RetrievalConfig(hybrid=False, rerank=False),
    "hybrid (dense+BM25, RRF)": RetrievalConfig(hybrid=True, rerank=False),
    "hybrid + rerank": RetrievalConfig(hybrid=True, rerank=True),
}


def load_golden(path: Path = GOLDEN) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score_ranking(match: list[bool]) -> tuple[float, float, float]:
    """(hit@1, recall@k, reciprocal rank) for one question, given which results were correct."""
    hit1 = float(bool(match) and match[0])
    recall = float(any(match))
    rr = next((1 / (i + 1) for i, m in enumerate(match) if m), 0.0)
    return hit1, recall, rr


def retrieval_eval(golden: list[dict], k: int = 5) -> list[dict]:
    rows = []
    for name, base in CONFIGS.items():
        cfg = RetrievalConfig(top_k=20, top_n=k, hybrid=base.hybrid, rerank=base.rerank)
        hit1 = recall = rr = prec = 0.0
        lat = []
        for g in golden:
            t = time.perf_counter()
            hits = retrieve(g["question"], cfg)
            lat.append(time.perf_counter() - t)
            match = [h.metadata.get("source") == g["source"] and h.metadata.get("section") == g["section"]
                     for h in hits]
            h1, rc, r = score_ranking(match)
            hit1, recall, rr = hit1 + h1, recall + rc, rr + r
            prec += sum(h.metadata.get("source") == g["source"] for h in hits) / max(len(hits), 1)
        n = len(golden)
        rows.append({"config": name, "hit@1": hit1 / n, f"recall@{k}": recall / n, f"MRR@{k}": rr / n,
                     f"doc precision@{k}": prec / n, "p50 ms": statistics.median(lat) * 1000})
    return rows


JUDGE = """You grade answers. Reference answer: "{ref}". Candidate answer: "{cand}".
Does the candidate state the same fact as the reference (extra detail is fine)? Reply with only YES or NO."""


def overlap_match(ref: str, cand: str) -> bool:
    """Offline fallback judge: at least 60% of the reference's meaningful words appear in the answer."""
    toks = [t for t in ref.lower().replace(",", " ").split() if len(t) > 2]
    return bool(toks) and sum(t in cand.lower() for t in toks) / len(toks) >= 0.6


async def judge(ref: str, cand: str) -> bool:
    if ref.lower() in cand.lower():
        return True
    try:
        r = await get_router().complete([Message("user", JUDGE.format(ref=ref, cand=cand))],
                                        task="classify", max_tokens=3, temperature=0.0)
        if r.provider != "extractive" and r.text.strip():
            return r.text.strip().upper().startswith("YES")
    except Exception:  # noqa: BLE001 - fall back to the offline check
        pass
    return overlap_match(ref, cand)


async def generation_eval(golden: list[dict], providers: list[str], sleep: float = 0.0) -> list[dict]:
    rows = []
    for prov in providers:
        get_cache().clear()  # measure real calls, not cached ones
        correct = cited = refused = 0
        lat = []
        model = get_router().providers[prov].model
        for i, g in enumerate(golden, 1):
            t = time.perf_counter()
            res = await answer_question(g["question"], provider=prov)
            lat.append(time.perf_counter() - t)
            ok = await judge(g["answer"], res.answer)
            correct += ok
            cited += bool(res.citations)
            print(f"  [{prov}] {i}/{len(golden)} {'OK ' if ok else 'MISS'} {g['question'][:60]}")
            await asyncio.sleep(sleep)
        for q in UNANSWERABLE:
            res = await answer_question(q, provider=prov)
            refused += "don't know" in res.answer.lower()
            await asyncio.sleep(sleep)
        n, s = len(golden), sorted(lat)
        rows.append({"provider": f"{prov} ({model})", "answer accuracy": correct / n, "citation rate": cited / n,
                     f"refusals (of {len(UNANSWERABLE)})": refused,
                     "p50 s": statistics.median(lat), "p95 s": s[int(0.95 * (len(s) - 1))]})
    return rows


def to_markdown(rows: list[dict]) -> str:
    if not rows:
        return ""
    cols = list(rows[0])
    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate retrieval and answer quality on the golden set")
    ap.add_argument("--generation", action="store_true", help="also evaluate LLM answers (uses API keys)")
    ap.add_argument("--providers", nargs="*", help="providers to evaluate, e.g. groq gemini")
    ap.add_argument("--limit", type=int, default=0, help="only use the first N questions for generation")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to wait between LLM calls")
    args = ap.parse_args()

    ingest_path("data/sample_docs")  # no-op if already ingested
    golden = load_golden()
    st = get_settings()
    out = [f"# Evaluation results\n\n{len(golden)} golden questions over {len({g['source'] for g in golden})} "
           f"documents. Embeddings: `{st.embedding_backend}` ({st.embedding_model}); "
           f"reranker: `{st.reranker_backend}`.\n",
           "## Retrieval\n", to_markdown(retrieval_eval(golden))]
    if args.generation:
        provs = args.providers or [p for p in get_router().available() if p != "extractive"] or ["extractive"]
        subset = golden[: args.limit] if args.limit else golden
        out += [f"\n## Generation ({len(subset)} questions)\n",
                to_markdown(asyncio.run(generation_eval(subset, provs, args.sleep)))]
    text = "\n".join(out) + "\n"
    RESULTS.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()