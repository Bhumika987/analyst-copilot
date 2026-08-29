import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from config import BM25_TOP_K, RERANK_TOP_K, RRF_K, RRF_TOP_K, SEMANTIC_TOP_K, get_embedding_model_name  # noqa: E402
from llm import _format_context, evaluate_retrieval_status  # noqa: E402
from query_analyzer import analyze_query  # noqa: E402
from retrieval import _rrf_fuse, deterministic_rerank, get_index  # noqa: E402


def _rank_map(items):
    return {chunk_id: rank for rank, chunk_id in enumerate(items, start=1)}


def _preview(text, limit=100):
    preview = " ".join((text or "").split())[:limit]
    return preview.encode("ascii", errors="replace").decode("ascii")


def _print_stage(title, rows):
    print(f"\n{title}")
    print("-" * len(title))
    if not rows:
        print("(none)")
        return
    for row in rows:
        print(str(row).encode("ascii", errors="replace").decode("ascii"))


def diagnose(query: str, doc_name: str, top_k: int):
    idx = get_index(doc_name)
    if idx is None or not idx.is_indexed():
        raise SystemExit(f"No indexed filing found for {doc_name}")

    query_info = analyze_query(query)
    bm25_ids = idx.search_bm25(query, top_k=BM25_TOP_K)
    metadata_hits = idx.search_metadata(query, top_k=BM25_TOP_K)
    metadata_ids = [chunk_id for chunk_id, _ in metadata_hits]
    metadata_scores = dict(metadata_hits)
    faiss_hits = idx.search_bge_faiss(query, top_k=SEMANTIC_TOP_K)
    faiss_ids = [chunk_id for chunk_id, _ in faiss_hits]
    faiss_scores = dict(faiss_hits)
    bm25_ranks = _rank_map(bm25_ids)
    metadata_ranks = _rank_map(metadata_ids)
    faiss_ranks = _rank_map(faiss_ids)

    fused = _rrf_fuse([(bm25_ids, 1.0), (metadata_ids, 0.9), (faiss_ids, 1.0)], k=RRF_K)
    rrf_ids_all = sorted(fused, key=lambda chunk_id: fused[chunk_id], reverse=True)
    rrf_ids_kept = rrf_ids_all[:RRF_TOP_K]

    candidates = []
    for chunk_id in rrf_ids_kept:
        if chunk_id < 0 or chunk_id >= len(idx.chunks):
            continue
        chunk = dict(idx.chunks[chunk_id])
        chunk.update(
            doc_name=idx.doc_name,
            company=idx.metadata.get("company", idx.doc_name.split("_")[0]),
            filing_type=idx.metadata.get("filing_type", "Filing"),
            fiscal_year=idx.metadata.get("fiscal_year", ""),
            source_filename=idx.metadata.get("source_filename", f"{idx.doc_name}.htm"),
            chunk_idx=chunk_id,
            retrieval_score=fused[chunk_id],
            bm25_score=idx.bm25_score(query, chunk_id),
            metadata_score=metadata_scores.get(chunk_id, 0.0),
            semantic_score=faiss_scores.get(chunk_id, 0.0),
            bm25_rank=bm25_ranks.get(chunk_id),
            metadata_rank=metadata_ranks.get(chunk_id),
            dense_rank=faiss_ranks.get(chunk_id),
            semantic_rank=faiss_ranks.get(chunk_id),
        )
        candidates.append(chunk)

    reranked = deterministic_rerank(query, candidates, query_info=query_info)
    final_results = reranked[:top_k]
    retrieval_status, retrieval_reason = evaluate_retrieval_status(final_results, query=query)
    _, context_chunks = _format_context(final_results, max_chunks=4)

    print("QUERY:", query)
    print("TARGET DOCUMENT:", doc_name)
    print("EMBEDDING MODEL:", get_embedding_model_name())
    print("VECTOR INDEX:", idx.vector_index_dir())
    print("TOP-K SETTINGS:", {
        "BM25_TOP_K": BM25_TOP_K,
        "SEMANTIC_TOP_K": SEMANTIC_TOP_K,
        "RRF_TOP_K": RRF_TOP_K,
        "RERANK_TOP_K": RERANK_TOP_K,
        "requested_top_k": top_k,
        "LLM_CONTEXT_TOP_K": 4,
    })
    print("RETRIEVAL STATUS:", retrieval_status, "-", retrieval_reason)

    _print_stage(
        "BM25",
        [
            f"{rank:>2} | chunk={chunk_id:<4} | score={idx.bm25_score(query, chunk_id):.4f} | "
            f"type={idx.chunks[chunk_id].get('chunk_type')} | page={idx.chunks[chunk_id].get('page_num')} | "
            f"{_preview(idx.chunks[chunk_id].get('text'))}"
            for rank, chunk_id in enumerate(bm25_ids, start=1)
        ],
    )
    _print_stage(
        "FAISS",
        [
            f"{rank:>2} | chunk={chunk_id:<4} | score={faiss_scores.get(chunk_id, 0.0):.4f} | "
            f"type={idx.chunks[chunk_id].get('chunk_type')} | page={idx.chunks[chunk_id].get('page_num')} | "
            f"{_preview(idx.chunks[chunk_id].get('text'))}"
            for rank, chunk_id in enumerate(faiss_ids, start=1)
        ],
    )
    _print_stage(
        "METADATA",
        [
            f"{rank:>2} | chunk={chunk_id:<4} | score={metadata_scores.get(chunk_id, 0.0):.4f} | "
            f"type={idx.chunks[chunk_id].get('chunk_type')} | page={idx.chunks[chunk_id].get('page_num')} | "
            f"{_preview(idx.chunks[chunk_id].get('text'))}"
            for rank, chunk_id in enumerate(metadata_ids, start=1)
        ],
    )
    _print_stage(
        "RRF ALL",
        [
            f"{rank:>2} | chunk={chunk_id:<4} | rrf={fused[chunk_id]:.5f} | "
            f"bm25_rank={bm25_ranks.get(chunk_id) or '-':>2} | "
            f"metadata_rank={metadata_ranks.get(chunk_id) or '-':>2} | "
            f"faiss_rank={faiss_ranks.get(chunk_id) or '-':>2}"
            for rank, chunk_id in enumerate(rrf_ids_all, start=1)
        ],
    )
    _print_stage(
        "RERANK",
        [
            f"{rank:>2} | chunk={c['chunk_idx']:<4} | score={c.get('rerank_score', 0.0):.2f} | "
            f"rrf={c.get('retrieval_score', 0.0):.5f} | bm25_rank={c.get('bm25_rank') or '-':>2} | "
            f"metadata_rank={c.get('metadata_rank') or '-':>2} | "
            f"faiss_rank={c.get('semantic_rank') or '-':>2} | type={c.get('chunk_type')} | page={c.get('page_num')}"
            for rank, c in enumerate(reranked, start=1)
        ],
    )
    _print_stage(
        "FINAL CONTEXT",
        [
            f"{rank:>2} | chunk={c.get('chunk_idx'):<4} | page={c.get('page_num')} | type={c.get('chunk_type')}"
            for rank, c in enumerate(context_chunks, start=1)
        ],
    )

    bm25_set = set(bm25_ids)
    rrf_all_set = set(rrf_ids_all)
    rrf_kept_set = set(rrf_ids_kept)
    rerank_set = {c["chunk_idx"] for c in reranked}
    context_set = {c["chunk_idx"] for c in context_chunks}
    _print_stage("BM25 -> RRF ALL DROPPED", [f"chunk={c}" for c in sorted(bm25_set - rrf_all_set)])
    _print_stage("RRF ALL -> RRF TOP-K DROPPED", [f"chunk={c}" for c in rrf_ids_all if c not in rrf_kept_set])
    _print_stage("RRF TOP-K -> RERANK DROPPED", [f"chunk={c}" for c in rrf_ids_kept if c not in rerank_set])
    _print_stage("RERANK -> LLM CONTEXT DROPPED", [f"chunk={c['chunk_idx']}" for c in reranked if c["chunk_idx"] not in context_set])


def main():
    parser = argparse.ArgumentParser(description="Trace BM25, FAISS, RRF, rerank, and LLM context selection.")
    parser.add_argument("--doc", required=True, help="Indexed filing doc_name to inspect.")
    parser.add_argument("--query", required=True, help="Natural-language query to diagnose.")
    parser.add_argument("--top-k", type=int, default=RERANK_TOP_K, help="Final retrieval top_k before LLM context cap.")
    parser.add_argument("--embedding-model", choices=["normal", "finlang", "financesmall"], default=None, help="Embedding model override. Defaults to EMBEDDING_MODEL or normal.")
    args = parser.parse_args()
    if args.embedding_model:
        os.environ["EMBEDDING_MODEL"] = args.embedding_model
    diagnose(args.query, args.doc, args.top_k)


if __name__ == "__main__":
    main()
