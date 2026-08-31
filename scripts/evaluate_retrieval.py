"""
Generic SEC Information Retrieval Benchmark Evaluation Harness.

Measures:
- Recall@5, Recall@10, Recall@20
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG@10)

Compares:
1. BM25 Only
2. BGE / FAISS Only
3. BM25 + BGE + RRF
4. BM25 + BGE + RRF + Structural Reranking
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from retrieval import FilingIndex, get_index, list_indexed_docs, _rrf_fuse, deterministic_rerank
from query_analyzer import analyze_query, expand_query
from config import BM25_TOP_K, SEMANTIC_TOP_K, RRF_K, RRF_TOP_K, RERANK_TOP_K, get_embedding_model_name


def dcg_at_k(relevances: List[int], k: int = 10) -> float:
    rel = relevances[:k]
    if not rel:
        return 0.0
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rel))


def ndcg_at_k(relevances: List[int], k: int = 10) -> float:
    dcg = dcg_at_k(relevances, k)
    ideal = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_query(index: FilingIndex, question: str, target_keywords: List[str]) -> Dict[str, Dict[str, float]]:
    """Evaluate single question across 4 retrieval modes."""
    # 1. BM25 Only
    bm25_indices = index.search_bm25(question, top_k=20)

    # 2. FAISS Only
    faiss_hits = index.search_bge_faiss(question, top_k=20)
    faiss_indices = [idx for idx, _ in faiss_hits]

    # 3. BM25 + BGE + RRF
    fused_scores = _rrf_fuse([(bm25_indices, 1.0), (faiss_indices, 1.0)], k=RRF_K)
    rrf_indices = sorted(fused_scores.keys(), key=lambda i: fused_scores[i], reverse=True)[:20]

    # 4. BM25 + BGE + RRF + Structural Reranking
    candidates = []
    for i in rrf_indices:
        if i < len(index.chunks):
            c = dict(index.chunks[i])
            c["chunk_idx"] = i
            c["retrieval_score"] = fused_scores[i]
            c["bm25_rank"] = (bm25_indices.index(i) + 1) if i in bm25_indices else None
            c["semantic_rank"] = (faiss_indices.index(i) + 1) if i in faiss_indices else None
            candidates.append(c)

    qa = analyze_query(question)
    reranked_chunks = deterministic_rerank(question, candidates, query_info=qa)[:20]
    reranked_indices = [c["chunk_idx"] for c in reranked_chunks]

    modes = {
        "BM25 Only": bm25_indices,
        "FAISS Only": faiss_indices,
        "RRF Fusion": rrf_indices,
        "RRF + Reranker": reranked_indices,
    }

    metrics = {}
    for mode_name, indices in modes.items():
        relevances = []
        hit_ranks = []

        for rank, idx in enumerate(indices, 1):
            if idx < len(index.chunks):
                text = index.chunks[idx].get("text", "").lower()
                is_hit = any(kw.lower() in text for kw in target_keywords)
                rel = 1 if is_hit else 0
                relevances.append(rel)
                if is_hit:
                    hit_ranks.append(rank)

        r5 = 1.0 if any(r <= 5 for r in hit_ranks) else 0.0
        r10 = 1.0 if any(r <= 10 for r in hit_ranks) else 0.0
        r20 = 1.0 if hit_ranks else 0.0
        mrr = 1.0 / hit_ranks[0] if hit_ranks else 0.0
        ndcg10 = ndcg_at_k(relevances, k=10)

        metrics[mode_name] = {
            "Recall@5": r5,
            "Recall@10": r10,
            "Recall@20": r20,
            "MRR": mrr,
            "nDCG@10": ndcg10,
        }

    return metrics


def run_benchmark(jsonl_path: Path):
    """Run IR benchmark over dataset file."""
    if not jsonl_path.exists():
        print(f"Benchmark file {jsonl_path} not found.")
        return

    questions = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    print(f"Loaded {len(questions)} test cases from {jsonl_path.name}.\n")

    results = {
        "BM25 Only": {"Recall@5": [], "Recall@10": [], "Recall@20": [], "MRR": [], "nDCG@10": []},
        "FAISS Only": {"Recall@5": [], "Recall@10": [], "Recall@20": [], "MRR": [], "nDCG@10": []},
        "RRF Fusion": {"Recall@5": [], "Recall@10": [], "Recall@20": [], "MRR": [], "nDCG@10": []},
        "RRF + Reranker": {"Recall@5": [], "Recall@10": [], "Recall@20": [], "MRR": [], "nDCG@10": []},
    }

    evaluated_cnt = 0
    for q in questions[:10]:
        doc_name = q.get("doc_name")
        question = q.get("question")
        answer = q.get("answer", "")
        evidences = q.get("evidence", [])

        if not doc_name or not question:
            continue

        index = get_index(doc_name)
        if index is None or not index.is_indexed():
            continue

        target_keywords = [answer.strip("$").strip()]
        # Add integer formatted variants (e.g. 1577 -> 1,577)
        ans_clean = answer.replace("$", "").replace(",", "").strip()
        try:
            val_float = float(ans_clean)
            val_int = int(val_float)
            target_keywords.append(str(val_int))
            target_keywords.append(f"{val_int:,}")
        except ValueError:
            pass

        for ev in evidences:
            text = ev.get("evidence_text", "")
            if text:
                target_keywords.append(text[:60])

        q_metrics = evaluate_query(index, question, target_keywords)
        evaluated_cnt += 1

        for mode_name, m_dict in q_metrics.items():
            for m_key, m_val in m_dict.items():
                results[mode_name][m_key].append(m_val)

    print("===================================================================================")
    print(f"RETRIEVAL EVALUATION SUMMARY ({evaluated_cnt} Questions Evaluated)")
    print(f"EMBEDDING MODEL: {get_embedding_model_name()}")
    print("===================================================================================")
    print(f"{'Retrieval Mode':<22} | {'Recall@5':<9} | {'Recall@10':<9} | {'Recall@20':<9} | {'MRR':<8} | {'nDCG@10':<8}")
    print("-----------------------------------------------------------------------------------")

    for mode_name, m_dict in results.items():
        if evaluated_cnt == 0:
            avg_r5 = avg_r10 = avg_r20 = avg_mrr = avg_ndcg = 0.0
        else:
            avg_r5 = sum(m_dict["Recall@5"]) / len(m_dict["Recall@5"]) if m_dict["Recall@5"] else 0.0
            avg_r10 = sum(m_dict["Recall@10"]) / len(m_dict["Recall@10"]) if m_dict["Recall@10"] else 0.0
            avg_r20 = sum(m_dict["Recall@20"]) / len(m_dict["Recall@20"]) if m_dict["Recall@20"] else 0.0
            avg_mrr = sum(m_dict["MRR"]) / len(m_dict["MRR"]) if m_dict["MRR"] else 0.0
            avg_ndcg = sum(m_dict["nDCG@10"]) / len(m_dict["nDCG@10"]) if m_dict["nDCG@10"] else 0.0

        print(f"{mode_name:<22} | {avg_r5:<9.4f} | {avg_r10:<9.4f} | {avg_r20:<9.4f} | {avg_mrr:<8.4f} | {avg_ndcg:<8.4f}")
    print("===================================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval over practice questions.")
    parser.add_argument("--input", type=Path, default=Path("C:/Users/Sakshi Sinha/Downloads/analyst-copilot-data 1/analyst-copilot-data/practice-questions.jsonl"))
    parser.add_argument("--embedding-model", choices=["normal", "finlang", "financesmall"], default=None, help="Embedding model override. Defaults to EMBEDDING_MODEL or normal.")
    args = parser.parse_args()
    if args.embedding_model:
        os.environ["EMBEDDING_MODEL"] = args.embedding_model
    run_benchmark(args.input)
