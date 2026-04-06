"""Evaluate retrieval quality against a generated test set."""

import argparse
import json
import os

import ollama
import psycopg
from pgvector.psycopg import register_vector

from gutenrag.db import MODELS
from gutenrag.rag import rerank, retrieve, rrf

RECALL_AT = [1, 5, 10, 20]


def evaluate(
    testset_path: str, top_k: int, ollama_host: str = "http://localhost:11434"
) -> None:
    pg_user = os.environ.get("POSTGRES_USER", "postgres")
    pg_password = os.environ.get("POSTGRES_PASSWORD", "")
    pg_db = os.environ.get("POSTGRES_DB", "postgres")
    pg_host = os.environ.get("PGVECTOR_HOST", "localhost")
    pg_port = os.environ.get("PGVECTOR_PORT", "5432")
    conninfo = (
        f"host={pg_host} port={pg_port} "
        f"dbname={pg_db} user={pg_user} password={pg_password}"
    )

    with open(testset_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    client = ollama.Client(host=ollama_host)

    hits: dict[int, int] = {k: 0 for k in RECALL_AT}
    reciprocal_ranks: list[float] = []
    skipped = 0

    with psycopg.connect(conninfo) as conn:
        register_vector(conn)
        for i, record in enumerate(records, start=1):
            question = record["question"]
            correct_id = record["chunk_id"]

            try:
                ranked = retrieve(
                    question, MODELS, conn, client, top_k=max(RECALL_AT + [top_k])
                )
            except ollama.ResponseError as e:
                print(f"[{i}/{len(records)}] SKIP (embed error: {e})  {question[:80]}")
                reciprocal_ranks.append(0.0)
                skipped += 1
                continue

            fused = rrf(ranked)
            reranked = rerank(question, fused)
            retrieved_ids = [doc_id for doc_id, _, _ in reranked]

            # Reciprocal rank
            try:
                rank = retrieved_ids.index(correct_id) + 1  # 1-based
                reciprocal_ranks.append(1.0 / rank)
            except ValueError:
                reciprocal_ranks.append(0.0)
                rank = None

            for k in RECALL_AT:
                if correct_id in retrieved_ids[:k]:
                    hits[k] += 1

            rank_str = str(rank) if rank is not None else "not found"
            print(f"[{i}/{len(records)}] rank={rank_str}  {question[:80]}")

    n = len(records)
    evaluated = n - skipped
    print(f"\n{'─' * 40}")
    print(
        f"Results over {evaluated}/{n} questions ({skipped} skipped due to embed errors)\n"
    )
    for k in RECALL_AT:
        print(f"  Recall@{k:<3} {hits[k] / n:.3f}  ({hits[k]}/{n})")
    mrr = sum(reciprocal_ranks) / n
    print(f"  MRR      {mrr:.3f}")
    print(f"{'─' * 40}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval on a test set."
    )
    parser.add_argument(
        "testset", help="Path to testset.jsonl produced by generate_testset.py"
    )
    parser.add_argument(
        "--top-k", type=int, default=20, help="Retrieval depth (default: 20)"
    )
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    args = parser.parse_args()

    evaluate(
        testset_path=args.testset,
        top_k=args.top_k,
        ollama_host=args.ollama_host,
    )
