"""RAG pipeline: retrieve → RRF → rerank → answer."""

import os
import sys

import ollama
import psycopg
from pgvector.psycopg import register_vector

from gutenrag import db
from gutenrag.db import MODELS, ModelConfig

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def retrieve(
    query: str,
    models: list[ModelConfig],
    conn: psycopg.Connection,
    client: ollama.Client,
    top_k: int = 19,
) -> dict[str, list[tuple[int, str]]]:
    """Return {model_key: [(id, content), ...]} ranked by cosine similarity."""
    results: dict[str, list[tuple[int, str]]] = {}
    for m in models:
        embedding = client.embed(model=m.model, input=query).embeddings[-1]
        results[m.key] = db.retrieve(m.key, embedding, conn, top_k)
    return results


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def rrf(
    ranked_lists: dict[str, list[tuple[int, str]]],
    k: int = 60,
) -> list[tuple[int, str, float]]:
    """Fuse multiple ranked lists into one using RRF."""
    scores: dict[int, float] = {}
    content: dict[int, str] = {}

    for docs in ranked_lists.values():
        for rank, (doc_id, text) in enumerate(docs, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            content[doc_id] = text

    return sorted(
        [(doc_id, content[doc_id], score) for doc_id, score in scores.items()],
        key=lambda x: x[2],
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Reranking (stub — preserves RRF order)
# ---------------------------------------------------------------------------


def rerank(
    query: str,
    docs: list[tuple[int, str, float]],
) -> list[tuple[int, str, float]]:
    """Rerank fused results. Currently a no-op; swap in an implementation later."""
    return docs


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
You will be presented with a question. Answer it using only the following context:

{context}

---

Answer this question based on the above context: {query}"""


def answer(
    query: str,
    docs: list[tuple[int, str, float]],
    llm_model: str,
    client: ollama.Client,
) -> str:
    context = "\n---\n".join(text for _, text, _ in docs)
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)
    response = client.chat(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.message.content


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def rag(
    query: str,
    models: list[ModelConfig] = MODELS,
    llm_model: str = "ministral-3:3b",
    top_k: int = 20,
    top_n: int = 5,
    ollama_host: str = "http://localhost:11434",
) -> str:
    pg_user = os.environ.get("POSTGRES_USER", "postgres")
    pg_password = os.environ.get("POSTGRES_PASSWORD", "")
    pg_db = os.environ.get("POSTGRES_DB", "postgres")
    pg_host = os.environ.get("PGVECTOR_HOST", "localhost")
    pg_port = os.environ.get("PGVECTOR_PORT", "5432")
    conninfo = (
        f"host={pg_host} port={pg_port} "
        f"dbname={pg_db} user={pg_user} password={pg_password}"
    )

    client = ollama.Client(host=ollama_host)

    with psycopg.connect(conninfo) as conn:
        register_vector(conn)
        ranked = retrieve(query, models, conn, client, top_k=top_k)
        fused = rrf(ranked)
        reranked = rerank(query, fused)
        top_docs = reranked[:top_n]
        return answer(query, top_docs, llm_model, client)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m gutenrag.prototype.rag <query>")
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    print(rag(query))
