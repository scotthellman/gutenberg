"""RAG pipeline: retrieve → RRF → rerank → answer."""

import sys

import ollama
import psycopg
from pgvector.psycopg import register_vector
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError

from gutenrag import consts, db
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
    results["fts"] = db.retrieve_fts(query, conn, top_k)
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

Answer the following question based on the above context."""


def answer(
    query: str,
    docs: list[tuple[int, str, float]],
    llm_model: str,
    client: ollama.Client,
) -> str | None:
    context = "\n---\n".join(text for _, text, _ in docs)
    instructions = PROMPT_TEMPLATE.format(context=context)
    # instructions = "foo bar"
    agent = Agent(
        "ollama:qwen3.5:0.8b",
        # Register static instructions using a keyword argument to the agent.
        # For more complex dynamically-generated instructions, see the example below.
        # instructions=instructions,
        model_settings={"max_tokens": 128, "timeout": 120},
    )
    # TODO: this can timeout, esp when i'm just running locally
    try:
        result = agent.run_sync(query, instructions=instructions)
    except ModelHTTPError:
        return None

    return result.output


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
    conninfo = (
        f"host={consts.PG_HOST} port={consts.PG_PORT} "
        f"dbname={consts.PG_DB} user={consts.PG_USER} password={consts.PG_PASSWORD}"
    )

    client = ollama.Client(host=ollama_host)

    with psycopg.connect(conninfo) as conn:
        register_vector(conn)
        ranked = retrieve(query, models, conn, client, top_k=top_k)
    fused = rrf(ranked)
    reranked = rerank(query, fused)
    top_docs = reranked[:top_n]
    response = answer(query, top_docs, llm_model, client)
    if response is None:
        # TODO: Better error handling
        return "No response received from LLM"
    return response


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m gutenrag.prototype.rag <query>")
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    print(rag(query))
