"""Generate a synthetic QA test set by sampling chunks and prompting an LLM."""

import argparse
import json
import os
import sys

import ollama
import psycopg
from pgvector.psycopg import register_vector

from gutenrag.prototype.db import MODELS, ModelConfig

QUESTION_PROMPT = """\
Given the following passage, write one factual question whose answer is contained \
in the passage. Output only the question, nothing else.

{content}"""


def generate_testset(
    limit: int,
    output: str,
    llm_model: str,
    embedding_model: ModelConfig,
    ollama_host: str = "http://localhost:11434",
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

    client = ollama.Client(host=ollama_host)
    table = f"chunks_{embedding_model.key}"

    with psycopg.connect(conninfo) as conn:
        register_vector(conn)
        rows = conn.execute(
            f"SELECT id, source, chunk_idx, content FROM {table} ORDER BY RANDOM() LIMIT %s",
            (limit,),
        ).fetchall()

    print(f"Sampled {len(rows)} chunks from {table}. Generating questions…")

    with open(output, "w") as f:
        for i, (chunk_id, source, chunk_idx, content) in enumerate(rows, start=1):
            prompt = QUESTION_PROMPT.format(content=content)
            response = client.chat(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
            )
            question = response.message.content.strip()
            record = {
                "chunk_id": chunk_id,
                "source": source,
                "chunk_idx": chunk_idx,
                "content": content,
                "question": question,
            }
            f.write(json.dumps(record) + "\n")
            print(f"[{i}/{len(rows)}] {question}")

    print(f"\nWrote {len(rows)} records to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a RAG evaluation test set.")
    parser.add_argument("--limit", type=int, default=100, help="Number of chunks to sample")
    parser.add_argument("--output", default="testset.jsonl", help="Output JSONL file path")
    parser.add_argument("--model", default="ministral-3:3b", help="Ollama LLM for question generation")
    parser.add_argument(
        "--embedding-model",
        default=MODELS[0].key,
        choices=[m.key for m in MODELS],
        help="Embedding model key (determines which chunk table to sample from)",
    )
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    args = parser.parse_args()

    embedding_model = next(m for m in MODELS if m.key == args.embedding_model)

    generate_testset(
        limit=args.limit,
        output=args.output,
        llm_model=args.model,
        embedding_model=embedding_model,
        ollama_host=args.ollama_host,
    )
