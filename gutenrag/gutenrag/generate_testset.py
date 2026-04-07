"""Generate a synthetic QA test set by sampling chunks and prompting an LLM."""

import argparse
import json

import ollama
import psycopg
from pgvector.psycopg import register_vector

from gutenrag import consts

QUESTION_PROMPT = """\
Given the following passage, write one factual question whose answer is contained \
in the passage. Output only the question, nothing else.

{content}"""


def generate_testset(
    limit: int,
    output: str,
    llm_model: str,
    ollama_host: str = "http://localhost:11434",
) -> None:
    conninfo = (
        f"host={consts.PG_HOST} port={consts.PG_PORT} "
        f"dbname={consts.PG_DB} user={consts.PG_USER} password={consts.PG_PASSWORD}"
    )

    client = ollama.Client(host=ollama_host)

    with psycopg.connect(conninfo) as conn:
        register_vector(conn)
        rows = conn.execute(
            "SELECT id, source, chunk_idx, content FROM chunks ORDER BY RANDOM() LIMIT %s",
            (limit,),
        ).fetchall()

    print(f"Sampled {len(rows)} chunks from chunks. Generating questions…")

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
    parser.add_argument(
        "--limit", type=int, default=100, help="Number of chunks to sample"
    )
    parser.add_argument(
        "--output", default="testset.jsonl", help="Output JSONL file path"
    )
    parser.add_argument(
        "--model", default="ministral-3:3b", help="Ollama LLM for question generation"
    )
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    args = parser.parse_args()

    generate_testset(
        limit=args.limit,
        output=args.output,
        llm_model=args.model,
        ollama_host=args.ollama_host,
    )
