"""Ingest Project Gutenberg ZIM archive into pgvector."""

import os
import sys

import ollama
import psycopg
from bs4 import BeautifulSoup
from pgvector.psycopg import register_vector
from zimscraperlib import zim

from gutenrag.db import MODELS, ModelConfig, setup_tables


def chunk_seq(words: list[str], window: int = 300, stride: int = 200) -> list[str]:
    chunks = []
    for i in range(0, max(1, len(words) - window + 1), stride):
        chunks.append(" ".join(words[i : i + window]))
    return chunks


def entry_text(entry):
    item = entry.get_item()
    html = bytes(item.content).decode("utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    pages = soup.find_all("div", class_="bodytext")
    if pages:
        for page in pages:
            yield page.text
    else:
        yield soup.text


def ingest(
    zim_path: str,
    models: list[ModelConfig] = MODELS,
    limit: int = 500,
    ollama_host: str = "http://localhost:11434",
    clear: bool = False,
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

    with psycopg.connect(conninfo) as conn:
        register_vector(conn)
        setup_tables(conn, models)

        if clear:
            for m in models:
                conn.execute(f"TRUNCATE TABLE chunks_{m.key}")
            conn.commit()
            print("Cleared existing chunks.")

        reader = zim.Archive(zim_path)
        count = 0

        for idx in range(reader.entry_count):
            entry = reader.get_entry_by_id(idx)
            suffix = entry.path.split(".")[-1]
            if count > limit:
                break
            if suffix.isdigit():
                if "_cover" not in entry.path:
                    source = str(entry.path)
                    chunk_count = 0
                    for i, text in enumerate(entry_text(entry)):
                        words = text.split()
                        chunks = chunk_seq(words)
                        chunk_count += len(chunks)

                        for chunk_idx, chunk in enumerate(chunks):
                            for m in models:
                                embedding = client.embed(
                                    model=m.model, input=chunk
                                ).embeddings[0]
                                conn.execute(
                                    f"""
                                    INSERT INTO chunks_{m.key} (source, chunk_idx, content, embedding)
                                    VALUES (%s, %s, %s, %s)
                                    ON CONFLICT (source, chunk_idx) DO NOTHING
                                    """,
                                    (source, chunk_idx, chunk, embedding),
                                )

                        conn.commit()
                    count += 1
                    print(f"[{count}/{limit}] {idx} {source} ({chunk_count} chunks)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m gutenrag.prototype.ingest <zim_path> [limit]")
        sys.exit(1)
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest a Gutenberg ZIM archive into pgvector."
    )
    parser.add_argument("zim_path")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument(
        "--clear", action="store_true", help="Truncate chunk tables before ingesting"
    )
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    args = parser.parse_args()
    ingest(
        args.zim_path, limit=args.limit, clear=args.clear, ollama_host=args.ollama_host
    )
