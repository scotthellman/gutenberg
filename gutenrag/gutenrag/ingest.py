"""Ingest Project Gutenberg ZIM archive into pgvector."""

import sys

import ollama
import psycopg
from bs4 import BeautifulSoup
from pgvector.psycopg import register_vector
from psycopg.sql import SQL, Identifier
from zimscraperlib import zim

from gutenrag import consts
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
    conninfo = (
        f"host={consts.PG_HOST} port={consts.PG_PORT} "
        f"dbname={consts.PG_DB} user={consts.PG_USER} password={consts.PG_PASSWORD}"
    )

    client = ollama.Client(host=ollama_host)

    with psycopg.connect(conninfo) as conn:
        register_vector(conn)
        setup_tables(conn, models)

        if clear:
            for m in models:
                conn.execute(
                    SQL("TRUNCATE TABLE {} CASCADE").format(
                        Identifier(f"embeddings_{m.key}")
                    )
                )
            conn.execute("TRUNCATE TABLE chunks CASCADE")
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
                            row = conn.execute(
                                """
                                INSERT INTO chunks (source, chunk_idx, content)
                                VALUES (%s, %s, %s)
                                ON CONFLICT (source, chunk_idx) DO UPDATE SET content = EXCLUDED.content
                                RETURNING id
                                """,
                                (source, chunk_idx, chunk),
                            ).fetchone()
                            chunk_id = row[0]

                            for m in models:
                                embedding = client.embed(
                                    model=m.model, input=chunk
                                ).embeddings[0]
                                conn.execute(
                                    SQL("""
                                        INSERT INTO {} (chunk_id, embedding)
                                        VALUES (%s, %s)
                                        ON CONFLICT (chunk_id) DO NOTHING
                                    """).format(Identifier(f"embeddings_{m.key}")),
                                    (chunk_id, embedding),
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
