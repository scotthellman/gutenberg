"""Database schema setup for the RAG prototype."""

from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector
from psycopg.sql import SQL, Identifier


@dataclass
class ModelConfig:
    key: str  # used as table name suffix, e.g. "bge_m3"
    model: str  # Ollama model name, e.g. "bge-m3:latest"
    dim: int  # embedding dimension


MODELS: list[ModelConfig] = [
    ModelConfig(key="bge_m3", model="bge-m3:latest", dim=1024),
]


def setup_tables(conn: psycopg.Connection, models: list[ModelConfig] = MODELS) -> None:
    register_vector(conn)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    for m in models:
        table = f"chunks_{m.key}"
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id        BIGSERIAL PRIMARY KEY,
                source    TEXT NOT NULL,
                chunk_idx INT  NOT NULL,
                content   TEXT NOT NULL,
                embedding vector({m.dim}) NOT NULL,
                UNIQUE (source, chunk_idx)
            )
        """)
        conn.execute(
            SQL("""
            CREATE INDEX IF NOT EXISTS {}_embedding_idx
            ON {} USING hnsw (embedding vector_cosine_ops)
        """).format(Identifier(table), Identifier(table))
        )

    conn.commit()


def retrieve(
    key: str,
    embedding: list[float],
    conn: psycopg.Connection,
    top_k: int = 19,
) -> list[tuple[int, str]]:
    """Return {model_key: [(id, content), ...]} ranked by cosine similarity."""
    table_name = f"chunks_{key}"
    rows = conn.execute(
        SQL(
            "SELECT id, content FROM {} ORDER BY embedding <-> %s::vector LIMIT %s"
        ).format(Identifier(table_name)),
        (embedding, top_k),
    ).fetchall()
    result = [(row[-1], row[1]) for row in rows]
    return result
