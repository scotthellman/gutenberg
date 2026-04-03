"""Database schema setup for the RAG prototype."""

from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector


@dataclass
class ModelConfig:
    key: str    # used as table name suffix, e.g. "bge_m3"
    model: str  # Ollama model name, e.g. "bge-m3:latest"
    dim: int    # embedding dimension


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
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {table}_embedding_idx
            ON {table} USING hnsw (embedding vector_cosine_ops)
        """)

    conn.commit()
