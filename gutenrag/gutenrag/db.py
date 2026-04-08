"""Database schema setup for the RAG prototype."""

from dataclasses import dataclass
from typing import Sequence

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

    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id            BIGSERIAL PRIMARY KEY,
            source        TEXT NOT NULL,
            chunk_idx     INT  NOT NULL,
            content       TEXT NOT NULL,
            search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
            UNIQUE (source, chunk_idx)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS chunks_search_vector_idx
        ON chunks USING GIN (search_vector)
    """)

    for m in models:
        conn.execute(
            SQL(f"""
                CREATE TABLE IF NOT EXISTS {{}} (
                    chunk_id  BIGINT PRIMARY KEY REFERENCES chunks(id),
                    embedding vector({m.dim}) NOT NULL
                )
            """).format(Identifier(f"embeddings_{m.key}"))
        )
        conn.execute(
            SQL("""
                CREATE INDEX IF NOT EXISTS {idx}
                ON {tbl} USING hnsw (embedding vector_cosine_ops)
            """).format(
                idx=Identifier(f"embeddings_{m.key}_embedding_idx"),
                tbl=Identifier(f"embeddings_{m.key}"),
            )
        )

    conn.commit()


async def retrieve(
    key: str,
    embedding: Sequence[float],
    conn: psycopg.AsyncConnection,
    top_k: int = 19,
) -> list[tuple[int, str]]:
    """Return [(id, content), ...] ranked by cosine similarity."""
    # TODO: Is there any reason to stream this?
    result = []
    curs = await conn.execute(
        SQL("""
            SELECT c.id, c.content
            FROM {} e
            JOIN chunks c ON c.id = e.chunk_id
            ORDER BY e.embedding <-> %s::vector
            LIMIT %s
        """).format(Identifier(f"embeddings_{key}")),
        (embedding, top_k),
    )
    rows = await curs.fetchall()
    for row in rows:
        result.append((row[0], row[1]))
    return result


async def retrieve_fts(
    query: str,
    conn: psycopg.AsyncConnection,
    top_k: int = 19,
) -> list[tuple[int, str]]:
    """Return [(id, content), ...] ranked by full-text search score."""
    curs = await conn.execute(
        """
        SELECT id, content
        FROM chunks
        WHERE search_vector @@ plainto_tsquery('english', %s)
        ORDER BY ts_rank(search_vector, plainto_tsquery('english', %s)) DESC
        LIMIT %s
        """,
        (query, query, top_k),
    )
    rows = await curs.fetchall()
    return [(row[0], row[1]) for row in rows]


async def fetch_by_ids(
    ids: list[int],
    conn: psycopg.AsyncConnection,
) -> list[tuple[int, str]]:
    """Fetch chunks by their IDs. Order is not guaranteed."""
    curs = await conn.execute(
        "SELECT id, content FROM chunks WHERE id = ANY(%s)",
        (ids,),
    )
    rows = await curs.fetchall()
    return [(row[0], row[1]) for row in rows]
