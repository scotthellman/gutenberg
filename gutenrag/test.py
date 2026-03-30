"""Sanity checks for Ollama and pgvector connectivity."""

import os
import sys

import ollama
import psycopg
from pgvector.psycopg import register_vector


def check(label: str, fn) -> bool:
    try:
        fn()
        print(f"PASS  {label}")
        return True
    except Exception as e:
        print(f"FAIL  {label}: {e}")
        return False


def main():
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    pg_user = os.environ.get("POSTGRES_USER", "postgres")
    pg_password = os.environ.get("POSTGRES_PASSWORD", "")
    pg_db = os.environ.get("POSTGRES_DB", "postgres")
    pg_host = os.environ.get("PGVECTOR_HOST", "localhost")
    pg_port = os.environ.get("PGVECTOR_PORT", "5432")

    results = []

    # --- Ollama ---

    client = ollama.Client(host=ollama_host)

    def ollama_reachable():
        client.list()

    results.append(check("Ollama reachable", ollama_reachable))

    def ollama_has_models():
        models = client.list().models
        if not models:
            raise RuntimeError("no models pulled — run: ollama pull <model>")

    results.append(check("Ollama has models", ollama_has_models))

    # --- pgvector ---

    conninfo = (
        f"host={pg_host} port={pg_port} "
        f"dbname={pg_db} user={pg_user} password={pg_password}"
    )

    def pg_reachable():
        with psycopg.connect(conninfo) as conn:
            conn.execute("SELECT 1")

    results.append(check("pgvector reachable", pg_reachable))

    def pgvector_works():
        with psycopg.connect(conninfo) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            register_vector(conn)
            conn.execute(
                "CREATE TEMP TABLE _vec_test (id serial PRIMARY KEY, v vector(3))"
            )
            conn.execute("INSERT INTO _vec_test (v) VALUES ('[1,2,3]'), ('[4,5,6]')")
            result = conn.execute(
                "SELECT id FROM _vec_test ORDER BY v <-> '[1,2,4]' LIMIT 1"
            ).fetchone()
            if result is None:
                raise RuntimeError("nearest-neighbor query returned no rows")

    results.append(check("pgvector extension + query", pgvector_works))

    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
