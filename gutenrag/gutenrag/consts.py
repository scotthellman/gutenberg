import os

PG_USER = os.environ.get("POSTGRES_USER", "postgres")
PG_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "")
PG_DB = os.environ.get("POSTGRES_DB", "postgres")
PG_HOST = os.environ.get("PGVECTOR_HOST", "localhost")
PG_PORT = os.environ.get("PGVECTOR_PORT", "5432")
