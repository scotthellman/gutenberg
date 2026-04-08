"""RAG pipeline: retrieve → RRF → rerank → evaluate → answer."""

from dataclasses import dataclass, field

import psycopg
from pgvector.psycopg import register_vector_async
from pydantic import BaseModel
from pydantic_ai import Agent, Embedder, RunContext
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.usage import UsageLimitExceeded, UsageLimits

from gutenrag import consts, db
from gutenrag.db import ModelConfig

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class Retriever:
    def __init__(self, model: ModelConfig, conn: psycopg.AsyncConnection):
        self.conn = conn
        self.model = model

    async def dense_retrieve(
        self,
        query: str,
        top_k: int = 19,
    ) -> dict[str, list[tuple[int, str]]]:
        """Return {model_key: [(id, content), ...]} ranked by cosine similarity."""
        results: dict[str, list[tuple[int, str]]] = {}
        # FIXME: unify where I do this, it's scattered in a few places right now
        embedder = Embedder(f"ollama:{self.model.model}")
        result = await embedder.embed_query(query)
        embedding = result[0]
        results[self.model.key] = await db.retrieve(
            self.model.key, embedding, self.conn, top_k
        )
        return results

    async def keyword_retrieve(self, query: str, top_k: int = 19):
        results = {}
        results["fts"] = await db.retrieve_fts(query, self.conn, top_k)
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
# Agentic evidence evaluation
# ---------------------------------------------------------------------------

EVAL_INSTRUCTIONS = """\
You are evaluating retrieved documents for relevance to a user's query.
You have been given an initial set of documents, each prefixed with its ID.
If the documents are insufficient to answer the query, call fetch_more_docs \
with a refined search query to retrieve additional context.
When you have enough relevant evidence, return the IDs of the useful documents.
Do not answer the user's query, your only task is to ensure we have sufficient evidence."""


class EvidenceResult(BaseModel):
    doc_ids: list[int]


@dataclass
class EvalDeps:
    query: str
    retriever: Retriever
    seen_docs: dict[int, str] = field(default_factory=dict)


async def fetch_more_docs(ctx: RunContext[EvalDeps], search_query: str) -> str:
    """Fetch additional documents relevant to search_query.

    Returns new documents (not seen before) formatted as [ID <n>]\\n<content>.
    """
    ranked = await ctx.deps.retriever.dense_retrieve(search_query, top_k=5)
    ranked.update(await ctx.deps.retriever.keyword_retrieve(search_query, top_k=5))
    new_docs = []
    for doc_id, content, _ in rrf(ranked)[:5]:
        if doc_id not in ctx.deps.seen_docs:
            ctx.deps.seen_docs[doc_id] = content
            new_docs.append(f"[ID {doc_id}]\n{content}")
    return "\n---\n".join(new_docs) if new_docs else "No new documents found."


async def evaluate_docs(
    query: str,
    initial_docs: list[tuple[int, str, float]],
    retriever: Retriever,
    model: str,
    max_rounds: int = 3,
) -> list[int]:
    """Agent that evaluates initial docs and may fetch more, returning chosen IDs."""
    print(model)
    eval_agent: Agent[EvalDeps, EvidenceResult] = Agent(
        f"ollama:{model}",
        deps_type=EvalDeps,
        output_type=str,
        instructions=EVAL_INSTRUCTIONS,
        model_settings={"max_tokens": 256, "timeout": 120},
        tools=[fetch_more_docs],
    )
    seen = {doc_id: content for doc_id, content, _ in initial_docs}
    formatted = "\n---\n".join(
        f"[ID {doc_id}]\n{content}" for doc_id, content, _ in initial_docs
    )
    user_msg = f"Query: {query}\n\nInitial documents:\n{formatted}"

    deps = EvalDeps(query=query, retriever=retriever, seen_docs=seen)
    try:
        result = await eval_agent.run(
            user_msg,
            deps=deps,
            usage_limits=UsageLimits(request_limit=max_rounds),
        )
        for message in result.all_messages():
            print(message)
            print("---" * 3)

    except UsageLimitExceeded:
        # Budget exhausted — return all accumulated IDs
        # maybe log this or something
        pass
    return list(deps.seen_docs.keys())


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
You will be presented with a question. Answer it using only the following context:

{context}

---

Answer the following question based on the above context."""


async def answer(
    query: str, docs: list[tuple[int, str, float]], model: str
) -> str | None:
    context = "\n---\n".join(text for _, text, _ in docs)
    instructions = PROMPT_TEMPLATE.format(context=context)
    agent = Agent(
        f"ollama:{model}",
        model_settings={"max_tokens": 128, "timeout": 120},
    )
    try:
        result = await agent.run(query, instructions=instructions)
    except ModelHTTPError:
        return None

    return result.output


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


async def rag(
    query: str,
    embedding_model: ModelConfig,
    llm_model: str = "ministral-3:3b",
    top_k: int = 20,
    top_n: int = 5,
    eval_rounds: int = 3,
) -> str:
    conninfo = (
        f"host={consts.PG_HOST} port={consts.PG_PORT} "
        f"dbname={consts.PG_DB} user={consts.PG_USER} password={consts.PG_PASSWORD}"
    )

    async with await psycopg.AsyncConnection.connect(conninfo) as aconn:
        await register_vector_async(aconn)
        retriever = Retriever(embedding_model, aconn)

        ranked = await retriever.dense_retrieve(query, top_k=top_k)
        ranked.update(await retriever.keyword_retrieve(query, top_k=top_k))
        fused = rrf(ranked)
        initial_docs = rerank(query, fused)[:top_n]

        chosen_ids = await evaluate_docs(
            query, initial_docs, retriever, llm_model, max_rounds=eval_rounds
        )

        id_to_content = {doc_id: content for doc_id, content, _ in initial_docs}
        id_to_content.update(
            {
                doc_id: content
                for doc_id, content in await db.fetch_by_ids(chosen_ids, aconn)
            }
        )

    chosen_docs = [
        (doc_id, id_to_content[doc_id], 0.0)
        for doc_id in chosen_ids
        if doc_id in id_to_content
    ]

    response = await answer(query, chosen_docs, llm_model)
    if response is None:
        # TODO: Better error handling
        return "No response received from LLM"
    return response
