"""RAG pipeline: retrieve → RRF → rerank → evaluate → answer."""

from __future__ import annotations

from dataclasses import dataclass, field

import psycopg
from pgvector.psycopg import register_vector_async
from pydantic import BaseModel
from pydantic_ai import Agent, Embedder, RunContext, capture_run_messages
from pydantic_ai.exceptions import (
    ModelHTTPError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
)
from pydantic_ai.usage import UsageLimits
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

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
        sources: list[str] | None = None,
    ) -> dict[str, list[tuple[int, str]]]:
        """Return {model_key: [(id, content), ...]} ranked by cosine similarity."""
        results: dict[str, list[tuple[int, str]]] = {}
        # FIXME: unify where I do this, it's scattered in a few places right now
        embedder = Embedder(f"ollama:{self.model.model}")
        result = await embedder.embed_query(query)
        embedding = result[0]
        results[self.model.key] = await db.retrieve(
            self.model.key, embedding, self.conn, top_k, sources=sources
        )
        return results

    async def keyword_retrieve(
        self,
        query: str,
        top_k: int = 19,
        sources: list[str] | None = None,
    ):
        results = {}
        results["fts"] = await db.retrieve_fts(query, self.conn, top_k, sources=sources)
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

BOOK_FILTER_INSTRUCTIONS = """\
You are determining whether a user's query is about specific book(s) in a Project Gutenberg library.
If the query mentions specific book(s), use search_books to find their source paths.
You may call search_books multiple times with different terms (title words, author name, etc.).
If the query is general knowledge not tied to specific titles, return sources as null.
Only return source paths returned by search_books — do not invent paths."""


class BookFilter(BaseModel):
    sources: list[str] | None
    """Matched source paths, or None if the query is not about specific book(s)."""


@dataclass
class BookFilterDeps:
    conn: psycopg.AsyncConnection
    sources: list[str]


async def search_books(ctx: RunContext[BookFilterDeps], search_query: str) -> str:
    """Search for book source paths matching search_query (trigram similarity)."""
    sources = await db.search_sources(search_query, ctx.deps.conn)
    print("We did search, and got back", sources)
    ctx.deps.sources.extend(sources)
    return "\n".join(sources) if sources else "No matches found."


async def identify_book_sources(
    query: str,
    conn: psycopg.AsyncConnection,
    model: str,
) -> list[str] | None:
    """Return source path(s) if the query targets specific book(s), else None."""
    agent: Agent[BookFilterDeps, BookFilter] = Agent(
        f"ollama:{model}",
        deps_type=BookFilterDeps,
        output_type=BookFilter,
        instructions=BOOK_FILTER_INSTRUCTIONS,
        model_settings={"max_tokens": 256, "timeout": 60},
        tools=[search_books],
    )
    deps = BookFilterDeps(conn=conn, sources=[])
    with capture_run_messages() as messages:
        try:
            result = await agent.run(query, deps=deps)
            for message in result.all_messages():
                print(message)
                print("---" * 3)
        except ModelHTTPError:
            return None
        except UnexpectedModelBehavior:
            print(messages)
            raise

    return deps.sources or result.output.sources or None


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
    sources: list[str] | None = None
    seen_docs: dict[int, str] = field(default_factory=dict)


async def fetch_more_docs(ctx: RunContext[EvalDeps], search_query: str) -> str:
    """Fetch additional documents relevant to search_query.

    Returns new documents (not seen before) formatted as [ID <n>]\\n<content>.
    """
    ranked = await ctx.deps.retriever.dense_retrieve(
        search_query, top_k=5, sources=ctx.deps.sources
    )
    ranked.update(
        await ctx.deps.retriever.keyword_retrieve(
            search_query, top_k=5, sources=ctx.deps.sources
        )
    )
    new_docs = []
    for doc_id, content, _ in rrf(ranked)[:5]:
        if doc_id not in ctx.deps.seen_docs:
            ctx.deps.seen_docs[doc_id] = content
            new_docs.append(f"[ID {doc_id}]\n{content}")
    return "\n---\n".join(new_docs) if new_docs else "No new documents found."


async def evaluate_docs(
    query: str,
    initial_docs: list[tuple[int, str]],
    retriever: Retriever,
    model: str,
    max_rounds: int = 3,
    sources: list[str] | None = None,
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
    seen = {doc_id: content for doc_id, content in initial_docs}
    formatted = "\n---\n".join(
        f"[ID {doc_id}]\n{content}" for doc_id, content in initial_docs
    )
    user_msg = f"Query: {query}\n\nInitial documents:\n{formatted}"

    deps = EvalDeps(query=query, retriever=retriever, sources=sources, seen_docs=seen)
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


async def answer(query: str, docs: list[tuple[int, str]], model: str) -> str | None:
    context = "\n---\n".join(text for _, text in docs)
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

# ok so to frame this as a graph, we need states like:
# get sources
# retrieve initial set
# evaluate docs
# retrieve more docs
# generate final response


@dataclass
class RagContext:
    query: str
    sources: list[str] | None
    embedding_model: ModelConfig
    llm_model: str
    top_k: int
    top_n: int
    eval_rounds: int
    docs: list[tuple[int, str]]


@dataclass
class RagDeps:
    aconn: psycopg.AsyncConnection


@dataclass
class GetSources(BaseNode[RagContext, RagDeps]):
    async def run(self, ctx: GraphRunContext[RagContext, RagDeps]) -> BuildInitialSet:
        sources = await identify_book_sources(
            ctx.state.query, ctx.deps.aconn, ctx.state.llm_model
        )
        ctx.state.sources = sources
        return BuildInitialSet()


@dataclass
class BuildInitialSet(BaseNode[RagContext, RagDeps]):
    async def run(self, ctx: GraphRunContext[RagContext, RagDeps]) -> EvaluateDocs:
        retriever = Retriever(ctx.state.embedding_model, ctx.deps.aconn)

        ranked = await retriever.dense_retrieve(
            ctx.state.query, top_k=ctx.state.top_k, sources=ctx.state.sources
        )
        ranked.update(
            await retriever.keyword_retrieve(
                ctx.state.query, top_k=ctx.state.top_k, sources=ctx.state.sources
            )
        )
        fused = rrf(ranked)
        initial_docs = rerank(ctx.state.query, fused)[: ctx.state.top_n]
        ctx.state.docs = [d[:2] for d in initial_docs]
        return EvaluateDocs()


@dataclass
class EvaluateDocs(BaseNode[RagContext, RagDeps]):
    async def run(self, ctx: GraphRunContext[RagContext, RagDeps]) -> AnswerQuery:
        retriever = Retriever(ctx.state.embedding_model, ctx.deps.aconn)
        chosen_ids = await evaluate_docs(
            ctx.state.query,
            ctx.state.docs,
            retriever,
            ctx.state.llm_model,
            max_rounds=ctx.state.eval_rounds,
            sources=ctx.state.sources,
        )

        id_to_content = {doc_id: content for doc_id, content in ctx.state.docs}
        id_to_content.update(
            {
                doc_id: content
                for doc_id, content in await db.fetch_by_ids(chosen_ids, ctx.deps.aconn)
            }
        )
        filtered_content = [(i, c) for i, c in id_to_content.items() if i in chosen_ids]
        return AnswerQuery(filtered_content)


@dataclass
class AnswerQuery(BaseNode[RagContext, RagDeps, str]):
    chosen_docs: list[tuple[int, str]]

    async def run(self, ctx: GraphRunContext[RagContext, RagDeps]) -> End[str]:
        response = await answer(ctx.state.query, self.chosen_docs, ctx.state.llm_model)
        if response is None:
            # TODO: Better error handling
            return End("No response received from LLM")
        return End(response)


async def rag(
    query: str,
    embedding_model: ModelConfig,
    llm_model: str = "ministral-3:3b",
    top_k: int = 20,
    top_n: int = 5,
    eval_rounds: int = 3,
) -> str:
    rag_graph = Graph(nodes=[GetSources, BuildInitialSet, EvaluateDocs, AnswerQuery])
    conninfo = (
        f"host={consts.PG_HOST} port={consts.PG_PORT} "
        f"dbname={consts.PG_DB} user={consts.PG_USER} password={consts.PG_PASSWORD}"
    )

    ctx = RagContext(
        query=query,
        sources=None,
        embedding_model=embedding_model,
        llm_model=llm_model,
        top_k=top_k,
        top_n=top_n,
        eval_rounds=eval_rounds,
        docs=[],
    )

    async with await psycopg.AsyncConnection.connect(conninfo) as aconn:
        await register_vector_async(aconn)
        deps = RagDeps(aconn=aconn)

        result = await rag_graph.run(GetSources(), state=ctx, deps=deps)
    return result.output
