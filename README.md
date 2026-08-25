# rag-search-engine

A from-scratch Retrieval-Augmented Generation pipeline over a movie dataset, built to understand
every layer of a modern RAG system rather than call a single vector-DB SDK. Each stage — keyword
search, semantic search, hybrid retrieval, re-ranking, evaluation, generation, and multimodal
search — is implemented by hand and exposed through its own CLI.

## What's implemented

| Stage | CLI | Core idea |
|---|---|---|
| Keyword search | `cli/keyword_search_cli.py` | Hand-built inverted index + BM25 scoring (TF, IDF, BM25 TF/IDF) with stopword filtering and Porter stemming |
| Semantic search | `cli/semantic_search_cli.py` | Sentence-transformer embeddings (`all-MiniLM-L6-v2`), cosine similarity, fixed-size and sentence-boundary chunking |
| Hybrid search | `cli/hybrid_search_cli.py` | Score-normalized weighted fusion and Reciprocal Rank Fusion (RRF) of keyword + semantic results |
| Query enhancement | used by hybrid CLI (`--enhance`) | LLM-based spell correction, query rewriting, query expansion |
| Re-ranking | used by hybrid CLI (`--rerank-method`) | LLM re-ranking (individual/batch) and cross-encoder re-ranking |
| Evaluation | `cli/evaluation_cli.py` | Precision@k, recall@k, F1 against a hand-labeled golden dataset; optional LLM-as-judge relevance scoring |
| RAG / generation | `cli/augmented_generation_cli.py` | Search-grounded answers, multi-document summarization, cited answers |
| Multimodal search | `cli/multimodal_search_cli.py` | CLIP image embeddings searched against movie text embeddings |

All retrieval is over a local movie dataset (title + description); the LLM-backed features go
through [OpenRouter](https://openrouter.ai/).

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Copy `.env.example` to `.env` and set your OpenRouter key (only needed for query enhancement,
re-ranking, evaluation's LLM judge, RAG generation, and image-query rewriting):

```
OPENROUTER_API_KEY=sk-or-...
```

### Data

The dataset isn't checked into the repo (see `data/README.md` for the expected format and where
to get it). Once `data/movies.json` and `data/stopwords.txt` are in place, build the keyword
index once:

```bash
uv run python cli/keyword_search_cli.py build
```

Semantic/hybrid/multimodal search build and cache their embeddings automatically on first run
(`cache/`, gitignored).

## Usage

```bash
# Keyword (BM25)
uv run python cli/keyword_search_cli.py search "heist thriller"

# Semantic
uv run python cli/semantic_search_cli.py search "heist thriller" --limit 5

# Hybrid: Reciprocal Rank Fusion, with query rewriting and cross-encoder re-ranking
uv run python cli/hybrid_search_cli.py rrf-search "heist thriller" \
  --enhance rewrite --rerank-method cross_encoder

# Evaluate retrieval quality against the golden dataset
uv run python cli/evaluation_cli.py --limit 5

# RAG: retrieve + generate a grounded answer
uv run python cli/augmented_generation_cli.py rag "movies where a heist goes wrong"

# Search by image
uv run python cli/multimodal_search_cli.py image_search data/paddington.jpeg
```

Run any CLI with no arguments (or `-h`) to see its full subcommand list.

## Project structure

```
cli/
  *_cli.py           # argparse entry points, one per pipeline stage
  lib/
    keyword_search.py       # inverted index, TF/IDF, BM25
    semantic_search.py      # embeddings, cosine similarity, chunking
    hybrid_search.py        # score normalization, weighted fusion, RRF
    query_enhancement.py    # LLM spell-correct / rewrite / expand
    reranking.py            # LLM and cross-encoder re-ranking
    evaluation.py           # precision/recall/F1, LLM-as-judge
    augmented_generation.py # RAG, summarization, citations, Q&A
    multimodal_search.py    # CLIP-based image search
    search_utils.py         # shared types, constants, data loading
data/                 # dataset + stopwords + golden eval set (gitignored, see data/README.md)
cache/                # generated indexes/embeddings (gitignored, rebuilt on demand)
tests/                # pytest unit tests for the pure-function pieces
```

## Testing

```bash
uv run pytest
```

Tests cover the deterministic, dependency-free logic (tokenization, BM25 math, RRF/score fusion,
precision/recall/F1) — not the parts that require the embedding model or a live LLM call.

## Background

Built while working through boot.dev's "Learn Retrieval Augmented Generation" course, then
extended (multimodal image search) beyond the course material.
