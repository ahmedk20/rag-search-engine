# Data

This directory is gitignored (the movie dataset is ~25MB) — nothing here is committed except this
file. To run the project, place the following alongside it:

## `movies.json` (required)

A JSON object with a single `movies` array:

```json
{
  "movies": [
    { "id": 1, "title": "Paddington", "description": "A young Peruvian bear..." },
    { "id": 2, "title": "...", "description": "..." }
  ]
}
```

`id` must be a unique integer per movie. This project was built against the movie dataset
supplied by boot.dev's "Learn Retrieval Augmented Generation" course — if you're following that
course, use the dataset it provides. Any dataset in the shape above works.

## `stopwords.txt` (required for keyword search)

Plain text, one stopword per line. Used by `cli/lib/keyword_search.py` to filter tokens before
indexing/BM25 scoring. Any standard English stopword list works.

## `golden_dataset.json` (required for `evaluation_cli.py`)

Hand-labeled query -> relevant-titles pairs used to compute precision@k/recall@k/F1:

```json
{
  "test_cases": [
    { "query": "cute british bear marmalade", "relevant_docs": ["Paddington"] }
  ]
}
```

`relevant_docs` entries must match `title` values in `movies.json` exactly.

## `paddington.jpeg` (optional)

Sample image used by `cli/multimodal_search_cli.py`'s examples. Any image works — pass its path
on the command line.
