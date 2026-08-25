from lib.keyword_search import InvertedIndex, preprocess_text, tokenize_text


def test_preprocess_text_lowercases_and_strips_punctuation():
    assert preprocess_text("Hello, World!") == "hello world"


def test_tokenize_text_removes_stopwords_and_stems():
    tokens = tokenize_text("The bears were running quickly")
    assert "the" not in tokens
    assert "were" not in tokens
    assert "run" in tokens  # PorterStemmer: running -> run


def test_tokenize_text_empty_string():
    assert tokenize_text("") == []


def _build_index() -> InvertedIndex:
    idx = InvertedIndex()
    idx.docmap = {
        1: {"id": 1, "title": "Paddington", "description": "A bear moves to London"},
        2: {"id": 2, "title": "Ted", "description": "A talking teddy bear comedy"},
        3: {"id": 3, "title": "Heat", "description": "A crew plans a bank heist"},
    }
    for doc_id, doc in idx.docmap.items():
        tokens = tokenize_text(f"{doc['title']} {doc['description']}")
        for token in set(tokens):
            idx.index[token].add(doc_id)
        from collections import Counter

        idx.term_frequencies[doc_id] = Counter(tokens)
        idx.doc_lengths[doc_id] = len(tokens)
    return idx


def test_get_documents_returns_matching_doc_ids():
    idx = _build_index()
    bear_term = tokenize_text("bear")[0]
    assert idx.get_documents(bear_term) == [1, 2]


def test_idf_is_lower_for_common_terms():
    idx = _build_index()
    bear_term = tokenize_text("bear")[0]  # appears in 2/3 docs
    heist_term = tokenize_text("heist")[0]  # appears in 1/3 docs
    assert idx.get_idf(heist_term) > idx.get_idf(bear_term)


def test_bm25_search_ranks_matching_doc_first():
    idx = _build_index()
    results = idx.bm25_search("bank heist", limit=3)
    assert results[0]["title"] == "Heat"
    assert results[0]["score"] > 0
