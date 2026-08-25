from lib.hybrid_search import combine_search_results, hybrid_score, normalize_scores, rrf_score
from lib.search_utils import format_search_result


def test_normalize_scores_scales_to_zero_one():
    assert normalize_scores([10.0, 20.0, 30.0]) == [0.0, 0.5, 1.0]


def test_normalize_scores_empty_list():
    assert normalize_scores([]) == []


def test_normalize_scores_all_equal():
    assert normalize_scores([5.0, 5.0, 5.0]) == [1.0, 1.0, 1.0]


def test_hybrid_score_alpha_zero_is_all_semantic():
    assert hybrid_score(bm25_score=1.0, semantic_score=0.4, alpha=0.0) == 0.4


def test_hybrid_score_alpha_one_is_all_bm25():
    assert hybrid_score(bm25_score=0.7, semantic_score=1.0, alpha=1.0) == 0.7


def test_rrf_score_decreases_with_rank():
    assert rrf_score(rank=1, k=60) > rrf_score(rank=2, k=60)


def test_combine_search_results_favors_doc_present_in_both():
    bm25_results = [
        format_search_result(1, "A", "doc a", score=10.0),
        format_search_result(2, "B", "doc b", score=5.0),
    ]
    semantic_results = [
        format_search_result(1, "A", "doc a", score=0.9),
        format_search_result(3, "C", "doc c", score=0.5),
    ]

    combined = combine_search_results(bm25_results, semantic_results, alpha=0.5)

    assert combined[0]["id"] == 1  # top of both lists
    assert {r["id"] for r in combined} == {1, 2, 3}
