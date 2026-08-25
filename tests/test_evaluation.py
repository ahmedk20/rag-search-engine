from lib.evaluation import f1_score, precision_at_k, recall_at_k


def test_precision_at_k_all_relevant():
    retrieved = ["A", "B", "C"]
    relevant = {"A", "B", "C", "D"}
    assert precision_at_k(retrieved, relevant, k=3) == 1.0


def test_precision_at_k_partial_match():
    retrieved = ["A", "X", "C"]
    relevant = {"A", "B", "C"}
    assert precision_at_k(retrieved, relevant, k=3) == 2 / 3


def test_recall_at_k_finds_fraction_of_relevant_docs():
    retrieved = ["A", "X", "Y"]
    relevant = {"A", "B"}
    assert recall_at_k(retrieved, relevant, k=3) == 0.5


def test_f1_score_combines_precision_and_recall():
    assert f1_score(precision=1.0, recall=1.0) == 1.0
    assert f1_score(precision=0.0, recall=0.0) == 0.0
    assert round(f1_score(precision=0.5, recall=0.5), 4) == 0.5
