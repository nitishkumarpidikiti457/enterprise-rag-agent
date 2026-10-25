"""Tests for the evaluation metrics (pure functions, no models needed)."""

from eval.run_eval import load_golden, overlap_match, score_ranking, to_markdown


def test_score_ranking():
    assert score_ranking([True, False, False]) == (1.0, 1.0, 1.0)      # correct at rank 1
    assert score_ranking([False, True, False]) == (0.0, 1.0, 0.5)      # rank 2 -> RR = 1/2
    assert score_ranking([False, False, False]) == (0.0, 0.0, 0.0)     # not found
    assert score_ranking([]) == (0.0, 0.0, 0.0)                        # no results


def test_overlap_match():
    assert overlap_match("within 2 hours of notification", "They must be pulled within 2 hours of notification [1].")
    assert not overlap_match("72 hours", "I don't know based on the available documents.")


def test_to_markdown_table():
    md = to_markdown([{"config": "dense", "hit@1": 0.9}, {"config": "hybrid", "hit@1": 1.0}])
    assert md.splitlines()[0] == "| config | hit@1 |"
    assert "| dense | 0.90 |" in md and "| hybrid | 1.00 |" in md


def test_golden_set_is_well_formed():
    golden = load_golden()
    assert len(golden) == 50
    assert all({"question", "answer", "source", "section"} <= g.keys() for g in golden)