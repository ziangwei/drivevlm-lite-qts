from drivevlm_lite.eval.metrics import accuracy, exact_match, relaxed_exact_match, token_f1, yes_no_match


def test_exact_match_normalizes_case_and_spacing():
    assert exact_match(" Keep   going ", "keep going") == 1.0


def test_accuracy():
    assert accuracy(["A", "B"], ["a", "C"]) == 0.5


def test_relaxed_exact_match_ignores_simple_punctuation():
    assert relaxed_exact_match("No.", "No") == 1.0


def test_token_f1_scores_partial_overlap():
    assert 0.0 < token_f1("keep going straight", "keep going") < 1.0


def test_yes_no_match():
    assert yes_no_match("No.", "No") == 1.0
    assert yes_no_match("Yes.", "No") == 0.0
    assert yes_no_match("Keep going", "Keep going") is None
