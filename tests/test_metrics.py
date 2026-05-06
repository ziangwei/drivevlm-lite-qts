from drivevlm_lite.eval.metrics import accuracy, exact_match


def test_exact_match_normalizes_case_and_spacing():
    assert exact_match(" Keep   going ", "keep going") == 1.0


def test_accuracy():
    assert accuracy(["A", "B"], ["a", "C"]) == 0.5
