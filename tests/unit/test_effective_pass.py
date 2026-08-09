"""Task 6: Effective pass tests."""

from homechef_booking.evaluation.effective_pass import effective_pass


def test_effective_pass_threshold():
    assert effective_pass(True, 0.95, False) is True
    assert effective_pass(True, 0.949, False) is False
    assert effective_pass(False, 1.0, False) is False
    assert effective_pass(True, 1.0, True) is False
