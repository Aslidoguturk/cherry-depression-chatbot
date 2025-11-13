import pytest
from chtr3 import get_phq_label


@pytest.mark.parametrize("score,expected", [
    (0, "Minimal or No Depression"),
    (4, "Minimal or No Depression"),
    (5, "Mild Depression"),
    (10, "Moderate Depression"),
    (15, "Severe Depression"),
    (24, "Severe Depression"),
])
def test_phq_scoring_logic(score, expected):
    assert get_phq_label(score) == expected
