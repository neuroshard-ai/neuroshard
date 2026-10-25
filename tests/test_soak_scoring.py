"""Soak scoring is narrow: a fact token and an expected source ID, matched exactly (review of PR #143)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import soak  # noqa: E402

QUESTION = soak.QUESTIONS[1][0]


def test_only_a_complete_real_document_id_counts():
    assert soak.grounded('61 units. (Sources: doc-88b36697f8)', QUESTION) is True
    assert soak.grounded('61 units. (Sources: doc-88b36697f8z)', QUESTION) is False  # not a complete ID
    assert soak.grounded('61 units. (Sources: doc-88b36697f8a1)', QUESTION) is False
    assert soak.grounded('61 units. (Sources: doc-88b36697f8, doc-292283f7a9)', QUESTION) is False
    assert soak.grounded('61 units. (Sources: doc-292283f7a9)', QUESTION) is False
