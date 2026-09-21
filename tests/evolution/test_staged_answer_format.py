import pytest

from neuroshard.evolution.staged_answer_format import answer_value


ADD = {"a": 1, "b": 5, "family": "addition"}
MOD = {"a": 8, "b": 16, "family": "modular-addition"}


@pytest.mark.parametrize("text, expected", [("6", "6"), (" 6.\n", "6"),
                                           ("1 + 5 = 6", "6"), ("1+5=7.", "7")])
def test_reads_complete_answers_without_computing_or_consulting_a_label(text, expected):
    assert answer_value(text, ADD, terminated=True) == expected


@pytest.mark.parametrize("text", ["1 + 5 =", "6 or 7", "6.5", "1 + 4 = 6", "5 + 1 = 6",
                                 "The answer is 6 but actually 7", "1 + 5 = 6\n7", "06", "6.."])
def test_rejects_partial_ambiguous_or_wrong_equation_answers(text):
    assert answer_value(text, ADD, terminated=True) is None


def test_generation_limit_cannot_turn_an_incomplete_answer_into_a_pass():
    assert answer_value("6", ADD, terminated=False) is None
    assert answer_value("1 + 5 = 6", ADD, terminated=False) is None


def test_modular_equation_must_preserve_the_operation_and_modulus():
    assert answer_value("(8 + 16) % 7 = 3", MOD, terminated=True) == "3"
    assert answer_value("(8 + 16) mod 7 = 3.", MOD, terminated=True) == "3"
    assert answer_value("8 + 16 = 24", MOD, terminated=True) is None
    assert answer_value("(8 + 16) % 6 = 0", MOD, terminated=True) is None
