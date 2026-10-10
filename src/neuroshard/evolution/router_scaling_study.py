"""The router scaling study's cases and texts, shared by the local runner and the Granite feature host.

Both must encode exactly the same texts, so the case construction lives here rather than in
either script. See docs/ROUTER_SCALING.md.
"""

from neuroshard.evolution import router_scaling_data as data
from neuroshard.evolution import router_scaling_paraphrase as paraphrase

PARAPHRASE_SEED = 66000
REAL_LIMIT = 48
CROSS_PER_PAIR = 1


def build(real_limit=REAL_LIMIT, cross_per_pair=CROSS_PER_PAIR):
    """Fit cases and the ``test`` and ``unseen`` evaluation sets."""
    fit = data.real_cases('fit', limit=real_limit) + data.cases('fit', cross_per_pair=cross_per_pair)
    test = data.real_cases('test') + data.cases('test', cross_per_pair=cross_per_pair)
    unseen = data.real_unseen_cases() + data.cases('unseen', cross_per_pair=cross_per_pair)
    return fit, {'test': test, 'unseen': unseen}


def paraphrase_sets(fit, sets):
    """Paraphrased copies of the real fit cases (for augmentation) and of the real test cases (a further check)."""
    real_fit = [case for case in fit if case['id'].startswith('real-')]
    real_test = [case for case in sets['test'] if case['id'].startswith('real-')]
    return (paraphrase.paraphrased_cases(real_fit, PARAPHRASE_SEED),
            paraphrase.paraphrased_cases(real_test, PARAPHRASE_SEED + 1))


def everything(real_limit=REAL_LIMIT, cross_per_pair=CROSS_PER_PAIR):
    """Every case the study routes, with its fit, evaluation and paraphrase sets."""
    fit, sets = build(real_limit, cross_per_pair)
    para_fit, para_test = paraphrase_sets(fit, sets)
    cases = fit + [case for group in sets.values() for case in group] + para_fit + para_test
    return cases, fit, sets, para_fit, para_test


def texts(real_limit=REAL_LIMIT, cross_per_pair=CROSS_PER_PAIR):
    """Every text the study encodes: all turns, paraphrases and unit descriptions, sorted."""
    cases = everything(real_limit, cross_per_pair)[0]
    return sorted({text for case in cases for text in case['user_turns']}
                  | {text for cards in data.DESCRIPTIONS.values() for text in cards})
