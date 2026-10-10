"""Executable rewordings of real requests: new content words, the same rules and expected outcomes.

`router_scaling_paraphrase` changes only a request's frame, and the pinned router routes all
of those correctly. The study's held-out rewordings (`router_scaling_data.REAL_UNSEEN`)
also change content words ("handoff written up", "hold it"), and the pinned A3 router
sends 33 of 38 reworded drafting turns to the scheduling route. Those rewordings drop
rule clauses, so they cannot be scored as conversations. This module writes rewordings in the
same style that keep every rule clause verbatim, so a reworded real integration case keeps
its workspace and expected outcomes and can be served and scored end to end.

Only integration cases are reworded; the code refuses training, sealed and development splits.
The frames are disjoint from the paraphrase frames used to fit routers.
"""

import random

from neuroshard.evolution import router_scaling_paraphrase as paraphrase

DRAFT_REWORDS = (
    'Could you put together a draft for {project} to the {recipient}? Go by the newest approved delivery plan '
    'and skip any revision that is not approved. {rules} Keep it unsent and list the sources you used.',
    'I need the {project} handoff written up for the {recipient} from the most recent approved plan, not an '
    'unapproved one. {rules} Leave it as a saved draft with its sources cited; nothing goes out.',
    'Write up the {project} note for the {recipient}, working from the newest approved plan and ignoring '
    'unapproved revisions. {rules} Hold it as an unsent draft and mention which documents it relies on.',
)
MEETING_REWORDS = (
    'Find a window for a {duration}-minute {purpose} on {project} with {rest} Pencil it in as a draft only; '
    'send no invitations.',
    'Get a {duration}-minute {purpose} on the books for {project} with {rest} Leave it as a draft and send '
    'nothing.',
)
SPLITS = ('integration', 'cross-integration')


def reword(text, rng):
    """A content-word rewording of a real opening request with its rule clauses verbatim, or ``None``."""
    match = paraphrase.DRAFT_BASE.match(text)
    if match:
        return rng.choice(DRAFT_REWORDS).format(**match.groupdict())
    match = paraphrase.MEETING_BASE.match(text)
    if match:
        return rng.choice(MEETING_REWORDS).format(**match.groupdict())
    return None


def reworded(cases, seed):
    """Copies of real executable cases with the opening request reworded; workspace, turns and goals unchanged."""
    result = []
    for case in cases:
        if case.get('split') not in SPLITS:
            raise ValueError('only integration cases are reworded')
        rng = random.Random(f'{seed}:{case["id"]}')
        opening = case['turns'][0]['user']
        text = reword(opening, rng)
        if text is None:
            raise ValueError(f'no rewording for {case["id"]}')
        if not paraphrase.preserved(opening, text):
            raise ValueError(f'rewording lost a rule clause: {case["id"]}')
        turns = [{**case['turns'][0], 'user': text}, *case['turns'][1:]]
        result.append({**case, 'id': case['id'] + '-reworded', 'turns': turns, 'reworded_from': case['id']})
    return result
