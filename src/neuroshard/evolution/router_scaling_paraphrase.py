"""Meaning-preserving paraphrases of the real drafting and scheduling requests, for router fitting.

The [router scaling study](../../../docs/ROUTER_SCALING.md) found that a router fitted on
the real grammars recalls reworded drafting requests 0.18-0.34 of the time: each grammar
phrases its capability one way, so the router learns the template. The cheapest measured
lever was a second fit phrasing. This module supplies those phrasings for the two real
capabilities without changing what any request asks for.

A paraphrase rewrites only the opening frame of a request and its closing instruction. The
clauses that carry its meaning (project, recipient, plan revision rule, due-date and total
rules, attendees, duration, date and time constraints) are kept verbatim, so the case's
expected outcome, which the scorer checks, is unchanged and the same paraphrased turns can
later be served end to end. Follow-up corrections are kept as written: they are routed by
the conversation's opening, which the context router sees.

The frames here are disjoint from the study's held-out reworded phrasings
(`router_scaling_data.REAL_UNSEEN`), so fitting on them never touches the turns that
measure generalisation. Only training and integration splits are paraphrased; the code
refuses sealed and development splits.
"""

import random
import re

DRAFT_BASE = re.compile(r'^Create a draft for (?P<project>.+?), addressed to the (?P<recipient>[a-z ]+?)\. '
                        r'Use the latest approved delivery plan, not an unapproved revision\. (?P<rules>.+?)'
                        r' Save a draft only, cite the documents used, and do not send anything\.$')
MEETING_BASE = re.compile(r'^Schedule a (?P<duration>\d+)-minute (?P<purpose>[a-z ]+?) for (?P<project>.+?) with '
                          r'(?P<rest>.+?) Save a meeting draft only and do not send anything\.$')

DRAFT_FRAMES = (
    'Please prepare a draft for {project} that goes to the {recipient}. Base it on the latest approved delivery '
    'plan rather than any unapproved revision. {rules} Keep it as an unsent draft and cite the documents you used.',
    'I need a draft written for {project}, for the {recipient}. Work from the most recent approved delivery plan, '
    'not an unapproved revision. {rules} Only save the draft, list the documents you relied on, and send nothing.',
    'Set up a draft for {project} addressed to the {recipient}, using the latest approved version of the delivery '
    'plan and ignoring unapproved revisions. {rules} Do not send it; just save it with the documents cited.',
    'For {project}, put together a draft to the {recipient}. Take the newest delivery plan that is approved, not '
    'a pending revision. {rules} It should stay a saved, unsent draft citing its sources.',
)
MEETING_FRAMES = (
    'Please book a {duration}-minute {purpose} for {project} with {rest} Keep it as a meeting draft and do not send '
    'anything.',
    'Can you set up a {duration}-minute {purpose} for {project} with {rest} Only save the meeting as a draft; send '
    'nothing.',
    'Arrange a {duration}-minute {purpose} for {project} with {rest} Save it as a draft meeting without sending '
    'invitations.',
    'Put a {duration}-minute {purpose} for {project} on the calendar with {rest} Leave it as an unsent meeting '
    'draft.',
)
SEALED_HINTS = ('confirmation', 'development')


def paraphrase(text, rng):
    """One paraphrase of a real opening request, or ``None`` when its form is not one this module rewrites."""
    match = DRAFT_BASE.match(text)
    if match:
        return rng.choice(DRAFT_FRAMES).format(**match.groupdict())
    match = MEETING_BASE.match(text)
    if match:
        return rng.choice(MEETING_FRAMES).format(**match.groupdict())
    return None


def preserved(original, rewritten):
    """The meaning-carrying clauses of ``original`` that ``rewritten`` must contain verbatim."""
    for pattern, keys in ((DRAFT_BASE, ('project', 'recipient', 'rules')),
                          (MEETING_BASE, ('duration', 'purpose', 'project', 'rest'))):
        match = pattern.match(original)
        if match:
            return all(match.group(key) in rewritten for key in keys)
    return False


def paraphrased_cases(cases, seed, copies=1):
    """Copies of real study cases whose opening request is paraphrased; other turns and labels are unchanged.

    ``cases`` are `router_scaling_data.real_cases` records. Each copy gets a distinct id, so a
    router can be fitted on originals and paraphrases together.
    """
    result = []
    for case in cases:
        if any(hint in case['id'] for hint in SEALED_HINTS):
            raise ValueError('paraphrases are only made from training and integration splits')
        for copy in range(copies):
            rng = random.Random(f'{seed}:{case["id"]}:{copy}')
            rewritten = paraphrase(case['user_turns'][0], rng)
            if rewritten is None:
                continue
            if not preserved(case['user_turns'][0], rewritten):
                raise ValueError(f'paraphrase lost a meaning-carrying clause: {case["id"]}')
            result.append({**case, 'id': f'{case["id"]}-para{copy}', 'user_turns': [rewritten, *case['user_turns'][1:]],
                           'paraphrase_of': case['id']})
    return result
