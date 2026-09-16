"""Answer-blind input augmentation for the existing router's development data.

The first router learned formatted expert requests. These transformations expose
the actual user questions without the domain and output-format decorations. This
is fitting data, never an inference-time rule or an answer lookup.
"""
import re

from .incremental_facts import QUESTIONS

FORMAT = 'neuroshard-raw-router-data-v1'

PROTOCOL_PREFIXES = (
    'NeuroShard 0.4.0 public profile. ', 'About NeuroShard 0.4.0: ',
    'NeuroShard 0.4.0 question:\n', 'For the NeuroShard 0.4.0 release, answer this briefly:\n',
    'Regarding NeuroShard 0.4.0, ', 'NeuroShard 0.4.0 — ', 'NeuroShard 0.4.0: ',
)
PROTOCOL_SUFFIXES = (
    ' Reply with only the short answer.', ' Return the answer without explanation.',
    '\nGive a concise answer only.', '\nOnly the requested answer.',
    ' Please provide just the answer.', ' Give just the answer.',
    ' Reply with the two short answers in order.',
)


def raw_questions(row, route):
    """Return input-only variants; labels, references and expected values unused.

    Directory entity names and requested attributes are input provenance. The
    known question templates provide forms for *every* selected entity on its
    existing split; an entity cannot cross sides through augmentation.
    """
    original = '\n\n'.join(message['content'] for message in row['messages']
                            if message['role'] == 'user')
    if not original.strip():
        raise ValueError('A routing example needs user text')
    if route == 'parent':
        return [original]
    if route == 'directory':
        task = row['task']
        name, attribute = task['name'], task['attribute']
        if (not isinstance(name, str) or name not in original or attribute not in QUESTIONS
                or not re.fullmatch(r'[A-Za-z]+(?: [A-Za-z]+){1,3}', name)):
            raise ValueError('Directory augmentation must bind actual input entities')
        forms = [template.format(name=name) for template in QUESTIONS[attribute]]
    elif route == 'protocol':
        bare = original
        for prefix in PROTOCOL_PREFIXES:
            if bare.startswith(prefix):
                bare = bare[len(prefix):]
                break
        for suffix in PROTOCOL_SUFFIXES:
            if bare.endswith(suffix):
                bare = bare[:-len(suffix)]
                break
        if bare == original or not bare.strip():
            raise ValueError('Unknown protocol training decoration; do not silently keep it')
        # A naked question without its product can be ambiguous. Include the
        # explicit product context as well; no answer or task ID is appended.
        forms = [bare, 'For NeuroShard, ' + bare,
                 bare.replace('the client', 'the NeuroShard client')]
    else:
        raise ValueError('Unknown route in this development data migration')
    return list(dict.fromkeys([original, *forms]))
