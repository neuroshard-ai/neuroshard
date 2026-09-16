"""Committed input scopes for specialists with a finite knowledge domain.

An eligibility scope contains subject names from training inputs, never their
answers. It restricts the learned classifier before selection and preserves
the general fallback. A scope is a routing constraint, not evidence of quality.
"""
import json
import re
import unicodedata

from .schema import root


def words(text):
    return re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold())


def validate(scopes, routes, fallback):
    if not isinstance(scopes, dict) or not set(scopes) <= set(routes) - {fallback}:
        raise ValueError('Scopes may restrict installed specialists, never the fallback')
    for rule in scopes.values():
        if not isinstance(rule, dict) or set(rule) not in ({'input_root', 'subjects'},
                {'input_root', 'subjects', 'record_fields'}):
            raise ValueError('Bind the training input root and finite subject inventory')
        root(rule['input_root'])
        subjects = rule['subjects']
        if (not isinstance(subjects, list) or not 1 <= len(subjects) <= 4096
                or any(not isinstance(subject, str) or not subject.strip()
                       or len(subject.encode()) > 256 or not words(subject) for subject in subjects)):
            raise ValueError('Require bounded nonempty subject names')
        normalized = [tuple(words(subject)) for subject in subjects]
        if len(set(normalized)) != len(normalized):
            raise ValueError('Duplicate normalized scope subject')
        if 'record_fields' in rule:
            schemas = rule['record_fields']
            if (not isinstance(schemas, list) or not 1 <= len(schemas) <= 16
                    or any(not isinstance(fields, list) or not 1 <= len(fields) <= 32
                        or any(not isinstance(name, str) or not name or len(name) > 128 for name in fields)
                        or len(set(fields)) != len(fields) for fields in schemas)):
                raise ValueError('Require bounded JSON record field schemas')
    return scopes


def matching_records(text, schemas):
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != '[':
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except (ValueError, RecursionError):
            continue
        if (isinstance(value, list) and 1 <= len(value) <= 128
                and any(all(isinstance(row, dict) and set(row) == set(fields) for row in value)
                        for fields in schemas)):
            return True
    return False


def eligible(question, scopes, routes, fallback):
    validate(scopes, routes, fallback)
    if not isinstance(question, str) or not question.strip() or len(question.encode()) > 32768:
        raise ValueError('Require bounded input for specialist eligibility')
    tokens = words(question)
    allowed = set(routes) - set(scopes)
    for route, rule in scopes.items():
        if 'record_fields' in rule and not matching_records(question, rule['record_fields']):
            continue
        for subject in rule['subjects']:
            needle = words(subject)
            if any(tokens[start:start+len(needle)] == needle for start in range(len(tokens)-len(needle)+1)):
                allowed.add(route)
                break
    return sorted(allowed)
