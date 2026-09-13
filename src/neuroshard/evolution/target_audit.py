"""Report mechanical failures of a narrow set of instruction templates.

This is a data-review aid, not a general instruction parser, a factual-quality
judge, or a native admission rule. A template may occur inside quoted text;
operators must inspect its context before rejecting or repairing a record.
"""
import re


FORMAT = "neuroshard-instruction-target-audit-v1"
WORD_MINIMUM = re.compile(
    r'In your response, the word "([A-Za-z][A-Za-z-]{0,79})" '
    r'should appear at least ([0-9]{1,4}) times\.'
)
WORD_LIMIT = re.compile(r"Your response should contain less than ([0-9]{1,4}) words\.")
QUOTED_ENDING = re.compile(r'Finish your response with this exact phrase "([^"\n]{1,200})"')
BULLET_COUNT = re.compile(r"The response must contain exactly ([0-9]{1,3}) bullet points\.")
KEYWORDS = re.compile(r"Include keywords \[([A-Za-z ,'-]{1,200})\] in the response\.")


def word_occurrences(answer, word):
    """Case-insensitive whole-word matching; 'happiness' is not 'happy'."""
    return len(re.findall(r"(?<!\w)" + re.escape(word.casefold()) + r"(?!\w)", answer.casefold()))


def audit_answer(instruction, answer):
    checks = []

    def record(kind, expected, observed, passes, matched):
        checks.append({"kind": kind, "expected": expected, "observed": observed,
                       "passes": passes, "matched_instruction": matched.group(0)})

    for match in WORD_MINIMUM.finditer(instruction):
        word, minimum = match[1], int(match[2])
        count = word_occurrences(answer, word)
        record("word_minimum", {"word": word, "minimum": minimum}, count, count >= minimum, match)
    for match in WORD_LIMIT.finditer(instruction):
        limit, count = int(match[1]), len(answer.split())
        record("word_limit", {"exclusive_limit": limit, "method": "whitespace-separated words"},
               count, count < limit, match)
    for match in QUOTED_ENDING.finditer(instruction):
        expected = match[1]
        observed = answer.rstrip()[-len(expected):]
        record("quoted_ending", expected, observed, answer.rstrip().endswith(expected), match)
    for match in BULLET_COUNT.finditer(instruction):
        expected = int(match[1])
        # Count all Markdown unordered-list markers, including indented ones.
        observed = len(re.findall(r"(?m)^[ \t]*[-*+][ \t]+\S", answer))
        record("bullet_count", expected, observed, observed == expected, match)
    for match in KEYWORDS.finditer(instruction):
        words = [word.strip() for word in match[1].split(",")]
        if not words or any(not re.fullmatch(r"[A-Za-z][A-Za-z'-]*", word) for word in words):
            continue
        # Corpus templates sometimes bind [keywords] in a later sentence.
        # Treating that variable name as a required literal creates false
        # failures. Resolving template variables is outside this parser.
        if any(word.casefold() in {"keyword", "keywords"} for word in words):
            continue
        counts = {word: word_occurrences(answer, word) for word in words}
        record("keywords", words, counts, all(counts.values()), match)
    return checks


def audit_conversation(messages):
    if not isinstance(messages, list) or not 1 <= len(messages) <= 128:
        raise ValueError("Expected a bounded conversation")
    for message in messages:
        if (not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant")
                or not isinstance(message.get("content"), str)
                or len(message["content"].encode()) > 256 * 1024):
            raise ValueError("Invalid or oversized message")
    system = messages[0]["content"] if messages[0]["role"] == "system" else ""
    checks = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant" or index == 0 or messages[index - 1]["role"] != "user":
            continue
        # Preserve both origins. Conflicting instructions require review; this
        # deliberately does not try to resolve instruction precedence.
        for origin, instruction in (("system", system), ("user", messages[index - 1]["content"])):
            checks.extend({**check, "assistant_turn": index, "instruction_origin": origin}
                          for check in audit_answer(instruction, message["content"]))
    return checks
