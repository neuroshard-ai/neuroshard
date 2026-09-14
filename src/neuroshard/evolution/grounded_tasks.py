"""Generated, executable task fixtures for a bounded learning experiment.

These tasks measure grounded extraction, filtering, arithmetic and ordering.
They are not a general assistant benchmark or a native data-admission rule.
"""
import hashlib
import json
import math
import random

from .reference_data import identity


FAMILIES = ("lookup", "filter", "total", "sort")
CITIES = ("Oslo", "Lima", "Kyoto", "Accra", "Perth", "Riga", "Bern", "Suva")


def make_case(seed, role, index, family=None):
    rng = random.Random(int(hashlib.sha256(f"{seed}:{role}:{index}".encode()).hexdigest(), 16))
    chosen = FAMILIES[index % len(FAMILIES)] if family is None else family
    if chosen not in FAMILIES:
        raise ValueError("Unknown grounded task family")
    family = chosen
    identifiers = rng.sample(range(100, 1000), 5)
    rows = [{"id": f"{chr(65 + rng.randrange(26))}{value}", "city": rng.choice(CITIES),
             "units": rng.randrange(0, 20), "price": rng.randrange(1, 20),
             "status": rng.choice(("ready", "pending", "closed")), "priority": rng.randrange(1, 10)}
            for value in identifiers]
    query = rng.choice(rows)["id"] if index % 5 else "Z000"
    variant = (index // len(FAMILIES)) % (4 if role == "test" else 2)
    return {"family": family, "rows": rows, "query": query, "variant": variant}


def expected(case):
    rows, family = case["rows"], case["family"]
    if family == "lookup":
        matches = [row for row in rows if row["id"] == case["query"]]
        return {"city": matches[0]["city"] if matches else None,
                "units": matches[0]["units"] if matches else None}
    if family == "filter":
        return {"ids": sorted(row["id"] for row in rows if row["status"] == "ready")}
    if family == "total":
        return {"total": sum(row["units"] * row["price"] for row in rows)}
    if family == "sort":
        ranked = sorted(rows, key=lambda row: (-row["priority"], row["id"]))
        return {"ids": [row["id"] for row in ranked[:2]]}
    raise ValueError("Unknown grounded task")


def prompt(case):
    family, variant, query = case["family"], case["variant"], case["query"]
    instructions = {
        "lookup": [
            f'Find record {query}. Return its city and units as JSON with exactly keys "city" and "units". Use null for both if the record is absent.',
            f'Look up id {query} in the supplied records. Reply with {{"city": city, "units": units}} only; absent ids require two null values.',
            f'Which city and unit count belong to {query}? Produce only a JSON object with keys "city" and "units"; both must be null if no id matches.',
            f'Extract city and units for id={query}. Required output schema: {{"city": string or null, "units": integer or null}}. If missing, return null in both fields.',
        ],
        "filter": [
            'Select records whose status is exactly "ready". Return {"ids": [...]} with their ids in ascending alphabetical order.',
            'Reply with one JSON key "ids": the alphabetically sorted ids of all ready records. Exclude pending and closed records.',
            'Keep only entries marked ready, then alphabetize their identifiers. Output a JSON object containing only the array "ids".',
            'List every ready id, sorted lexicographically ascending. The answer must have the shape {"ids": [...]}.',
        ],
        "total": [
            'For each record multiply units by price, then add the products across all records. Return only {"total": integer}.',
            'Calculate the total invoice value: sum(units * price) for every entry, irrespective of status. Answer as JSON with exactly one integer key "total".',
            'What is the combined value of all entries? Each contributes its unit count times its price. Output {"total": integer} without explanation.',
            'Compute the sum of quantity-times-price using units as quantity for all listed rows. Use only the JSON field "total", with an integer value.',
        ],
        "sort": [
            'Choose the two records with highest priority. Larger priority numbers rank first; break ties by ascending id. Return their ids in ranking order as {"ids": [...]}.',
            'Rank rows by descending priority, then ascending identifier for equal priorities. Output the first two ids under the JSON key "ids".',
            'Which two identifiers lead when priority is sorted highest first and tied ids alphabetically? Return {"ids": [...]} in that order.',
            'Take the top two records using priority descending and id ascending as the tie-break. Answer with only a JSON object containing their ordered "ids".',
        ],
    }
    fields = {"lookup": ("id", "city", "units"), "filter": ("id", "status"),
              "total": ("id", "units", "price"), "sort": ("id", "priority")}[family]
    visible = [{key: row[key] for key in fields} for row in case["rows"]]
    return ("Use only the records below. Return a single JSON object, without Markdown or commentary.\n"
            + instructions[family][variant] + "\nRecords:\n" + json.dumps(visible, separators=(",", ":")))


def parse_answer(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("Nonfinite JSON constant")
    if not isinstance(text, str) or len(text.encode()) > 32768:
        raise ValueError("Invalid answer size")
    value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(value, dict):
        raise ValueError("Expected one JSON object")
    return value


def check_answer(case, text):
    target = expected(case)
    try:
        answer = parse_answer(text)
    except (ValueError, TypeError):
        return {"correct": False, "valid_json_object": False, "reason": "invalid_json_object"}
    # Python equality otherwise treats True as 1 and 1.0 as an integer.
    exact = (set(answer) == set(target)
             and all(type(answer[key]) is type(target[key]) and answer[key] == target[key] for key in target))
    return {"correct": exact, "valid_json_object": True,
            "reason": "correct" if exact else "wrong_schema_or_value"}


def damaged_target(case):
    """Deliberately false labels for an explicit laboratory control arm."""
    value = expected(case)
    if case["family"] == "lookup":
        value["units"] = 1 if value["units"] is None else value["units"] + 1
    elif case["family"] == "total":
        value["total"] += 1
    else:
        value["ids"] = value["ids"] + ["Z000"]
    return json.dumps(value, separators=(",", ":"))


def paired_accuracy(before, after):
    if not before or [r["id"] for r in before] != [r["id"] for r in after]:
        raise ValueError("Require aligned nonempty task results")
    if len({r["id"] for r in before}) != len(before):
        raise ValueError("Repeated evaluation identity")
    wins = sum(not a["correct"] and b["correct"] for a, b in zip(before, after))
    losses = sum(a["correct"] and not b["correct"] for a, b in zip(before, after))
    discordant = wins + losses
    # Exact one-sided paired sign/McNemar test under equiprobable discordance.
    probability = sum(math.comb(discordant, k) for k in range(wins, discordant + 1)) / 2**discordant
    return {"documents": len(before), "baseline_correct": sum(r["correct"] for r in before),
            "candidate_correct": sum(r["correct"] for r in after), "wins": wins, "losses": losses,
            "accuracy_change": (wins - losses) / len(before), "one_sided_p": probability}


def task_identity(case):
    return identity({"task": case, "prompt": prompt(case)})
